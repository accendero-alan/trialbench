"""T27 -- The mapping-noise floor.

disease-representation-test-plan.md. Every code-derived arm (T21, T23, T24,
T25's arm (c)/(d)) inherits TrialBench's lexical string->ICD mapping (NLM
Clinical Tables API on the ``condition`` strings). The best published
normalizer of CTGov condition strings resolves 86.74%, so a ~13% error floor
may cap every code result and could explain a null anywhere above.

(a) Re-map every unique ``condition`` string via SapBERT nearest-neighbor
    over the official ICD-10-CM code-description index -- reusing
    ``src.data.icd10_hierarchy._load_full_descriptors`` (the same pinned
    Tabular List XML P9 already parses; 46,881 full codes with an official
    descriptor, per P9/P11 standing rule 10) as that index, rather than
    fetching a second copy of the same data under a different name.
(b) Measure agreement between the remap and the shipped ``icdcode`` column
    at chapter/char3/full levels, per trial (Jaccard over that trial's
    rolled-up term sets -- the shipped column has no clean per-string
    alignment with ``condition`` to lean on instead: verified empirically,
    ~11% of trials with any code have condition-list length != icdcode
    group count even after dropping unmapped/None slots, so this is a
    trial-level comparison, not a string-level one).
(c) A stratified 100-trial disagreement sample is written to
    ``results/experiments/t27_disagreement_sample.csv`` for hand adjudication
    -- genuinely not automatable (the plan calls it "an evening"); this
    script produces the sample, not the taxonomy.
(d) Re-run the best (rung, method) from T23 with re-mapped codes in place of
    shipped ones, on the 4 cells with the lowest char3-level agreement, 5
    seeds, paired against the existing shipped-code run already on disk (no
    need to refit that side).

Usage: `python -m experiments.t27_mapping_noise` (after T23's artifact
exists). The SapBERT encode of the 46,881-code index plus every unique
condition string is cached per-string under results/cache/sapbert/ (P11) --
a re-run is nearly free.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.data.ccsr import icd10_ccsr
from src.data.features import CodeFeaturizer, TabularFeaturizer, _icd_char3, _icd_level_terms, _recursive_parse_terms
from src.data.icd10_hierarchy import _load_full_descriptors
from src.data.loader import TASKS, load_task_phase
from src.data.mol_features import vocab_matrix
from src.eval import metrics as M
from src.eval.pooled_bootstrap import pool_predictions, pooled_paired_bootstrap
from src.eval.predictions import load_predictions
from src.methods.registry import get as get_method
from src.methods.text_nlp import SapBERTEncoder

LEVELS = ("chapter", "char3", "full")
SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"
ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]
RUNG_DIRS = {
    "char3": "results_codes", "chapter": "results_codes_chapter",
    "block": "results_codes_block", "full": "results_codes_full", "ccsr": "results_codes_ccsr",
}
T23_ARTIFACT = "results/experiments/t23_granularity_ladder.json"
OUT_PATH = "results/experiments/t27_mapping_noise.json"
SAMPLE_CSV = "results/experiments/t27_disagreement_sample.csv"
REMAP_RESULTS_DIR = "results_t27_remapped"
N_HAND_ADJUDICATE = 100
N_RERUN_CELLS = 4


# ---------------------------------------------------------------------------
# The official index + SapBERT nearest-neighbor remap.
# ---------------------------------------------------------------------------
def _load_official_index():
    d = _load_full_descriptors()
    codes = sorted(d.keys())
    descs = [d[c] for c in codes]
    return codes, descs


def _encode_official_index(encoder: SapBERTEncoder, descs: list) -> np.ndarray:
    vecs_by_str = encoder.encode_strings(descs)
    mat = np.stack([vecs_by_str[d] for d in descs]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _extend_remap(remap_dict: dict, cond_strings: list, encoder: SapBERTEncoder,
                   codes: list, code_mat_normed: np.ndarray) -> None:
    new = sorted(set(cond_strings) - set(remap_dict.keys()))
    if not new:
        return
    vecs_by_str = encoder.encode_strings(new)
    for s in new:
        v = vecs_by_str[s].astype(np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            remap_dict[s] = None
            continue
        sims = code_mat_normed @ (v / n)
        remap_dict[s] = codes[int(np.argmax(sims))]


def _condition_terms(v) -> list:
    return sorted(set(_recursive_parse_terms(v)))


def _remapped_level_terms(cond_terms: list, remap_dict: dict, level: str) -> set:
    codes = [remap_dict[t] for t in cond_terms if remap_dict.get(t)]
    if not codes:
        return set()
    if level == "ccsr":  # _icd_level_terms doesn't handle ccsr -- separate HCUP lookup, per CodeFeaturizer
        return {t for c in codes for t in icd10_ccsr(c)}
    return _icd_level_terms(codes, level)


def _shipped_level_terms(icdcode_value, level: str) -> set:
    codes = [c for c in _recursive_parse_terms(icdcode_value) if c.strip()]
    return _icd_level_terms(codes, level)


def _jaccard(a: set, b: set):
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Part (d): CodeFeaturizer-equivalent multi-hot built from the remap instead
# of the shipped icdcode column, at a fixed single granularity.
# ---------------------------------------------------------------------------
def _remapped_code_matrix(X_fit: pd.DataFrame, X_list: list, remap_dict: dict, level: str, min_df: int = 10):
    """Fit a min_df vocabulary on ``X_fit`` (train) and transform every frame
    in ``X_list`` -- mirrors CodeFeaturizer's fit/transform contract exactly,
    just sourced from the remap instead of ``icdcode``."""
    def terms_for(X):
        cond_col = X["condition"] if "condition" in X.columns else pd.Series([None] * len(X), index=X.index)
        return [_remapped_level_terms(_condition_terms(v), remap_dict, level) for v in cond_col.values]

    fit_terms = terms_for(X_fit)
    counts = Counter(t for terms in fit_terms for t in terms)
    vocab = sorted(t for t, c in counts.items() if c >= min_df)
    out = []
    for X in X_list:
        terms = fit_terms if X is X_fit else terms_for(X)
        mh = vocab_matrix(terms, vocab)
        n_terms = np.array([len(t) for t in terms], dtype=float).reshape(-1, 1)
        has = (n_terms > 0).astype(float)
        out.append(np.hstack([mh, n_terms, has]))
    return out


def main():
    with open(T23_ARTIFACT) as f:
        t23 = json.load(f)
    if "optimum_rung_per_cell_valid" not in t23:
        raise KeyError(
            "T23's artifact has no optimum_rung_per_cell_valid -- it predates the A2 fix. "
            "Re-run `python -m experiments.t23_granularity_ladder` first."
        )
    best_rung_per_task, best_method_valid_by_cell = {}, {}
    for row in t23["optimum_rung_per_cell_valid"]:
        best_rung_per_task.setdefault(row["task"], []).append(row["optimum_rung"])
    for task, rungs in best_rung_per_task.items():
        non_null = [r for r in rungs if r is not None]
        best_rung_per_task[task] = Counter(non_null).most_common(1)[0][0] if non_null else "char3"
    print("best_rung_per_task:", best_rung_per_task, flush=True)

    encoder = SapBERTEncoder()
    codes, descs = _load_official_index()
    print(f"official index: {len(codes)} codes with descriptors", flush=True)

    with Timer() as t:
        code_mat_normed = _encode_official_index(encoder, descs)
        print("encoded official index", flush=True)

        remap_dict: dict = {}
        cell_agreement_rows, per_trial_rows = [], []
        for task, phase in ALL_CELLS:
            td = load_task_phase(DATA_ROOT, task, phase, seed=42)
            cond_col = (td.X_test["condition"] if "condition" in td.X_test.columns
                        else pd.Series([None] * len(td.X_test), index=td.X_test.index))
            icd_col = (td.X_test["icdcode"] if "icdcode" in td.X_test.columns
                       else pd.Series([None] * len(td.X_test), index=td.X_test.index))
            all_terms = [_condition_terms(v) for v in cond_col.values]
            _extend_remap(remap_dict, [s for terms in all_terms for s in terms], encoder, codes, code_mat_normed)

            per_level_jaccards = {lvl: [] for lvl in LEVELS}
            for nct_id, terms, icd_val in zip(td.X_test.index, all_terms, icd_col.values):
                row = {"task": task, "phase": phase, "nct_id": str(nct_id),
                       "condition_terms": "; ".join(terms)}
                for lvl in LEVELS:
                    shipped = _shipped_level_terms(icd_val, lvl)
                    remapped = _remapped_level_terms(terms, remap_dict, lvl)
                    j = _jaccard(shipped, remapped)
                    row[f"{lvl}_shipped"] = ",".join(sorted(shipped))
                    row[f"{lvl}_remapped"] = ",".join(sorted(remapped))
                    row[f"{lvl}_jaccard"] = j
                    if j is not None:
                        per_level_jaccards[lvl].append(j)
                per_trial_rows.append(row)

            cell_agreement_rows.append({
                "task": task, "phase": phase,
                **{f"mean_jaccard_{lvl}": (float(np.mean(per_level_jaccards[lvl]))
                                            if per_level_jaccards[lvl] else None) for lvl in LEVELS},
                **{f"n_scored_{lvl}": len(per_level_jaccards[lvl]) for lvl in LEVELS},
            })
            print(f"  {task}/{phase}: mean char3 jaccard = "
                  f"{cell_agreement_rows[-1]['mean_jaccard_char3']}", flush=True)

        # ---- (c): stratified 100-trial hand-adjudication sample, char3 level ----
        disagreements = [r for r in per_trial_rows if r["char3_jaccard"] is not None and r["char3_jaccard"] < 1.0]
        rng = np.random.default_rng(42)
        if disagreements:
            jvals = np.array([r["char3_jaccard"] for r in disagreements])
            bin_edges = np.quantile(jvals, [0, 0.25, 0.5, 0.75, 1.0])
            bin_idx = np.clip(np.digitize(jvals, bin_edges[1:-1], right=True), 0, 3)
            sample_rows = []
            per_bin_quota = max(1, N_HAND_ADJUDICATE // 4)
            for b in range(4):
                pool = [i for i in range(len(disagreements)) if bin_idx[i] == b]
                take = rng.choice(pool, size=min(per_bin_quota, len(pool)), replace=False) if pool else []
                sample_rows.extend(disagreements[i] for i in take)
            sample_rows = sample_rows[:N_HAND_ADJUDICATE]
        else:
            sample_rows = []
        os.makedirs(os.path.dirname(SAMPLE_CSV), exist_ok=True)
        if sample_rows:
            with open(SAMPLE_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()) + ["adjudication_note"])
                w.writeheader()
                for r in sample_rows:
                    w.writerow({**r, "adjudication_note": ""})
        print(f"wrote {len(sample_rows)}-row disagreement sample to {SAMPLE_CSV} (hand adjudication is manual, "
              f"not performed by this script)", flush=True)

        # ---- (d): re-run the 4 lowest-char3-agreement cells with remapped codes ----
        cell_agreement_df = pd.DataFrame(cell_agreement_rows).dropna(subset=["mean_jaccard_char3"])
        rerun_cells = (cell_agreement_df.sort_values("mean_jaccard_char3").head(N_RERUN_CELLS)
                       [["task", "phase", "mean_jaccard_char3"]].to_dict("records"))
        print("re-run cells (lowest char3 agreement):", rerun_cells, flush=True)

        rerun_results = []
        for cell in rerun_cells:
            task, phase = cell["task"], cell["phase"]
            best_rung = best_rung_per_task.get(task, "char3")
            # A2-style: best method for this (task, phase, rung) selected on
            # validation PR-AUC, from the existing shipped-code sweep already
            # on disk -- the same method the remapped side must use for a
            # fair paired comparison.
            best_method, best_mean = None, float("-inf")
            for m in ["majority", "logreg_l2", "logreg_l1", "random_forest", "extra_trees",
                      "hist_gbm", "knn", "svm_linear", "xgboost", "lightgbm", "catboost"]:
                try:
                    vals = []
                    for s in SEEDS:
                        df = load_predictions(RUNG_DIRS[best_rung], task, phase, m, s, split="valid")
                        y, p = df["y_true"].to_numpy(), df["y_proba"].to_numpy()
                        if len(np.unique(y)) > 1:
                            from sklearn.metrics import average_precision_score
                            vals.append(average_precision_score(y, p))
                except FileNotFoundError:
                    continue
                if vals and float(np.mean(vals)) > best_mean:
                    best_method, best_mean = m, float(np.mean(vals))
            if best_method is None:
                rerun_results.append({**cell, "best_rung": best_rung, "error": "no validation predictions on disk"})
                continue

            fits = []
            for seed in SEEDS:
                td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
                all_terms_tva = {split: [_condition_terms(v) for v in (
                    X["condition"] if "condition" in X.columns else pd.Series([None] * len(X), index=X.index)
                ).values] for split, X in (("train", td.X_train), ("valid", td.X_valid), ("test", td.X_test))}
                _extend_remap(remap_dict, [s for terms in sum(all_terms_tva.values(), []) for s in terms],
                              encoder, codes, code_mat_normed)

                Ctr, Cva, Cte = _remapped_code_matrix(td.X_train, [td.X_train, td.X_valid, td.X_test],
                                                       remap_dict, best_rung)
                fz = TabularFeaturizer(task_type=td.task_type)
                Ttr, Tva, Tte = fz.fit_transform(td.X_train, td.y_train), fz.transform(td.X_valid), fz.transform(td.X_test)
                Xtr, Xva, Xte = np.hstack([Ttr, Ctr]), np.hstack([Tva, Cva]), np.hstack([Tte, Cte])

                MethodCls = get_method(best_method)
                meth = MethodCls(task_type=td.task_type, num_classes=td.num_classes, seed=seed, n_jobs=-1)
                meth.fit(Xtr, td.y_train, Xva, td.y_valid)
                proba = meth.predict_proba(Xte)
                fits.append((np.asarray(td.X_test.index, dtype=str), td.y_test.astype(float), proba))

            remapped_pool = pool_predictions(fits)
            shipped_rows = []
            for seed in SEEDS:
                try:
                    df = load_predictions(RUNG_DIRS[best_rung], task, phase, best_method, seed, split="test")
                except FileNotFoundError:
                    continue
                shipped_rows.append((df["nct_id"].to_numpy(), df["y_true"].to_numpy(), df["y_proba"].to_numpy()))
            shipped_pool = pool_predictions(shipped_rows)

            if len(remapped_pool["y_true"]) and len(remapped_pool["y_true"]) == len(shipped_pool["y_true"]) \
                    and np.array_equal(remapped_pool["nct_id"], shipped_pool["nct_id"]):
                contrast = pooled_paired_bootstrap(remapped_pool["nct_id"], remapped_pool["y_true"],
                                                    remapped_pool["proba"], shipped_pool["proba"], metric="prauc")
            else:
                contrast = {"error": "pooled remapped/shipped predictions not aligned"}
            rerun_results.append({**cell, "best_rung": best_rung, "best_method": best_method,
                                   "remapped_vs_shipped": contrast})
            print(f"  re-run {task}/{phase} ({best_rung}/{best_method}): {contrast}", flush=True)

    # ---- decision rule ---------------------------------------------------------
    # remapped_vs_shipped's mean_delta is remapped_prauc - shipped_prauc (see
    # the pooled_paired_bootstrap call above), so direction matters: a CI that
    # clears *positive* means the SapBERT remap is the better feature and
    # should become the recommended default (the plan's literal wording);
    # a CI that clears *negative* means remapping measurably hurts relative
    # to the shipped mapping despite the low agreement -- a different,
    # equally reportable finding the plan's wording doesn't spell out but
    # its intent (recommend whichever mapping actually predicts better)
    # clearly implies.
    def _direction(r):
        d = r.get("remapped_vs_shipped", {})
        if "lo" not in d:
            return None
        if d["lo"] > 0:
            return "remap_better"
        if d["hi"] < 0:
            return "remap_worse"
        return "indistinguishable"

    directions = {(r["task"], r["phase"]): _direction(r) for r in rerun_results}
    remap_better = [k for k, v in directions.items() if v == "remap_better"]
    remap_worse = [k for k, v in directions.items() if v == "remap_worse"]

    if remap_better:
        verdict = (f"Re-mapping IMPROVES pooled PR-AUC beyond the pooled CI on {remap_better}: every "
                   f"T23-T26 conclusion gets a mapping-sensitivity caveat, and the cleaned "
                   f"(SapBERT-remapped) mapping becomes the campaign's recommended default"
                   + (f" (note: it also measurably HURTS on {remap_worse} -- report both directions, "
                      f"not just the improvement)" if remap_worse else "") + ".")
    elif remap_worse:
        verdict = (f"Re-mapping measurably HURTS pooled PR-AUC (relative to the shipped mapping) on "
                   f"{remap_worse}, despite low shipped/remap agreement there (see cell_agreement_table). "
                   f"This is not the direction the plan's default framing anticipated: the shipped mapping "
                   f"is not improved upon by this SapBERT nearest-neighbor remap, so it stays the "
                   f"campaign's default; the disagreement is reported as a caveat on interpretation "
                   f"(condition strings may carry predictive signal beyond what any single ICD code "
                   f"captures), not as grounds to switch mappings. The 13%-error-floor literature figure "
                   f"is not vindicated as a *quality* problem here -- disagreement is real, but cleaning "
                   f"it does not help this benchmark's predictions.")
    elif rerun_results and all("remapped_vs_shipped" in r and "error" not in r["remapped_vs_shipped"] for r in rerun_results):
        verdict = ("Re-mapping does not move pooled PR-AUC beyond the pooled CI (in either direction) on "
                   "any of the 4 lowest-agreement cells: the 13%-error-floor literature figure is "
                   "declared non-binding here.")
    else:
        verdict = "Re-run incomplete or inconclusive on one or more of the 4 target cells; see rerun_results."

    artifact = {
        "test_id": "T27",
        "claim_at_stake": "how noisy is the shipped string->ICD mapping, and does cleaning it move scores",
        "inputs": {"levels": LEVELS, "seeds": SEEDS, "n_hand_adjudicate": N_HAND_ADJUDICATE,
                   "n_rerun_cells": N_RERUN_CELLS, "official_index_size": len(codes),
                   "best_rung_per_task": best_rung_per_task},
        "cell_agreement_table": cell_agreement_rows,
        "disagreement_sample_csv": SAMPLE_CSV,
        "disagreement_sample_n": len(sample_rows),
        "hand_adjudication_status": "NOT PERFORMED -- the sample is written for manual review; the plan "
                                    "calls this 'an evening' of human work, not automatable.",
        "rerun_cells": rerun_cells,
        "rerun_results": rerun_results,
        "rerun_directions": {f"{k[0]}/{k[1]}": v for k, v in directions.items()},
        "decision_rule": {
            "primary": "If re-mapping moves pooled PR-AUC by more than the pooled CI on the re-run "
                        "cells, every T23-T26 conclusion gets a mapping-sensitivity caveat and the "
                        "cleaned mapping becomes the campaign's recommended default. If not, the 13% "
                        "literature figure is non-binding here. The agreement table is reported "
                        "regardless -- the first published error estimate for this mapping pipeline.",
        },
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}")
    print(artifact["verdict"])
    return artifact


if __name__ == "__main__":
    main()
