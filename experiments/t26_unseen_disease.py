"""T26 -- Unseen-disease transfer.

disease-representation-test-plan.md. Multi-hot representations of a
never-seen disease are all-zero by construction; hierarchy (ancestors) and
dense vectors (SapBERT) claim to degrade gracefully. Which representation
still predicts when the disease family was never in training?

Per P12 (``src.splits.disease_holdout``): within each clinical-task cell's
train+valid pool, K=8 rotating folds hold out whole ICD-10 3-char families.
Arms: the best multi-hot rung per task (T23's validation-selected optimum),
``ancestors`` (T24), SapBERT (mean-pooled trial vectors), ``disease_text_only``
(P10's TF-IDF text arm, run as its own self-contained classifier, not fused
with the tabular block), and a presence control (has-a-code / n-codes only,
no disease identity at all -- the floor every real representation must beat).
Methods: random_forest + logreg_l2 for the four tabular-composed arms;
disease_text_only supplies its own internal classifier. 8 clinical cells
(mortality_rate_yn + serious_adverse_rate_yn, all 4 phases), 3 seeds (varying
model randomness only -- the fold assignment itself is fixed per P12).

Transfer penalty: for every arm, each held-out family fold is re-run against
a *random* K-way partition of the same eligible (has-a-code) trials at
identical fold sizes -- same code path, same train/eval mechanics, the only
difference is whether the held-out group is a real disease family or an
arbitrary same-size group. Pooled paired bootstrap (cluster by nct_id, same
infra as T22-T24) on ``(random_prauc - family_prauc)`` per arm is the
transfer penalty: positive means the family-holdout regime really did cost
that arm accuracy relative to no distributional shift.

Usage: `python -m experiments.t26_unseen_disease` (after T23's artifact
exists; P12 folds are built on first use and cached under
results/splits/disease_holdout/).
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import CodeFeaturizer, TabularFeaturizer, _recursive_parse_terms
from src.data.loader import load_task_phase
from src.eval.pooled_bootstrap import pool_predictions, pooled_paired_bootstrap
from src.methods.registry import get as get_method
from src.methods.text_nlp import DiseaseTextOnly, SapBERTEncoder
from src.splits.disease_holdout import build_and_save

CLINICAL_TASKS = ["mortality_rate_yn", "serious_adverse_rate_yn"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
CLINICAL_CELLS = [(t, p) for t in CLINICAL_TASKS for p in PHASES]
METHODS = ["random_forest", "logreg_l2"]
SEEDS = [42, 7, 123]
ARMS = ("best_rung", "ancestors", "sapbert", "presence")  # disease_text_only handled separately
DATA_ROOT = "data"
T23_ARTIFACT = "results/experiments/t23_granularity_ladder.json"
OUT_PATH = "results/experiments/t26_unseen_disease.json"


# ---------------------------------------------------------------------------
# Feature builders (fit on the fold's TRAIN rows only in every case).
# ---------------------------------------------------------------------------
def _condition_lists(X: pd.DataFrame) -> list:
    col = X["condition"] if "condition" in X.columns else pd.Series([None] * len(X), index=X.index)
    return [sorted(set(_recursive_parse_terms(v))) for v in col.values]


def _icd_lists(X: pd.DataFrame) -> list:
    col = X["icdcode"] if "icdcode" in X.columns else pd.Series([None] * len(X), index=X.index)
    return [sorted(set(c for c in _recursive_parse_terms(v) if c.strip())) for v in col.values]


def _presence_block(term_lists) -> tuple:
    n_terms = np.array([len(t) for t in term_lists], dtype=float).reshape(-1, 1)
    has = (n_terms > 0).astype(float)
    return n_terms, has


def build_arm_X(arm: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, encoder: SapBERTEncoder,
                best_rung: str):
    fz = TabularFeaturizer(task_type="binary")
    Ttr, Tte = fz.fit_transform(Xtr, ytr), fz.transform(Xte)
    if arm in ("best_rung", "ancestors"):
        granularity = best_rung if arm == "best_rung" else "ancestors"
        cz = CodeFeaturizer(min_df=10, granularity=granularity).fit(Xtr)
        Ctr, Cte = cz.transform(Xtr), cz.transform(Xte)
        return np.hstack([Ttr, Ctr]), np.hstack([Tte, Cte])
    if arm == "sapbert":
        ctr, cte = _condition_lists(Xtr), _condition_lists(Xte)
        Str, Ste = encoder.trial_vectors(ctr), encoder.trial_vectors(cte)
        ntr, htr = _presence_block(ctr)
        nte, hte = _presence_block(cte)
        return np.hstack([Ttr, Str, ntr, htr]), np.hstack([Tte, Ste, nte, hte])
    if arm == "presence":
        itr, ite = _icd_lists(Xtr), _icd_lists(Xte)
        ntr, htr = _presence_block(itr)
        nte, hte = _presence_block(ite)
        return np.hstack([Ttr, ntr, htr]), np.hstack([Tte, nte, hte])
    raise ValueError(arm)


def _random_partition(eligible_ids: list, fold_sizes: list, seed: int) -> list:
    """A random K-way partition of ``eligible_ids`` at exactly ``fold_sizes``
    (the same sizes P12's family folds produced) -- the paired baseline for
    the transfer-penalty contrast: identical mechanics, no family structure."""
    ids = list(eligible_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    out, start = [], 0
    for sz in fold_sizes:
        out.append([ids[i] for i in perm[start:start + sz]])
        start += sz
    return out


def main():
    with open(T23_ARTIFACT) as f:
        t23 = json.load(f)
    if "optimum_rung_per_cell_valid" not in t23:
        raise KeyError(
            "T23's artifact has no optimum_rung_per_cell_valid -- it predates the A2 fix. "
            "Re-run `python -m experiments.t23_granularity_ladder` first."
        )
    best_rung_per_task = {}
    for row in t23["optimum_rung_per_cell_valid"]:
        best_rung_per_task.setdefault(row["task"], []).append(row["optimum_rung"])
    for task, rungs in best_rung_per_task.items():
        non_null = [r for r in rungs if r is not None]
        best_rung_per_task[task] = Counter(non_null).most_common(1)[0][0] if non_null else "char3"
    print("best_rung_per_task:", best_rung_per_task, flush=True)

    encoder = SapBERTEncoder()
    # rows[(task, arm, method)] -> list of (nct_id_arr, y_true_arr, family_proba_arr, random_proba_arr)
    rows = {}
    text_rows = {}  # rows[task] for disease_text_only, same shape
    n_skipped_single_class = 0

    with Timer() as t:
        for task, phase in CLINICAL_CELLS:
            best_rung = best_rung_per_task.get(task, "char3")
            fold_art = build_and_save(task, phase, data_root=DATA_ROOT)
            td = load_task_phase(DATA_ROOT, task, phase, seed=42)
            X_pool = pd.concat([td.X_train, td.X_valid])
            y_pool = np.concatenate([td.y_train, td.y_valid])
            id_to_pos = {str(i): p for p, i in enumerate(X_pool.index)}

            K = fold_art["K"]
            fold_eval_ids = fold_art["fold_eval_ids"]
            eligible_ids = [i for ids in fold_eval_ids for i in ids]  # union == every has-code trial

            for seed in SEEDS:
                random_fold_ids = _random_partition(eligible_ids, fold_art["fold_sizes"], seed=seed)
                for k in range(K):
                    fam_eval = fold_eval_ids[k]
                    rand_eval = random_fold_ids[k]
                    if not fam_eval or not rand_eval:
                        continue
                    fam_eval_pos = [id_to_pos[i] for i in fam_eval]
                    rand_eval_pos = [id_to_pos[i] for i in rand_eval]
                    fam_train_pos = [p for p in range(len(X_pool)) if p not in set(fam_eval_pos)]
                    rand_train_pos = [p for p in range(len(X_pool)) if p not in set(rand_eval_pos)]

                    Xtr_fam, ytr_fam = X_pool.iloc[fam_train_pos], y_pool[fam_train_pos]
                    Xte_fam, yte_fam = X_pool.iloc[fam_eval_pos], y_pool[fam_eval_pos]
                    Xtr_rand, ytr_rand = X_pool.iloc[rand_train_pos], y_pool[rand_train_pos]
                    Xte_rand, yte_rand = X_pool.iloc[rand_eval_pos], y_pool[rand_eval_pos]

                    if len(np.unique(yte_fam)) < 2 or len(np.unique(yte_rand)) < 2:
                        n_skipped_single_class += 1
                        continue

                    for arm in ARMS:
                        Xtr_f, Xte_f = build_arm_X(arm, Xtr_fam, ytr_fam, Xte_fam, encoder, best_rung)
                        Xtr_r, Xte_r = build_arm_X(arm, Xtr_rand, ytr_rand, Xte_rand, encoder, best_rung)
                        for method in METHODS:
                            MethodCls = get_method(method)
                            m_fam = MethodCls(task_type="binary", num_classes=2, seed=seed, n_jobs=-1)
                            m_fam.fit(Xtr_f, ytr_fam)
                            proba_fam = m_fam.predict_proba(Xte_f)
                            m_rand = MethodCls(task_type="binary", num_classes=2, seed=seed, n_jobs=-1)
                            m_rand.fit(Xtr_r, ytr_rand)
                            proba_rand = m_rand.predict_proba(Xte_r)
                            key = (task, arm, method)
                            rows.setdefault(key, []).append(
                                (np.array(fam_eval, dtype=str), yte_fam.astype(float), proba_fam,
                                 np.array(rand_eval, dtype=str), yte_rand.astype(float), proba_rand))

                    # disease_text_only: its own internal LogReg, "raw" view, no tabular fusion
                    dt_fam = DiseaseTextOnly(task_type="binary", num_classes=2, seed=seed)
                    dt_fam.fit(Xtr_fam, ytr_fam)
                    proba_fam_dt = dt_fam.predict_proba(Xte_fam)
                    dt_rand = DiseaseTextOnly(task_type="binary", num_classes=2, seed=seed)
                    dt_rand.fit(Xtr_rand, ytr_rand)
                    proba_rand_dt = dt_rand.predict_proba(Xte_rand)
                    text_rows.setdefault(task, []).append(
                        (np.array(fam_eval, dtype=str), yte_fam.astype(float), proba_fam_dt,
                         np.array(rand_eval, dtype=str), yte_rand.astype(float), proba_rand_dt))

            print(f"  {task}/{phase} done (K={K}, n_eligible={len(eligible_ids)})", flush=True)

        # ---- pool + paired bootstrap: transfer_penalty = random_prauc - family_prauc ----
        def _pool_and_contrast(entries):
            fam_rows = [(e[0], e[1], e[2]) for e in entries]
            rand_rows = [(e[3], e[4], e[5]) for e in entries]
            fam_pool = pool_predictions(fam_rows)
            rand_pool = pool_predictions(rand_rows)
            if len(fam_pool["y_true"]) < 20:
                return {"error": "too few pooled rows"}
            # family and random pools are independent partitions of the same
            # eligible-trial universe (a trial's family-fold neighbor set and
            # random-fold neighbor set differ), so they are not row-aligned by
            # position -- pooled_paired_bootstrap needs one nct_id/y_true axis
            # shared by both proba vectors. Align on the intersection of ids
            # actually evaluated under both regimes (thinned only by the rare
            # single-class-skip above), sorted for a deterministic join.
            common = sorted(set(fam_pool["nct_id"].tolist()) & set(rand_pool["nct_id"].tolist()))
            if len(common) < 20:
                return {"error": "too few ids common to both partitions"}
            fam_idx = {i: p for p, i in enumerate(fam_pool["nct_id"])}
            rand_idx = {i: p for p, i in enumerate(rand_pool["nct_id"])}
            # A trial can recur (once per seed, and possibly once per fold-k it
            # was drawn into under each regime) -- keep every occurrence pair
            # in id order so the cluster bootstrap still resamples by trial.
            fam_pos_by_id, rand_pos_by_id = {}, {}
            for p, i in enumerate(fam_pool["nct_id"]):
                fam_pos_by_id.setdefault(i, []).append(p)
            for p, i in enumerate(rand_pool["nct_id"]):
                rand_pos_by_id.setdefault(i, []).append(p)
            ids_out, y_out, fam_p_out, rand_p_out = [], [], [], []
            for i in common:
                fp = fam_pos_by_id[i]
                rp = rand_pos_by_id[i]
                n = min(len(fp), len(rp))  # pair occurrences positionally (same seed order both sides)
                for j in range(n):
                    ids_out.append(i)
                    y_out.append(fam_pool["y_true"][fp[j]])
                    fam_p_out.append(fam_pool["proba"][fp[j]])
                    rand_p_out.append(rand_pool["proba"][rp[j]])
            return pooled_paired_bootstrap(np.array(ids_out), np.array(y_out),
                                            np.array(rand_p_out), np.array(fam_p_out), metric="prauc")

        transfer_penalty = {}
        for (task, arm, method), entries in rows.items():
            transfer_penalty[f"{task}|{arm}|{method}"] = {
                "task": task, "arm": arm, "method": method,
                "n_fold_fits": len(entries), **_pool_and_contrast(entries),
            }
        for task, entries in text_rows.items():
            transfer_penalty[f"{task}|disease_text_only|internal"] = {
                "task": task, "arm": "disease_text_only", "method": "internal",
                "n_fold_fits": len(entries), **_pool_and_contrast(entries),
            }

    # ---- decision rule (PR-3, second half) ------------------------------------
    def penalty_of(task, arm, method="random_forest"):
        key = f"{task}|{arm}|{method}" if arm != "disease_text_only" else f"{task}|disease_text_only|internal"
        r = transfer_penalty.get(key, {})
        return r.get("mean_delta", float("nan"))

    per_task_ordering = {}
    for task in CLINICAL_TASKS:
        penalties = {arm: penalty_of(task, arm) for arm in ARMS}
        penalties["disease_text_only"] = penalty_of(task, "disease_text_only")
        per_task_ordering[task] = dict(sorted(penalties.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=1e9))))

    # PR-3's actual wording is "smaller than every MULTI-HOT arm's" -- i.e.
    # best_rung and ancestors, not `presence` (a trivial floor with near-zero
    # penalty by construction: it carries no disease identity to lose under
    # family shift, so it is never a real contender and must not win the
    # "smallest" comparison by default). "Smaller... by more than the pooled
    # fold CI" is operationalized the same way the rest of this campaign
    # reports a per-task clear: non-overlapping CIs (sapbert's upper bound
    # below the multi-hot arm's lower bound), not just a point-estimate rank.
    def _sapbert_clears(task, multihot_arm):
        sap = transfer_penalty.get(f"{task}|sapbert|random_forest", {})
        other = transfer_penalty.get(f"{task}|{multihot_arm}|random_forest", {})
        if "hi" not in sap or "lo" not in other:
            return None
        return sap["hi"] < other["lo"]

    per_task_clears = {
        task: {arm: _sapbert_clears(task, arm) for arm in ("best_rung", "ancestors")}
        for task in CLINICAL_TASKS
    }
    sapbert_clears_both_tasks = all(
        all(v is True for v in per_task_clears[t].values()) for t in CLINICAL_TASKS
    )
    sapbert_clears_no_task = all(
        all(v is False for v in per_task_clears[t].values()) for t in CLINICAL_TASKS
    )
    all_indistinguishable = all(
        transfer_penalty.get(f"{t}|{a}|random_forest", {}).get("lo", float("nan")) <= 0 <= transfer_penalty.get(
            f"{t}|{a}|random_forest", {}).get("hi", float("nan"))
        for t in CLINICAL_TASKS for a in ARMS
        if "lo" in transfer_penalty.get(f"{t}|{a}|random_forest", {})
    )

    if sapbert_clears_both_tasks:
        verdict = ("PR-3 (second half) SUPPORTED: SapBERT's transfer penalty clears (non-overlapping "
                    "CI, strictly smaller) both multi-hot arms (best_rung, ancestors) on both clinical "
                    "tasks (random_forest).")
    elif all_indistinguishable:
        verdict = ("All representations' transfer penalties are indistinguishable from zero (CI "
                    "contains 0): transfer robustness does not sell any representation here; PR-3's "
                    "second half is not supported and Part 3 should drop the 'generalizes to new "
                    "diseases' angle.")
    elif sapbert_clears_no_task:
        verdict = ("PR-3 (second half) NOT SUPPORTED: SapBERT never clears both multi-hot arms on "
                    "either clinical task; see per_task_clears/per_task_ordering for the actual, "
                    "task-dependent pattern.")
    else:
        verdict = (f"Mixed, per-task result -- not the clean 'both clinical tasks' pattern PR-3 asks "
                   f"for: per_task_clears={per_task_clears}. Report per_task_ordering and "
                   f"transfer_penalty descriptively; do not claim PR-3 SUPPORTED or CUT overall.")

    artifact = {
        "test_id": "T26",
        "claim_at_stake": "which representation still predicts when the disease family was never in training",
        "inputs": {"arms": list(ARMS) + ["disease_text_only"], "methods": METHODS, "seeds": SEEDS,
                   "clinical_cells": CLINICAL_CELLS, "best_rung_per_task": best_rung_per_task},
        "n_skipped_single_class_folds": n_skipped_single_class,
        "transfer_penalty": transfer_penalty,
        "per_task_ordering_random_forest": per_task_ordering,
        "per_task_clears_multihot_random_forest": per_task_clears,
        "decision_rule": {
            "primary": "PR-3 (second half) SUPPORTED if SapBERT's transfer penalty is smaller than "
                        "every multi-hot arm's by more than the pooled fold CI, with ancestors between "
                        "them. All penalties indistinguishable -> transfer robustness sells nothing, "
                        "drop the angle. Report disease_text_only's penalty beside them regardless -- "
                        "if free text transfers best, that leads the section.",
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
