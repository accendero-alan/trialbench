"""T25 -- Sparse versus dense disease representations.

disease-representation-test-plan.md. Part 2's embedding verdict ("768 dense
dimensions lose to 50,000 sparse ones") was earned on full-document
Bio_ClinicalBERT and is being quietly generalized to all dense
representations. SapBERT condition vectors are a different object:
entity-level, synonym-collapsing, trained for exactly this. Do dense disease
vectors beat multi-hot codes on the standard split, and does combining them
add anything?

Arms, each composed as [TabularFeaturizer admin block] + [disease block]:
  (a) sapbert     -- SapBERT trial vectors (P11, mean-pooled; a max-pooled
                      variant is run too, method=random_forest only, and
                      recorded but not decision-bearing).
  (b) mesh        -- shipped MeSH graph embeddings (P11/src/data/mesh_embeddings.py),
                      mean-pooled.
  (c) best_code   -- the best multi-hot rung per task (T23's validation-selected
                      optimum, same lookup T24 uses) -- reuses those sweeps'
                      existing runs, no new fits.
  (d) concat      -- (c)'s tabular+codes(best_rung) block, concatenated with
                      (a)'s SapBERT block.
Methods: random_forest, extra_trees, logreg_l2, lightgbm (the tree family
that took T21's code-channel gain, plus a linear margin method -- same-family
contrasts only, per T12's lesson) -- plus `majority`, run in every arm purely
for the T21 null check (standing rule 8), never fed into the verdict. All 20
cells, 5 seeds.

Arm (e) (the GPU fine-tune) is **not implemented in this pass** -- deferred
by explicit user decision (2026-08-27); PR-7 is not evaluated here and the
artifact says so rather than silently omitting it.

Usage: `python -m experiments.t25_dense_vs_sparse` (after T23's artifact
exists). Resumable: skips any (task, phase, method, seed) run already on
disk in its own results_t25_* directory; pass --force to recompute.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import CodeFeaturizer, TabularFeaturizer, _recursive_parse_terms
from src.data.loader import TASKS, load_task_phase
from src.data.mesh_embeddings import mesh_trial_vectors
from src.eval import metrics as M
from src.eval.pooled_bootstrap import pool_predictions, pooled_paired_bootstrap
from src.eval.predictions import load_predictions, save_predictions
from src.methods.registry import get as get_method
from src.methods.text_nlp import SapBERTEncoder

FIT_METHODS = ["majority", "logreg_l2", "random_forest", "extra_trees", "lightgbm"]
VERDICT_METHODS = ["logreg_l2", "random_forest", "extra_trees", "lightgbm"]
SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"

RESULTS_DIRS = {"sapbert": "results_t25_sapbert", "mesh": "results_t25_mesh",
                "concat": "results_t25_concat", "sapbert_maxpool": "results_t25_sapbert_maxpool"}
RUNG_DIRS = {  # T23/T24's own rung directory names, kept in sync manually
    "char3": "results_codes", "chapter": "results_codes_chapter",
    "block": "results_codes_block", "full": "results_codes_full", "ccsr": "results_codes_ccsr",
}
T23_ARTIFACT = "results/experiments/t23_granularity_ladder.json"
OUT_PATH = "results/experiments/t25_dense_vs_sparse.json"
ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]
CLINICAL_TASKS = ["mortality_rate_yn", "serious_adverse_rate_yn"]


# ---------------------------------------------------------------------------
# Feature block builders. Every block is either fit on TRAIN only
# (TabularFeaturizer/CodeFeaturizer) or a frozen pretrained encoder applied
# identically to every split (SapBERT/MeSH) -- no leakage path either way.
# ---------------------------------------------------------------------------
def _condition_lists(X: pd.DataFrame) -> list:
    col = X["condition"] if "condition" in X.columns else pd.Series([None] * len(X), index=X.index)
    return [sorted(set(_recursive_parse_terms(v))) for v in col.values]


def _mesh_cond_lists(X: pd.DataFrame) -> list:
    col = (X["condition_browse/mesh_term"] if "condition_browse/mesh_term" in X.columns
           else pd.Series([None] * len(X), index=X.index))
    return [sorted(set(_recursive_parse_terms(v))) for v in col.values]


def _presence_block(term_lists) -> tuple:
    n_terms = np.array([len(t) for t in term_lists], dtype=float).reshape(-1, 1)
    has = (n_terms > 0).astype(float)
    return n_terms, has


def _tabular_block(td):
    fz = TabularFeaturizer(task_type=td.task_type)
    return fz.fit_transform(td.X_train, td.y_train), fz.transform(td.X_valid), fz.transform(td.X_test)


def _sapbert_block(td, encoder: SapBERTEncoder, pool: str = "mean"):
    lists = {"train": _condition_lists(td.X_train), "valid": _condition_lists(td.X_valid),
             "test": _condition_lists(td.X_test)}
    out = {}
    for split, terms in lists.items():
        vecs = encoder.trial_vectors(terms, pool=pool)
        n_terms, has = _presence_block(terms)
        out[split] = np.hstack([vecs, n_terms, has])
    return out["train"], out["valid"], out["test"]


def _mesh_block(td):
    lists = {"train": _mesh_cond_lists(td.X_train), "valid": _mesh_cond_lists(td.X_valid),
             "test": _mesh_cond_lists(td.X_test)}
    out = {}
    for split, terms in lists.items():
        vecs = mesh_trial_vectors(terms)
        n_terms, has = _presence_block(terms)
        out[split] = np.hstack([vecs, n_terms, has])
    return out["train"], out["valid"], out["test"]


def _code_block(td, granularity: str):
    cz = CodeFeaturizer(min_df=10, granularity=granularity).fit(td.X_train)
    return cz.transform(td.X_train), cz.transform(td.X_valid), cz.transform(td.X_test)


def build_arm_X(arm: str, td, encoder: SapBERTEncoder, best_rung: str = None):
    Ttr, Tva, Tte = _tabular_block(td)
    if arm == "sapbert":
        Btr, Bva, Bte = _sapbert_block(td, encoder, pool="mean")
        return np.hstack([Ttr, Btr]), np.hstack([Tva, Bva]), np.hstack([Tte, Bte])
    if arm == "sapbert_maxpool":
        Btr, Bva, Bte = _sapbert_block(td, encoder, pool="max")
        return np.hstack([Ttr, Btr]), np.hstack([Tva, Bva]), np.hstack([Tte, Bte])
    if arm == "mesh":
        Btr, Bva, Bte = _mesh_block(td)
        return np.hstack([Ttr, Btr]), np.hstack([Tva, Bva]), np.hstack([Tte, Bte])
    if arm == "concat":
        Ctr, Cva, Cte = _code_block(td, best_rung)
        Str, Sva, Ste = _sapbert_block(td, encoder, pool="mean")
        return (np.hstack([Ttr, Ctr, Str]), np.hstack([Tva, Cva, Sva]), np.hstack([Tte, Cte, Ste]))
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Grid runner -- mirrors src/run_benchmark.py's run_cell tail (predictions +
# run-JSON schema) closely enough that load_predictions/_valid_point-style
# helpers below work unchanged against these results dirs.
# ---------------------------------------------------------------------------
def run_grid(arm: str, results_dir: str, methods: list, cells: list, seeds: list,
             best_rung_per_task: dict = None, force: bool = False):
    runs_dir = os.path.join(results_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    encoder = SapBERTEncoder()
    grid = [(t, p, m, s) for (t, p) in cells for m in methods for s in seeds]
    print(f"[{arm}] {len(grid)} fits -> {results_dir}", flush=True)
    for i, (task, phase, method, seed) in enumerate(grid, 1):
        stem = f"{task}__{phase}__{method}__seed{seed}"
        out_path = os.path.join(runs_dir, stem + ".json")
        if os.path.exists(out_path) and not force:
            print(f"  [{arm} {i}/{len(grid)}] skip (done): {stem}", flush=True)
            continue
        try:
            best_rung = (best_rung_per_task or {}).get(task, "char3")
            td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
            Xtr, Xva, Xte = build_arm_X(arm, td, encoder, best_rung=best_rung)
            MethodCls = get_method(method)
            meth = MethodCls(task_type=td.task_type, num_classes=td.num_classes, seed=seed, n_jobs=-1)
            t0 = time.time()
            meth.fit(Xtr, td.y_train, Xva, td.y_valid)
            valid_proba = meth.predict_proba(Xva)
            proba = meth.predict_proba(Xte)
            fit_secs = time.time() - t0

            if td.task_type == "binary":
                save_predictions(results_dir, task, phase, method, seed,
                                  valid_ids=td.X_valid.index, valid_proba=valid_proba, valid_y=td.y_valid,
                                  test_ids=td.X_test.index, test_proba=proba, test_y=td.y_test)

            boot = M.bootstrap(td.y_test, proba, td.task_type, td.num_classes,
                                n_resamples=1000, ci=0.95, seed=seed)
            point = M.compute(td.y_test, proba, td.task_type, td.num_classes)
            feature_view = f"t25_{arm}" + (f"(best_rung={best_rung})" if arm == "concat" else "")
            rec = {"task": task, "phase": phase, "method": method, "seed": seed,
                   "task_type": td.task_type, "num_classes": td.num_classes,
                   "n_train": int(len(td.y_train)), "n_test": int(len(td.y_test)),
                   "n_features": int(Xtr.shape[1]), "fit_secs": round(fit_secs, 2),
                   "feature_view": feature_view, "headline": M.HEADLINE,
                   "point": point, "bootstrap": boot, "status": "ok"}
            print(f"  [{arm} {i}/{len(grid)}] ok   {stem}: prauc={point.get('prauc', float('nan')):.4f} "
                  f"({fit_secs:.1f}s)", flush=True)
        except (NotImplementedError, ImportError) as e:
            rec = {"task": task, "phase": phase, "method": method, "seed": seed,
                   "status": "skipped", "reason": f"{type(e).__name__}: {e}"}
            print(f"  [{arm} {i}/{len(grid)}] SKIP {stem}: {rec['reason']}", flush=True)
        except FileNotFoundError as e:
            rec = {"task": task, "phase": phase, "method": method, "seed": seed,
                   "status": "no_data", "reason": str(e)}
            print(f"  [{arm} {i}/{len(grid)}] DATA {stem}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec = {"task": task, "phase": phase, "method": method, "seed": seed,
                   "status": "error", "reason": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()}
            print(f"  [{arm} {i}/{len(grid)}] ERR  {stem}: {rec['reason']}", flush=True)

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# Analysis -- same A1/A2/A3-safe pattern as t23/t24: select rung/method on
# VALIDATION, bootstrap on test, cluster by nct_id, restrict cross-cell
# aggregates to cells complete (5/5 seeds) for every arm being compared.
# ---------------------------------------------------------------------------
def _load_point(runs_dir, task, phase, method, seed):
    path = os.path.join(runs_dir, f"{task}__{phase}__{method}__seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    return rec["point"]["prauc"] if rec.get("status") == "ok" else None


def _mean_over_seeds(runs_dir, task, phase, method, seeds=SEEDS):
    vals = [_load_point(runs_dir, task, phase, method, s) for s in seeds]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)) if vals else float("nan")), vals


def _valid_point(results_dir, task, phase, method, seed):
    try:
        df = load_predictions(results_dir, task, phase, method, seed, split="valid")
    except FileNotFoundError:
        return None
    y, p = df["y_true"].to_numpy(), df["y_proba"].to_numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def _best_method_valid(results_dir, task, phase, methods=VERDICT_METHODS, seeds=SEEDS):
    best_name, best_mean = None, float("-inf")
    for m in methods:
        vals = [_valid_point(results_dir, task, phase, m, s) for s in seeds]
        vals = [v for v in vals if v is not None]
        if vals and float(np.mean(vals)) > best_mean:
            best_name, best_mean = m, float(np.mean(vals))
    return best_name


def _arm_results_dir(arm, best_rung_per_task, task):
    if arm == "best_code":
        return RUNG_DIRS[best_rung_per_task.get(task, "char3")]
    return RESULTS_DIRS[arm]


def _pooled_test(arm, task, best_rung_per_task, seeds=SEEDS):
    """Pool this arm's test predictions for `task` across its 4 phases and
    `seeds`, per-phase method selected on validation (VERDICT_METHODS only).
    Returns pool_predictions()'s dict, plus how many phases contributed."""
    results_dir = _arm_results_dir(arm, best_rung_per_task, task)
    rows, n_phases_used = [], 0
    for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]:
        method = _best_method_valid(results_dir, task, phase)
        if method is None:
            continue
        phase_contributed = False
        for seed in seeds:
            try:
                df = load_predictions(results_dir, task, phase, method, seed, split="test")
            except FileNotFoundError:
                continue
            rows.append((df["nct_id"].to_numpy(), df["y_true"].to_numpy(), df["y_proba"].to_numpy()))
            phase_contributed = True
        n_phases_used += int(phase_contributed)
    return pool_predictions(rows), n_phases_used


def _contrast(pool_a, pool_b):
    if len(pool_a["y_true"]) < 20 or len(pool_a["y_true"]) != len(pool_b["y_true"]):
        return {"error": "too few pooled rows or misaligned"}
    if not np.array_equal(pool_a["nct_id"], pool_b["nct_id"]):
        return {"error": "nct_id mismatch between arms -- not paired"}
    return pooled_paired_bootstrap(pool_a["nct_id"], pool_a["y_true"], pool_a["proba"], pool_b["proba"],
                                    metric="prauc")


def clears(res):
    return isinstance(res, dict) and res.get("lo", float("nan")) > 0


def clears_negative(res):
    return isinstance(res, dict) and res.get("hi", float("nan")) < 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-fits", action="store_true",
                    help="analysis-only re-run against whatever is already on disk")
    args = ap.parse_args()

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

    with Timer() as t:
        if not args.skip_fits:
            run_grid("sapbert", RESULTS_DIRS["sapbert"], FIT_METHODS, ALL_CELLS, SEEDS, force=args.force)
            run_grid("mesh", RESULTS_DIRS["mesh"], FIT_METHODS, ALL_CELLS, SEEDS, force=args.force)
            run_grid("concat", RESULTS_DIRS["concat"], FIT_METHODS, ALL_CELLS, SEEDS,
                     best_rung_per_task=best_rung_per_task, force=args.force)
            run_grid("sapbert_maxpool", RESULTS_DIRS["sapbert_maxpool"], ["random_forest"], ALL_CELLS, SEEDS,
                     force=args.force)

        # ---- null check: majority must be identical across arms per cell ----
        maj_rows = []
        for task, phase in ALL_CELLS:
            per_arm = {}
            for arm in ("sapbert", "mesh", "concat"):
                mean, vals = _mean_over_seeds(os.path.join(RESULTS_DIRS[arm], "runs"), task, phase, "majority")
                if vals:
                    per_arm[arm] = mean
            best_rung = best_rung_per_task.get(task, "char3")
            mean_c, vals_c = _mean_over_seeds(RUNG_DIRS[best_rung] + "/runs", task, phase, "majority")
            if vals_c:
                per_arm["best_code"] = mean_c
            if len(per_arm) >= 2:
                spread = max(per_arm.values()) - min(per_arm.values())
                maj_rows.append({"task": task, "phase": phase, "per_arm": per_arm, "spread": spread})
        majority_max_spread = float(max((r["spread"] for r in maj_rows), default=float("nan")))
        majority_null_check_passed = bool(majority_max_spread < 1e-6) if maj_rows else None

        # ---- per-cell, per-method means (descriptive) -------------------------
        rows = []
        for task, phase in ALL_CELLS:
            best_rung = best_rung_per_task.get(task, "char3")
            for method in FIT_METHODS:
                for arm in ("sapbert", "mesh", "concat"):
                    mean, vals = _mean_over_seeds(os.path.join(RESULTS_DIRS[arm], "runs"), task, phase, method)
                    rows.append({"task": task, "phase": phase, "method": method, "arm": arm,
                                 "mean_prauc": mean, "n_seeds": len(vals)})
                mean_c, vals_c = _mean_over_seeds(RUNG_DIRS[best_rung] + "/runs", task, phase, method)
                rows.append({"task": task, "phase": phase, "method": method, "arm": "best_code",
                             "mean_prauc": mean_c, "n_seeds": len(vals_c), "best_rung_used": best_rung})
            print(f"  {task}/{phase} (best_rung={best_rung}) done", flush=True)
        per_cell_df = pd.DataFrame(rows)

        # ---- pooled per-task contrasts: a vs c, d vs a, d vs c -----------------
        pooled_contrasts = {}
        for task in TASKS:
            pool_a, nph_a = _pooled_test("sapbert", task, best_rung_per_task)
            pool_c, nph_c = _pooled_test("best_code", task, best_rung_per_task)
            pool_d, nph_d = _pooled_test("concat", task, best_rung_per_task)
            entry = {"n_phases_used": {"a_sapbert": nph_a, "c_best_code": nph_c, "d_concat": nph_d}}
            entry["a_vs_c"] = _contrast(pool_a, pool_c)
            entry["d_vs_a"] = _contrast(pool_d, pool_a)
            entry["d_vs_c"] = _contrast(pool_d, pool_c)
            pooled_contrasts[task] = entry
            print(f"  pooled {task}: a_vs_c={entry['a_vs_c']} d_vs_a={entry['d_vs_a']} d_vs_c={entry['d_vs_c']}",
                  flush=True)

        # ---- max-pool secondary variant (descriptive, not decision-bearing) ---
        maxpool_rows = []
        for task, phase in ALL_CELLS:
            mean_mean, _ = _mean_over_seeds(os.path.join(RESULTS_DIRS["sapbert"], "runs"), task, phase, "random_forest")
            mean_max, _ = _mean_over_seeds(os.path.join(RESULTS_DIRS["sapbert_maxpool"], "runs"), task, phase, "random_forest")
            maxpool_rows.append({"task": task, "phase": phase, "mean_pool_prauc": mean_mean,
                                  "max_pool_prauc": mean_max,
                                  "max_minus_mean": (mean_max - mean_mean) if not (np.isnan(mean_mean) or np.isnan(mean_max)) else None})

    # ---- decision rule (PR-3) -------------------------------------------------
    clinical_a_within_ci_of_c = all(
        (not clears_negative(pooled_contrasts[t]["a_vs_c"])) for t in CLINICAL_TASKS
        if "error" not in pooled_contrasts.get(t, {}).get("a_vs_c", {"error": 1})
    )
    a_loses_clinical = [t for t in CLINICAL_TASKS if clears_negative(pooled_contrasts.get(t, {}).get("a_vs_c", {}))]
    d_beats_both_parents_anywhere = [
        t for t in TASKS
        if clears(pooled_contrasts.get(t, {}).get("d_vs_a", {})) and clears(pooled_contrasts.get(t, {}).get("d_vs_c", {}))
    ]

    if a_loses_clinical:
        verdict = (f"PR-3 (first half) CUT on {a_loses_clinical}: SapBERT (a) loses to best multi-hot (c) "
                   f"by more than the pooled CI -- dense-entity vectors join Bio_ClinicalBERT in the "
                   f"'lost here' column on these tasks. T26 remains PR-3's second half.")
    elif clinical_a_within_ci_of_c:
        verdict = ("PR-3 (first half) SUPPORTED: SapBERT (a) is within pooled CI of best multi-hot (c) "
                   "on the clinical tasks.")
    else:
        verdict = "PR-3 (first half) not cleanly decided; report pooled_contrasts per task descriptively."

    if d_beats_both_parents_anywhere:
        verdict += (f" Fusion (d) beats both parents (a) and (c) by more than the pooled CI on "
                    f"{d_beats_both_parents_anywhere} -- first fusion result in this benchmark to clear "
                    f"its error bars; earns a line in Part 3.")

    artifact = {
        "test_id": "T25",
        "claim_at_stake": "dense disease-entity vectors (SapBERT) versus multi-hot codes; does fusion add anything",
        "arm_e_status": "DEFERRED -- fine-tuned encoder (PR-7) not implemented this pass (user decision, 2026-08-27). "
                        "PR-7 is not evaluated by this artifact.",
        "inputs": {"fit_methods": FIT_METHODS, "verdict_methods": VERDICT_METHODS, "seeds": SEEDS,
                   "best_rung_per_task": best_rung_per_task, "results_dirs": RESULTS_DIRS},
        "null_check_majority": {"rows": maj_rows, "max_spread": majority_max_spread,
                                "passed": majority_null_check_passed},
        "tfidf_logreg_null_check": "satisfied by construction -- tfidf_logreg is feature_view='raw' and is "
                                   "never touched by any tabular/codes/dense composition in this script.",
        "per_cell_method_arm": rows,
        "pooled_contrasts_per_task": pooled_contrasts,
        "sapbert_pool_variant_maxpool_vs_meanpool": {
            "note": "descriptive only, method=random_forest, not decision-bearing",
            "rows": maxpool_rows,
        },
        "decision_rule": {
            "primary": "PR-3 (first half) SUPPORTED if (a) is within pooled CI of (c) on the clinical "
                        "tasks. Losing by more than the pooled CI -> CUT on that task. (d) beating both "
                        "(a) and (c) by more than the pooled CI on any task -> fusion earns a line.",
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
