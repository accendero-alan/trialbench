"""T24 -- Ancestors: the flattened-GRAM arm.

disease-representation-test-plan.md. GRAM (KDD 2017) claims ancestor
expansion is worth +10% on rare diseases -- imported by HINT/TrialBench as an
architecture, never tested as a feature encoding. Two arms, P9's
``"ancestors"`` (leaf/char3/block/chapter unioned into one multi-hot) and
``"stack"`` (the same four levels kept as four separate blocks), same grid
as T23 (11 non-text Tier A methods, 20 cells, 5 seeds, "tabular+codes").

Reads T23's artifact for "the best single rung" per task (the comparison
point GRAM's ancestor expansion has to beat) -- run T23 first.

Usage: `python -m experiments.t24_ancestors` (after T23's artifact and both
arms' run dirs -- results_codes_ancestors, results_codes_stack -- exist).
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import _icd_char3, _recursive_parse_terms
from src.data.loader import TASKS, load_task_phase
from src.eval.pooled_bootstrap import pool_predictions, pooled_paired_bootstrap
from src.eval.predictions import load_predictions

METHODS = [
    "majority", "logreg_l2", "logreg_l1", "random_forest", "extra_trees",
    "hist_gbm", "knn", "svm_linear", "xgboost", "lightgbm", "catboost",
]
SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"

ARMS = ("ancestors", "stack")
ARM_DIRS = {"ancestors": "results_codes_ancestors", "stack": "results_codes_stack"}
T23_ARTIFACT = "results/experiments/t23_granularity_ladder.json"
RUNG_DIRS = {  # from t23_granularity_ladder.py -- kept in sync manually, both are P9 rung names
    "char3": "results_codes", "chapter": "results_codes_chapter",
    "block": "results_codes_block", "full": "results_codes_full", "ccsr": "results_codes_ccsr",
}
OUT_PATH = "results/experiments/t24_ancestors.json"
ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]


def _load_point(runs_dir, task, phase, method, seed):
    path = os.path.join(runs_dir, f"{task}__{phase}__{method}__seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    return rec["point"]["prauc"] if rec.get("status") == "ok" else None


def _mean_over_seeds(runs_dir, task, phase, method):
    vals = [_load_point(runs_dir, task, phase, method, s) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)) if vals else float("nan")), vals


def _best_method(runs_dir, task, phase, methods=METHODS):
    best_name, best_mean = None, float("-inf")
    for m in methods:
        mean, vals = _mean_over_seeds(runs_dir, task, phase, m)
        if vals and mean > best_mean:
            best_name, best_mean = m, mean
    return best_name, best_mean


def _rare_common_split(task, phase):
    """Test rows with >=1 ICD code, split at the median of each row's
    rarest code's char3 TRAIN frequency (raw counts, not the min_df-
    thresholded CodeFeaturizer vocab -- GRAM's claim is about rare codes,
    exactly the ones min_df would drop). Rows with zero codes are excluded
    (undefined "rarest code frequency"); their count is reported so the
    exclusion isn't silent."""
    td = load_task_phase(DATA_ROOT, task, phase, seed=42)
    train_char3 = [
        {_icd_char3(c) for c in _recursive_parse_terms(v) if c.strip()}
        for v in (td.X_train["icdcode"].values if "icdcode" in td.X_train.columns else [])
    ]
    train_freq = Counter(c for terms in train_char3 for c in terms)

    test_char3 = [
        {_icd_char3(c) for c in _recursive_parse_terms(v) if c.strip()}
        for v in (td.X_test["icdcode"].values if "icdcode" in td.X_test.columns else [[] for _ in range(len(td.X_test))])
    ]
    rarest_freq = np.full(len(test_char3), np.nan)
    for i, terms in enumerate(test_char3):
        if terms:
            rarest_freq[i] = min(train_freq.get(c, 0) for c in terms)

    has_code = ~np.isnan(rarest_freq)
    n_excluded = int((~has_code).sum())
    if has_code.sum() < 20:
        return None, None, n_excluded, td.X_test.index
    median = np.nanmedian(rarest_freq)
    rare_mask = has_code & (rarest_freq <= median)
    common_mask = has_code & (rarest_freq > median)
    return rare_mask, common_mask, n_excluded, td.X_test.index


def main():
    with open(T23_ARTIFACT) as f:
        t23 = json.load(f)
    best_rung_per_task = {}
    for row in t23["optimum_rung_per_cell"]:
        best_rung_per_task.setdefault(row["task"], []).append(row["optimum_rung"])
    # per-task "best single rung": the rung that's optimal on the most cells for that task
    for task, rungs in best_rung_per_task.items():
        best_rung_per_task[task] = Counter(rungs).most_common(1)[0][0] if rungs else "char3"

    with Timer() as t:
        # ---- per-cell, per-method, per-arm means (vs. that task's best rung)
        rows = []
        for task, phase in ALL_CELLS:
            best_rung = best_rung_per_task.get(task, "char3")
            for method in METHODS:
                base_mean, _ = _mean_over_seeds(RUNG_DIRS[best_rung], task, phase, method)
                for arm in ARMS:
                    arm_mean, arm_vals = _mean_over_seeds(ARM_DIRS[arm], task, phase, method)
                    gain = arm_mean - base_mean if (arm_vals and not np.isnan(base_mean)) else float("nan")
                    rows.append({"task": task, "phase": phase, "method": method, "arm": arm,
                                 "best_rung_used": best_rung, "best_rung_mean": base_mean,
                                 "arm_mean": arm_mean if arm_vals else None, "gain_vs_best_rung": gain})
            print(f"  {task}/{phase} (best_rung={best_rung}) done", flush=True)
        pcm_df = pd.DataFrame(rows)

        # ---- null check ----------------------------------------------------
        maj = pcm_df[pcm_df["method"] == "majority"]
        majority_max_abs_gain = float(maj["gain_vs_best_rung"].abs().max()) if maj["gain_vs_best_rung"].notna().any() else float("nan")
        majority_null_check_passed = bool(majority_max_abs_gain < 1e-6) if not np.isnan(majority_max_abs_gain) else None

        # ---- rare/common split, per arm, pooled per task -------------------
        rare_common = []
        for task in TASKS:
            for arm in ARMS:
                rare_rows, common_rows, base_rare_rows, base_common_rows = [], [], [], []
                n_excluded_total = 0
                best_rung = best_rung_per_task.get(task, "char3")
                for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]:
                    rare_mask, common_mask, n_excl, test_ids = _rare_common_split(task, phase)
                    if rare_mask is None:
                        continue
                    n_excluded_total += n_excl
                    best_method, _ = _best_method(ARM_DIRS[arm], task, phase)
                    base_method, _ = _best_method(RUNG_DIRS[best_rung], task, phase)
                    if best_method is None or base_method is None:
                        continue
                    for seed in SEEDS:
                        try:
                            df_arm = load_predictions(ARM_DIRS[arm], task, phase, best_method, seed, split="test")
                            df_base = load_predictions(RUNG_DIRS[best_rung], task, phase, base_method, seed, split="test")
                        except FileNotFoundError:
                            continue
                        if len(df_arm) != len(rare_mask) or len(df_base) != len(rare_mask):
                            continue
                        y_a, p_a = df_arm["y_true"].to_numpy(), df_arm["y_proba"].to_numpy()
                        y_b, p_b = df_base["y_true"].to_numpy(), df_base["y_proba"].to_numpy()
                        rare_rows.append((y_a[rare_mask], p_a[rare_mask]))
                        common_rows.append((y_a[common_mask], p_a[common_mask]))
                        base_rare_rows.append((y_b[rare_mask], p_b[rare_mask]))
                        base_common_rows.append((y_b[common_mask], p_b[common_mask]))

                def _contrast(arm_pool, base_pool):
                    a, b = pool_predictions(arm_pool), pool_predictions(base_pool)
                    if len(a["y_true"]) < 20 or len(a["y_true"]) != len(b["y_true"]):
                        return {"error": "too few pooled rows or misaligned"}
                    if not np.array_equal(a["y_true"], b["y_true"]):
                        return {"error": "label mismatch"}
                    return pooled_paired_bootstrap(a["y_true"], a["proba"], b["proba"], metric="prauc")

                rare_res = _contrast(rare_rows, base_rare_rows)
                common_res = _contrast(common_rows, base_common_rows)
                rare_common.append({"task": task, "arm": arm, "best_rung": best_rung,
                                     "n_excluded_no_code_rows": n_excluded_total,
                                     "rare_half": rare_res, "common_half": common_res})
                print(f"  rare/common {task}/{arm}: rare={rare_res} common={common_res}", flush=True)

    # ---- decision rule -------------------------------------------------------
    def clears(res):
        return isinstance(res, dict) and res.get("lo", float("nan")) > 0

    rare_clears_any = [r for r in rare_common if clears(r["rare_half"])]
    common_clears_any = [r for r in rare_common if clears(r["common_half"])]
    rare_only = [r for r in rare_clears_any if not clears(next(
        (r2["common_half"] for r2 in rare_common if r2["task"] == r["task"] and r2["arm"] == r["arm"]), {}))]

    any_gain_all_cells = pcm_df["gain_vs_best_rung"].notna().any() and pcm_df["gain_vs_best_rung"].mean() > 0
    uniform_gain = (pcm_df.groupby("arm")["gain_vs_best_rung"].mean() > 0).all() and not rare_only

    if rare_only:
        verdict = (f"PR-2 SUPPORTED: rare-half gain clears its pooled CI while common-half doesn't, for "
                   f"{[(r['task'], r['arm']) for r in rare_only]}.")
    elif uniform_gain and any_gain_all_cells:
        verdict = "Uniform gains across rare and common halves: hierarchy helps, but the rare-disease mechanism claim is SOFTENED."
    else:
        verdict = "No gain pattern supports PR-2: the flattened form of GRAM's idea does not survive in this data."

    artifact = {
        "test_id": "T24",
        "claim_at_stake": "ontology ancestors pay exactly where data is thin (flattened GRAM)",
        "inputs": {"methods": METHODS, "seeds": SEEDS, "arms": ARMS, "arm_dirs": ARM_DIRS,
                   "best_rung_per_task": best_rung_per_task},
        "null_check_majority_max_abs_gain": majority_max_abs_gain,
        "null_check_passed": majority_null_check_passed,
        "per_cell_method": rows,
        "rare_common_split_pooled": rare_common,
        "decision_rule": {
            "primary": "PR-2 SUPPORTED if the pooled gain over T23's best single rung clears the pooled "
                        "CI on the rare half and not the common half. Uniform gains -> hierarchy helps "
                        "but SOFTENED. No gain anywhere -> flattened GRAM is dead in this data.",
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
