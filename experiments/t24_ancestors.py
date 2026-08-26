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
from sklearn.metrics import average_precision_score

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


def _valid_point(results_dir, task, phase, method, seed):
    """A2 (wave1-preflight-review.md): same as t23_granularity_ladder.py's
    helper of the same name -- validation-split PR-AUC from the predictions
    parquet, binary tasks only."""
    try:
        df = load_predictions(results_dir, task, phase, method, seed, split="valid")
    except FileNotFoundError:
        return None
    y, p = df["y_true"].to_numpy(), df["y_proba"].to_numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def _mean_over_seeds_valid(results_dir, task, phase, method):
    vals = [_valid_point(results_dir, task, phase, method, s) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)) if vals else float("nan")), vals


def _best_method_valid(results_dir, task, phase, methods=METHODS):
    """Same as ``_best_method`` but selecting on validation, not test -- for
    every argmax that feeds the verdict-bearing rare/common bootstrap below.
    ``results_dir`` here is the top-level results dir (no ``/runs`` suffix),
    matching ``load_predictions``' own path convention -- not ``_best_method``'s
    ``runs_dir`` (JSON-based, ``+"/runs"``)."""
    best_name, best_mean = None, float("-inf")
    for m in methods:
        mean, vals = _mean_over_seeds_valid(results_dir, task, phase, m)
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
    # A2 (wave1-preflight-review.md): read T23's validation-selected per-cell
    # optimum, not its test-set one -- t24's own rare/common bootstrap below
    # scores these methods' *test* predictions, so feeding it a test-argmax
    # "best rung" would carry the same selection-on-test bias T23 itself had.
    if "optimum_rung_per_cell_valid" not in t23:
        raise KeyError(
            "T23's artifact has no optimum_rung_per_cell_valid -- it predates the A2 fix. "
            "Re-run `python -m experiments.t23_granularity_ladder` first."
        )
    best_rung_per_task = {}
    for row in t23["optimum_rung_per_cell_valid"]:
        best_rung_per_task.setdefault(row["task"], []).append(row["optimum_rung"])
    # per-task "best single rung": the rung that's optimal on the most cells
    # for that task. optimum_rung is None for a cell with no validation data
    # (multiclass -- e.g. failure_reason, which has no persisted predictions
    # at all -- or simply not run yet); drop those before taking the mode
    # rather than let a task made entirely of Nones resolve to best_rung=None
    # (RUNG_DIRS[None] then KeyErrors below). Falls back to "char3" same as
    # a task missing from the dict entirely.
    for task, rungs in best_rung_per_task.items():
        non_null = [r for r in rungs if r is not None]
        best_rung_per_task[task] = Counter(non_null).most_common(1)[0][0] if non_null else "char3"

    with Timer() as t:
        # ---- per-cell, per-method, per-arm means (vs. that task's best rung)
        rows = []
        for task, phase in ALL_CELLS:
            best_rung = best_rung_per_task.get(task, "char3")
            for method in METHODS:
                base_mean, base_vals = _mean_over_seeds(RUNG_DIRS[best_rung] + "/runs", task, phase, method)
                for arm in ARMS:
                    arm_mean, arm_vals = _mean_over_seeds(ARM_DIRS[arm] + "/runs", task, phase, method)
                    gain = arm_mean - base_mean if (arm_vals and not np.isnan(base_mean)) else float("nan")
                    rows.append({"task": task, "phase": phase, "method": method, "arm": arm,
                                 "best_rung_used": best_rung, "best_rung_mean": base_mean,
                                 "arm_mean": arm_mean if arm_vals else None, "gain_vs_best_rung": gain,
                                 "n_seeds_arm": len(arm_vals), "n_seeds_base": len(base_vals)})
            print(f"  {task}/{phase} (best_rung={best_rung}) done", flush=True)
        pcm_df = pd.DataFrame(rows)

        # A3 (wave1-preflight-review.md): a (task, phase, arm) cell counts as
        # complete only if every method has all 5 seeds on *both* the arm and
        # the base rung used for that task -- gain_vs_best_rung is otherwise a
        # NaN-skipping mean that would let a partially-run arm or base rung
        # rank however its available (easier) cells happen to fall.
        cell_complete = (pcm_df.groupby(["task", "phase", "arm"])
                          .apply(lambda d: bool((d["n_seeds_arm"] == len(SEEDS)).all()
                                                 and (d["n_seeds_base"] == len(SEEDS)).all()),
                                 include_groups=False)
                          .rename("complete").reset_index())
        complete_triples = set(map(tuple, cell_complete.loc[cell_complete["complete"], ["task", "phase", "arm"]].values))
        all_triples = set(map(tuple, pcm_df[["task", "phase", "arm"]].drop_duplicates().values))
        cells_dropped_gain = sorted(all_triples - complete_triples)
        pcm_df_complete = pcm_df[pcm_df.apply(lambda r: (r["task"], r["phase"], r["arm"]) in complete_triples, axis=1)]

        # ---- null check ----------------------------------------------------
        maj = pcm_df[pcm_df["method"] == "majority"]
        majority_max_abs_gain = float(maj["gain_vs_best_rung"].abs().max()) if maj["gain_vs_best_rung"].notna().any() else float("nan")
        majority_null_check_passed = bool(majority_max_abs_gain < 1e-6) if not np.isnan(majority_max_abs_gain) else None

        # ---- rare/common split, per arm, pooled per task -------------------
        # A2: method selection (both the arm and the base rung, per phase) is
        # now validation-based (_best_method_valid), not the test argmax
        # _best_method used to feed straight into this same phase's test-split
        # bootstrap below. A1: predictions are pooled with nct_id and the
        # bootstrap cluster-resamples by trial instead of by row.
        rare_common = []
        for task in TASKS:
            for arm in ARMS:
                rare_rows, common_rows, base_rare_rows, base_common_rows = [], [], [], []
                n_excluded_total = 0
                n_phases_used = 0
                best_rung = best_rung_per_task.get(task, "char3")
                for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]:
                    rare_mask, common_mask, n_excl, test_ids = _rare_common_split(task, phase)
                    if rare_mask is None:
                        continue
                    n_excluded_total += n_excl
                    best_method, _ = _best_method_valid(ARM_DIRS[arm], task, phase)
                    base_method, _ = _best_method_valid(RUNG_DIRS[best_rung], task, phase)
                    if best_method is None or base_method is None:
                        continue
                    phase_contributed = False
                    for seed in SEEDS:
                        try:
                            df_arm = load_predictions(ARM_DIRS[arm], task, phase, best_method, seed, split="test")
                            df_base = load_predictions(RUNG_DIRS[best_rung], task, phase, base_method, seed, split="test")
                        except FileNotFoundError:
                            continue
                        if len(df_arm) != len(rare_mask) or len(df_base) != len(rare_mask):
                            continue
                        id_a = df_arm["nct_id"].to_numpy()
                        y_a, p_a = df_arm["y_true"].to_numpy(), df_arm["y_proba"].to_numpy()
                        id_b = df_base["nct_id"].to_numpy()
                        y_b, p_b = df_base["y_true"].to_numpy(), df_base["y_proba"].to_numpy()
                        rare_rows.append((id_a[rare_mask], y_a[rare_mask], p_a[rare_mask]))
                        common_rows.append((id_a[common_mask], y_a[common_mask], p_a[common_mask]))
                        base_rare_rows.append((id_b[rare_mask], y_b[rare_mask], p_b[rare_mask]))
                        base_common_rows.append((id_b[common_mask], y_b[common_mask], p_b[common_mask]))
                        phase_contributed = True
                    n_phases_used += int(phase_contributed)

                def _contrast(arm_pool, base_pool):
                    a, b = pool_predictions(arm_pool), pool_predictions(base_pool)
                    if len(a["y_true"]) < 20 or len(a["y_true"]) != len(b["y_true"]):
                        return {"error": "too few pooled rows or misaligned"}
                    if not np.array_equal(a["nct_id"], b["nct_id"]):
                        return {"error": "label mismatch"}
                    return pooled_paired_bootstrap(a["nct_id"], a["y_true"], a["proba"], b["proba"], metric="prauc")

                rare_res = _contrast(rare_rows, base_rare_rows)
                common_res = _contrast(common_rows, base_common_rows)
                rare_common.append({"task": task, "arm": arm, "best_rung": best_rung,
                                     "n_phases_used": n_phases_used,
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

    # A3: gated on pcm_df_complete (cells where every method has 5/5 seeds on
    # both the arm and the base rung), not the raw NaN-skipping pcm_df -- see
    # its construction above.
    if len(pcm_df_complete):
        any_gain_all_cells = bool(pcm_df_complete["gain_vs_best_rung"].notna().any()
                                   and pcm_df_complete["gain_vs_best_rung"].mean() > 0)
        uniform_gain = bool((pcm_df_complete.groupby("arm")["gain_vs_best_rung"].mean() > 0).all()) and not rare_only
    else:
        any_gain_all_cells = False
        uniform_gain = False

    if rare_only:
        verdict = (f"PR-2 SUPPORTED: rare-half gain clears its pooled CI while common-half doesn't, for "
                   f"{[(r['task'], r['arm']) for r in rare_only]}.")
    elif not len(pcm_df_complete):
        verdict = ("NO DATA for the uniform-gain check: no (task, phase, arm) cell has every method at "
                    "5/5 seeds on both the arm and its task's base rung yet. rare_common_split_pooled's "
                    "per-task/arm bootstraps above are unaffected (they pool whatever's on disk).")
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
        "uniform_gain_check": {
            "n_cells_used": int(len(complete_triples)), "n_cells_total": int(len(all_triples)),
            "cells_dropped": [{"task": t, "phase": p, "arm": a} for (t, p, a) in cells_dropped_gain],
            "any_gain_all_cells": any_gain_all_cells, "uniform_gain": uniform_gain,
        },
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
