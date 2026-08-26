"""T23 -- The granularity ladder: where is the ICD encoding sweet spot?

disease-representation-test-plan.md (continuation of t21-code-channel-plan.md
and T22). T21 picked the 3-character rollup as "the primary encoding" without
comparing it to alternatives. This runs all 5 rungs -- chapter (22), block
(~280), char3 (509), full (1,617), CCSR (~540) -- across the 11 non-text
Tier A methods (tfidf_logreg is "raw" and untouched by design, per T21), all
20 cells, 5 seeds, view "tabular+codes" (P9), so every rung pairs against
both T1 and T21.

char3 is T21's own run (results_codes) -- not re-run, per standing rule 6.
The other 4 rungs are fresh sweeps into results_codes_<rung>/.

Usage: `python -m experiments.t23_granularity_ladder` (after all 5 rungs'
run dirs exist -- char3 from T21, the rest from this test's own sweep).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import CodeFeaturizer
from src.data.loader import TASKS, load_task_phase
from src.eval.pooled_bootstrap import pool_predictions, pooled_paired_bootstrap
from src.eval.predictions import load_predictions

METHODS = [
    "majority", "logreg_l2", "logreg_l1", "random_forest", "extra_trees",
    "hist_gbm", "knn", "svm_linear", "xgboost", "lightgbm", "catboost",
]
LINEAR_FAMILY = ["logreg_l1", "logreg_l2", "svm_linear"]
TREE_FAMILY = ["random_forest", "extra_trees", "hist_gbm", "xgboost", "lightgbm", "catboost"]
KNN_FAMILY = ["knn"]
SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"

RUNGS = ("chapter", "block", "char3", "full", "ccsr")
RUNG_DIRS = {
    "char3": "results_codes",  # T21's own run, not re-run (standing rule 6)
    "chapter": "results_codes_chapter",
    "block": "results_codes_block",
    "full": "results_codes_full",
    "ccsr": "results_codes_ccsr",
}
BASELINE_RUNS = "results/extracted/trialbench/results/runs"  # T1's "tabular" arm
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
OUT_PATH = "results/experiments/t23_granularity_ladder.json"
ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]
CLINICAL_TASKS = ["mortality_rate_yn", "serious_adverse_rate_yn"]
OPERATIONAL_TASK = "patient_dropout_rate_yn"


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


def _valid_point(results_dir, task, phase, method, seed):
    """A2 (wave1-preflight-review.md): validation-split PR-AUC for one fit,
    from the predictions parquet (not the run JSON, which only stores a test
    point). Binary tasks only -- P2 doesn't persist multiclass predictions,
    so this returns None for failure_reason and callers must handle that."""
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


def _has_any_code_mask(task, phase):
    """Test-set mask: does this row have >=1 usable ICD code? Granularity-
    independent -- a trial with zero raw codes has zero codes at every rung
    -- so this is computed once (char3, matching T21's own check) and reused
    for every rung's covered-subset scoring."""
    td = load_task_phase(DATA_ROOT, task, phase, seed=42)
    cz = CodeFeaturizer(min_df=10, granularity="char3").fit(td.X_train)
    parsed = cz._parse_block(td.X_test, "icd")
    has_any = np.array([len(t) > 0 for t in parsed])
    return has_any, td.X_test.index


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}
    powered_cells = set((r["task"], r["phase"]) for r in t1["delta_cell_table"] if not r["underpowered"])

    with Timer() as t:
        # ---- per-cell, per-method, per-rung means -----------------------
        # mean_prauc (test) drives every *descriptive* table below, unchanged.
        # mean_prauc_valid (A2) is what the *verdict-bearing* rung/method
        # selection further down uses instead -- computed from the saved
        # predictions parquet (binary tasks only; None/NaN for failure_reason,
        # since P2 doesn't persist multiclass predictions).
        rows = []
        for task, phase in ALL_CELLS:
            for method in METHODS:
                for rung in RUNGS:
                    mean, vals = _mean_over_seeds(RUNG_DIRS[rung] + "/runs", task, phase, method)
                    mean_v, vals_v = _mean_over_seeds_valid(RUNG_DIRS[rung], task, phase, method)
                    rows.append({"task": task, "phase": phase, "method": method, "rung": rung,
                                 "mean_prauc": mean, "n_seeds": len(vals),
                                 "mean_prauc_valid": mean_v, "n_seeds_valid": len(vals_v)})
            print(f"  {task}/{phase} done", flush=True)
        pcm_df = pd.DataFrame(rows)

        # A3 (wave1-preflight-review.md): a (task, phase, rung) cell counts as
        # complete only if every one of the 11 methods completed all 5 seeds --
        # same bar the run inventory uses. Cross-rung aggregates below are
        # restricted to cells complete at this bar for *every* rung being
        # compared, instead of a NaN-skipping mean silently rewarding whichever
        # rung happens to be missing its hardest cell.
        cell_complete = (pcm_df.groupby(["task", "phase", "rung"])["n_seeds"]
                          .apply(lambda s: bool((s == len(SEEDS)).all())).rename("complete").reset_index())

        # ---- null check: majority must show identical PR-AUC across rungs
        maj = pcm_df[pcm_df["method"] == "majority"].pivot(index=["task", "phase"], columns="rung", values="mean_prauc")
        majority_max_spread = float((maj.max(axis=1) - maj.min(axis=1)).abs().max()) if len(maj) else float("nan")
        majority_null_check_passed = bool(majority_max_spread < 1e-6)

        # ---- optimum rung per cell (best method's mean, per rung) --------
        best_per_cell_rung = (pcm_df.groupby(["task", "phase", "rung"])["mean_prauc"]
                               .max().reset_index().rename(columns={"mean_prauc": "best_method_mean"}))
        optimum_per_cell = []
        for (task, phase), sub in best_per_cell_rung.groupby(["task", "phase"]):
            if sub["best_method_mean"].notna().any():
                row = sub.loc[sub["best_method_mean"].idxmax()]
                optimum_per_cell.append({"task": task, "phase": phase, "optimum_rung": row["rung"],
                                           "optimum_mean": float(row["best_method_mean"]),
                                           "all_rungs": sub.set_index("rung")["best_method_mean"].round(4).to_dict()})
            else:
                optimum_per_cell.append({"task": task, "phase": phase, "optimum_rung": None,
                                           "optimum_mean": None, "all_rungs": {}})

        # ---- optimum rung per cell, validation-selected (A2) --------------
        # Same shape as optimum_per_cell above but selected on
        # mean_prauc_valid, not mean_prauc -- this is what T24 reads for "the
        # best single rung per task" (t24_ancestors.py), since feeding it a
        # test-argmax rung would carry A2's selection-on-test bias into T24's
        # own verdict-bearing contrast. Binary tasks only (see _valid_point).
        best_per_cell_rung_valid = (pcm_df.groupby(["task", "phase", "rung"])["mean_prauc_valid"]
                                     .max().reset_index().rename(columns={"mean_prauc_valid": "best_method_mean_valid"}))
        optimum_per_cell_valid = []
        for (task, phase), sub in best_per_cell_rung_valid.groupby(["task", "phase"]):
            if sub["best_method_mean_valid"].notna().any():
                row = sub.loc[sub["best_method_mean_valid"].idxmax()]
                optimum_per_cell_valid.append({"task": task, "phase": phase, "optimum_rung": row["rung"],
                                                 "optimum_mean_valid": float(row["best_method_mean_valid"]),
                                                 "all_rungs": sub.set_index("rung")["best_method_mean_valid"].round(4).to_dict()})
            else:
                optimum_per_cell_valid.append({"task": task, "phase": phase, "optimum_rung": None,
                                                 "optimum_mean_valid": None, "all_rungs": {}})

        # ---- per-family gain over char3 ----------------------------------
        def family_gain_table(family, cells):
            sub = pcm_df[pcm_df["method"].isin(family) & pcm_df.apply(lambda r: (r["task"], r["phase"]) in cells, axis=1)]
            piv = sub.pivot_table(index=["task", "phase", "method"], columns="rung", values="mean_prauc")
            # pivot_table drops a rung entirely (rather than an all-NaN column)
            # when it has literally zero completed cells yet (e.g. mid-sweep) --
            # reindex it back in as all-NaN so this reports "0 cells, no data"
            # instead of a KeyError.
            piv = piv.reindex(columns=list(RUNGS))
            gain = piv.sub(piv["char3"], axis=0)
            return {rung: {"mean_gain_vs_char3": float(gain[rung].mean()), "n_cells_methods": int(gain[rung].notna().sum())}
                    for rung in RUNGS if rung != "char3"}

        family_results_all = {"linear_margin": family_gain_table(LINEAR_FAMILY, set(ALL_CELLS)),
                               "trees_boosting": family_gain_table(TREE_FAMILY, set(ALL_CELLS)),
                               "knn": family_gain_table(KNN_FAMILY, set(ALL_CELLS))}
        family_results_powered = {"linear_margin": family_gain_table(LINEAR_FAMILY, powered_cells),
                                   "trees_boosting": family_gain_table(TREE_FAMILY, powered_cells),
                                   "knn": family_gain_table(KNN_FAMILY, powered_cells)}

        # ---- covered-subset scoring (computed once, per cell) -------------
        covered_subset = []
        for task, phase in ALL_CELLS:
            has_any, _ids = _has_any_code_mask(task, phase)
            frac_covered = float(has_any.mean())
            entry = {"task": task, "phase": phase, "frac_test_rows_with_any_code": frac_covered, "rungs": {}}
            for rung in RUNGS:
                full_vals, covered_vals = [], []
                for method in METHODS:
                    for seed in SEEDS:
                        try:
                            df = load_predictions(RUNG_DIRS[rung], task, phase, method, seed, split="test")
                        except FileNotFoundError:
                            continue
                        y, p = df["y_true"].to_numpy(), df["y_proba"].to_numpy()
                        full_vals.append(float(average_precision_score(y, p)))
                        if has_any.sum() > 20 and len(np.unique(y[has_any])) > 1:
                            covered_vals.append(float(average_precision_score(y[has_any], p[has_any])))
                if full_vals:
                    entry["rungs"][rung] = {
                        "full_test_mean_prauc": float(np.mean(full_vals)),
                        "covered_subset_mean_prauc": float(np.mean(covered_vals)) if covered_vals else None,
                    }
            covered_subset.append(entry)
            print(f"  covered-subset {task}/{phase}: frac_covered={frac_covered:.3f}", flush=True)

        # ---- gap closure to tfidf_logreg (best rung vs T1 tabular) --------
        gap_closure = []
        for task, phase in ALL_CELLS:
            sub = pcm_df[(pcm_df["task"] == task) & (pcm_df["phase"] == phase)]
            t1_means = {m: _mean_over_seeds(BASELINE_RUNS, task, phase, m)[0] for m in METHODS}
            best_tabular = float(np.nanmax(list(t1_means.values()))) if t1_means else float("nan")
            best_codes_arm = float(sub["mean_prauc"].max()) if len(sub) else float("nan")
            tfidf_mean, _ = _mean_over_seeds(BASELINE_RUNS, task, phase, "tfidf_logreg")
            denom = tfidf_mean - best_tabular
            ratio = (best_codes_arm - best_tabular) / denom if denom else float("nan")
            gap_closure.append({"task": task, "phase": phase, "best_tabular": best_tabular,
                                  "best_codes_arm": best_codes_arm, "tfidf_logreg": tfidf_mean,
                                  "gap_closure_ratio": ratio})
        median_gap_closure = float(np.nanmedian([g["gap_closure_ratio"] for g in gap_closure]))

        # ---- primary contrast: best rung vs char3, pooled per task --------
        # "Best rung" chosen once per task (mean over that task's cells, best
        # method per (cell, rung)), not per-phase -- same documented
        # simplification as T22, for the same reason (a per-phase "best"
        # would fragment the pooled trial set across phases inconsistently).
        #
        # A2: the rung argmax and the per-phase method argmax both select on
        # mean_prauc_valid (validation), never on the test-set numbers the
        # verdict-bearing bootstrap below then scores -- selection and
        # inference no longer share data. The old test-argmax is still
        # computed and recorded as test_oracle_best_rung (same convention T22
        # already used for its own oracle argmax), purely descriptive and
        # never fed into the verdict.
        #
        # A3: both argmaxes are restricted to phases where *every* rung in
        # RUNGS is a complete cell (11 methods x 5 seeds) -- so a rung can't
        # win by having skipped its hardest cell (measured concretely on
        # today's partial data: block wins the old unrestricted argmax on
        # mortality by not having run Phase4, while losing every phase it
        # actually ran). A task with no such phase gets an explicit "no data"
        # status instead of silently losing the argmax.
        pooled_best_vs_char3 = {}
        for task in TASKS:
          try:
            complete_here = cell_complete[(cell_complete["task"] == task) & cell_complete["complete"]]
            phases_by_rung = {rung: set(complete_here[complete_here["rung"] == rung]["phase"]) for rung in RUNGS}
            complete_phases = sorted(set.intersection(*phases_by_rung.values())) if phases_by_rung else []
            cells_dropped = [
                {"phase": phase, "incomplete_rungs": sorted(r for r in RUNGS if phase not in phases_by_rung[r])}
                for phase in ["Phase1", "Phase2", "Phase3", "Phase4"] if phase not in complete_phases
            ]

            # descriptive-only test-set oracle argmax, unrestricted by completeness
            task_best_test = best_per_cell_rung[best_per_cell_rung["task"] == task]
            rung_task_means_test = task_best_test.groupby("rung")["best_method_mean"].mean()
            test_oracle_best_rung = (rung_task_means_test.idxmax()
                                      if not rung_task_means_test.dropna().empty else None)

            if not complete_phases:
                pooled_best_vs_char3[task] = {
                    "status": "no data", "selection_split": "valid",
                    "error": "no phase is a complete cell (11 methods x 5 seeds) for every rung",
                    "n_cells_used": 0, "cells_dropped": cells_dropped,
                    "test_oracle_best_rung": test_oracle_best_rung,
                }
                print(f"  pooled {task}: no data ({len(cells_dropped)} phase(s) incomplete for at least one rung)", flush=True)
                continue

            task_best = best_per_cell_rung_valid[(best_per_cell_rung_valid["task"] == task)
                                                  & best_per_cell_rung_valid["phase"].isin(complete_phases)]
            rung_task_means = task_best.groupby("rung")["best_method_mean_valid"].mean()
            if rung_task_means.dropna().empty:
                pooled_best_vs_char3[task] = {
                    "status": "no data", "selection_split": "valid",
                    "error": "no validation predictions available for any rung on the complete phases "
                             "(multiclass task, or predictions missing)",
                    "n_cells_used": len(complete_phases), "cells_dropped": cells_dropped,
                    "test_oracle_best_rung": test_oracle_best_rung,
                }
                continue
            best_rung = rung_task_means.idxmax()
            common = {"selection_split": "valid", "n_cells_used": len(complete_phases),
                      "cells_dropped": cells_dropped, "test_oracle_best_rung": test_oracle_best_rung}
            if best_rung == "char3":
                pooled_best_vs_char3[task] = {**common, "best_rung": "char3",
                                                "note": "char3 is already the best rung; no contrast needed"}
                continue
            # per-cell best METHOD for the best rung (validation-selected), then pool its test predictions
            best_method_by_cell = (pcm_df[(pcm_df["task"] == task) & (pcm_df["rung"] == best_rung)
                                           & pcm_df["phase"].isin(complete_phases) & pcm_df["mean_prauc_valid"].notna()]
                                    .loc[lambda d: d.groupby("phase")["mean_prauc_valid"].idxmax()])
            char3_method_by_cell = (pcm_df[(pcm_df["task"] == task) & (pcm_df["rung"] == "char3")
                                            & pcm_df["phase"].isin(complete_phases) & pcm_df["mean_prauc_valid"].notna()]
                                     .loc[lambda d: d.groupby("phase")["mean_prauc_valid"].idxmax()])
            rung_rows, char3_rows = [], []
            for phase in complete_phases:
                bm_row = best_method_by_cell[best_method_by_cell["phase"] == phase]
                c3_row = char3_method_by_cell[char3_method_by_cell["phase"] == phase]
                if bm_row.empty or c3_row.empty:
                    continue
                bm_method, c3_method = bm_row["method"].iat[0], c3_row["method"].iat[0]
                for seed in SEEDS:
                    try:
                        df_r = load_predictions(RUNG_DIRS[best_rung], task, phase, bm_method, seed, split="test")
                        df_c = load_predictions(RUNG_DIRS["char3"], task, phase, c3_method, seed, split="test")
                    except FileNotFoundError:
                        continue
                    if len(df_r) != len(df_c) or not np.array_equal(df_r["nct_id"].to_numpy(), df_c["nct_id"].to_numpy()):
                        continue
                    rung_rows.append((df_r["nct_id"].to_numpy(), df_r["y_true"].to_numpy(), df_r["y_proba"].to_numpy()))
                    char3_rows.append((df_c["nct_id"].to_numpy(), df_c["y_true"].to_numpy(), df_c["y_proba"].to_numpy()))
            pooled_rung, pooled_char3 = pool_predictions(rung_rows), pool_predictions(char3_rows)
            if len(pooled_rung["y_true"]) and len(pooled_rung["y_true"]) == len(pooled_char3["y_true"]):
                res = pooled_paired_bootstrap(pooled_char3["nct_id"], pooled_char3["y_true"],
                                               pooled_rung["proba"], pooled_char3["proba"], metric="prauc")
                res.update(common)
                res["best_rung"] = best_rung
                pooled_best_vs_char3[task] = res
            else:
                pooled_best_vs_char3[task] = {**common, "best_rung": best_rung, "error": "no aligned pooled predictions"}
            print(f"  pooled {task}: best_rung={best_rung} (test_oracle={test_oracle_best_rung}, "
                  f"n_cells_used={len(complete_phases)}) {pooled_best_vs_char3[task]}", flush=True)
          except (ValueError, KeyError, IndexError) as e:
            pooled_best_vs_char3[task] = {"error": str(e)}
            print(f"  pooled {task}: FAILED -- {e}", flush=True)

    # ---- decision rule -----------------------------------------------------
    def clears_ci(task):
        r = pooled_best_vs_char3.get(task, {})
        return r.get("lo", float("nan")) > 0 if "lo" in r else False

    def has_verdict_data(task):
        r = pooled_best_vs_char3.get(task, {})
        return r.get("status") != "no data" and "best_rung" in r

    clinical_optimum_finer_or_equal = all(
        pooled_best_vs_char3.get(t, {}).get("best_rung") in ("char3", "ccsr") or not clears_ci(t)
        for t in CLINICAL_TASKS
    )
    dropout_indistinguishable = not clears_ci(OPERATIONAL_TASK)
    full_wins_anywhere = [task for task, r in pooled_best_vs_char3.items()
                           if r.get("best_rung") == "full" and r.get("lo", float("nan")) > 0]

    # A3: clinical_optimum_finer_or_equal/dropout_indistinguishable are both
    # "X or not clears_ci" disjunctions, which a "no data" task satisfies
    # vacuously (no CI to clear -> not clears_ci is True) -- so without this
    # gate, mortality/SAE/dropout all lacking data would still read as
    # "PR-1 SUPPORTED" from zero evidence. Require actual data on every task
    # that conjunct depends on before trusting it.
    primary_tasks_missing_data = [t for t in CLINICAL_TASKS + [OPERATIONAL_TASK] if not has_verdict_data(t)]

    if full_wins_anywhere:
        verdict = f"PR-1 CUT on: {full_wins_anywhere} (full codes won with a pooled CI excluding 0)."
    elif all(pooled_best_vs_char3.get(t, {}).get("best_rung") == "char3" for t in TASKS if t in pooled_best_vs_char3):
        verdict = "All rungs tie everywhere: granularity does not matter above chapter level. Endpoint-dependence story is not supported."
    elif primary_tasks_missing_data:
        verdict = (f"NO DATA yet for T23's primary contrast on {primary_tasks_missing_data}: no phase is a "
                   f"complete cell (11 methods x 5 seeds) for every rung on {'these tasks' if len(primary_tasks_missing_data) > 1 else 'this task'} "
                   f"-- see pooled_best_rung_vs_char3_per_task's cells_dropped. Not evaluating PR-1 SUPPORTED/CUT until it is.")
    elif clinical_optimum_finer_or_equal and dropout_indistinguishable:
        verdict = ("PR-1 SUPPORTED: pooled optimum on mortality/SAE is at char3/ccsr or finer with a "
                   "clearing CI, while dropout's rungs are indistinguishable from char3.")
    else:
        verdict = "PR-1 not cleanly supported or cut; report pooled_best_vs_char3 per task descriptively."

    artifact = {
        "test_id": "T23",
        "claim_at_stake": "there is a granularity sweet spot, and it depends on the endpoint",
        "inputs": {"methods": METHODS, "seeds": SEEDS, "rungs": RUNGS, "rung_dirs": RUNG_DIRS,
                   "baseline_runs": BASELINE_RUNS},
        "null_check_majority_max_spread": majority_max_spread,
        "null_check_passed": majority_null_check_passed,
        "per_cell_method_rung": rows,
        "optimum_rung_per_cell": optimum_per_cell,
        "optimum_rung_per_cell_valid": optimum_per_cell_valid,
        "family_gain_vs_char3_all_20_cells": family_results_all,
        "family_gain_vs_char3_powered_cells": family_results_powered,
        "covered_subset_scoring": covered_subset,
        "gap_closure_per_cell": gap_closure, "gap_closure_median": median_gap_closure,
        "pooled_best_rung_vs_char3_per_task": pooled_best_vs_char3,
        "decision_rule": {
            "primary": "PR-1 SUPPORTED if pooled optimum on mortality/SAE is at char3/ccsr or finer "
                        "(CI excludes 0 vs char3) while dropout's rungs are indistinguishable from char3. "
                        "All-tie -> 'granularity doesn't matter above chapter level', reported plainly. "
                        "Full codes winning anywhere (CI excludes 0) -> PR-1 CUT on that task.",
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
