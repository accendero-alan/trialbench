"""T21 -- The code channel: ICD and MeSH in the Tier A methods.

t21-code-channel-plan.md (continuation of newsletter-part2-test-plan.md). A
six-cell single-method (LightGBM) probe found +0.018 mean PR-AUC from adding
ICD/MeSH codes, closing 13-43% of the tabular-to-tfidf_logreg gap -- but
trees are poorly matched to 500-1200 sparse binary columns, so that's
plausibly a floor. This runs all 11 non-text Tier A methods (tfidf_logreg is
"raw" and untouched by design) across the full 20-cell x 5-seed grid, in two
new arms: "tabular+codes" (administrative view + the code blocks) and
"codes" (code blocks alone, no administrative columns), produced by
`run_benchmark.py --feature-view-override` (P6/P7) and
`CodeFeaturizer` (P5). The "tabular" baseline arm is T1's -- not re-run.

Reads three run-JSON trees (T1 baseline, arm 1, arm 2) plus arm predictions
(parquet, P2) for the covered-subset sub-analysis.

Usage: `python -m experiments.t21_code_channel` (after both arms complete)
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import CodeFeaturizer
from src.data.loader import TASKS, load_task_phase
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

BASELINE_RUNS = "results/extracted/trialbench/results/runs"  # T1's "tabular" arm
ARM1_DIR = "results_codes"          # tabular+codes
ARM2_DIR = "results_codes_only"     # codes
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
OUT_PATH = "results/experiments/t21_code_channel.json"
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
    return float(np.mean(vals)) if vals else float("nan"), vals


def _has_any_code_mask(task, phase):
    """Test-set mask: does this row have >=1 usable code in any block? Test
    rows are seed-independent (only train/valid split varies with seed), so
    this is computed once per cell and reused across all 5 seeds."""
    td = load_task_phase(DATA_ROOT, task, phase, seed=42)
    cz = CodeFeaturizer(min_df=10).fit(td.X_train)
    parsed = {b: cz._parse_block(td.X_test, b) for b in cz.BLOCKS}
    has_any = np.zeros(len(td.X_test), dtype=bool)
    for b in cz.BLOCKS:
        has_any |= np.array([len(t) > 0 for t in parsed[b]])
    return has_any, td.X_test.index


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}
    powered_cells = [(r["task"], r["phase"]) for r in t1["delta_cell_table"] if not r["underpowered"]]

    with Timer() as t:
        # ---- gather per-cell, per-method, per-arm means -----------------
        per_cell_method = []
        for task, phase in ALL_CELLS:
            for method in METHODS:
                base_mean, base_vals = _mean_over_seeds(BASELINE_RUNS, task, phase, method)
                arm1_mean, arm1_vals = _mean_over_seeds(ARM1_DIR + "/runs", task, phase, method)
                arm2_mean, arm2_vals = _mean_over_seeds(ARM2_DIR + "/runs", task, phase, method)
                gain1 = arm1_mean - base_mean if not (np.isnan(arm1_mean) or np.isnan(base_mean)) else float("nan")
                per_seed_gain1 = ([a - b for a, b in zip(arm1_vals, base_vals)]
                                   if len(arm1_vals) == len(base_vals) == len(SEEDS) else [])
                per_cell_method.append({
                    "task": task, "phase": phase, "method": method,
                    "tabular_mean": base_mean, "tabular_plus_codes_mean": arm1_mean, "codes_only_mean": arm2_mean,
                    "gain_tabular_plus_codes": gain1,
                    "per_seed_gain_tabular_plus_codes": per_seed_gain1,
                })
                print(f"  {task}/{phase}/{method}: tab={base_mean:.4f} +codes={arm1_mean:.4f} "
                      f"codes_only={arm2_mean:.4f} gain={gain1:+.4f}", flush=True)

        pcm_df = pd.DataFrame(per_cell_method)

        # ---- null check: majority must show 0.0000 gain everywhere ------
        maj = pcm_df[pcm_df["method"] == "majority"]
        majority_max_abs_gain = float(maj["gain_tabular_plus_codes"].abs().max())
        majority_null_check_passed = bool(majority_max_abs_gain < 1e-6)

        # ---- sub-analysis 1: per method family ---------------------------
        def family_stats(family, cells):
            sub = pcm_df[(pcm_df["method"].isin(family)) & (pcm_df.apply(lambda r: (r["task"], r["phase"]) in cells, axis=1))]
            per_cell_gain = sub.groupby(["task", "phase"])["gain_tabular_plus_codes"].mean()
            n_exceed = int((per_cell_gain.abs() > per_cell_gain.index.map(lambda k: delta_cells.get(k, np.inf))).sum())
            sign_holds = 0
            for (task, phase), g in per_cell_gain.items():
                rows = sub[(sub["task"] == task) & (sub["phase"] == phase)]
                per_seed = np.mean([r for r in rows["per_seed_gain_tabular_plus_codes"] if r], axis=0) \
                    if any(rows["per_seed_gain_tabular_plus_codes"]) else np.array([])
                if len(per_seed) and np.all(np.sign(per_seed) == np.sign(g)):
                    sign_holds += 1
            return {
                "mean_gain": float(per_cell_gain.mean()), "n_cells_exceed_delta": n_exceed,
                "n_cells_sign_holds": sign_holds, "n_cells": len(per_cell_gain),
                "std_across_cells": float(per_cell_gain.std()),
            }

        family_results_all = {
            "linear_margin": family_stats(LINEAR_FAMILY, set(ALL_CELLS)),
            "trees_boosting": family_stats(TREE_FAMILY, set(ALL_CELLS)),
            "knn": family_stats(KNN_FAMILY, set(ALL_CELLS)),
        }
        family_results_powered = {
            "linear_margin": family_stats(LINEAR_FAMILY, set(powered_cells)),
            "trees_boosting": family_stats(TREE_FAMILY, set(powered_cells)),
            "knn": family_stats(KNN_FAMILY, set(powered_cells)),
        }

        # ---- sub-analysis 2: covered-subset scoring -----------------------
        covered_subset = []
        for task, phase in ALL_CELLS:
            has_any, test_ids = _has_any_code_mask(task, phase)
            frac_covered = float(has_any.mean())
            entry = {"task": task, "phase": phase, "frac_test_rows_with_any_code": frac_covered, "methods": {}}
            for method in LINEAR_FAMILY + TREE_FAMILY:
                full_vals, covered_vals = [], []
                for seed in SEEDS:
                    try:
                        df = load_predictions(ARM1_DIR, task, phase, method, seed, split="test")
                    except FileNotFoundError:
                        continue
                    from sklearn.metrics import average_precision_score
                    y, p = df["y_true"].to_numpy(), df["y_proba"].to_numpy()
                    full_vals.append(float(average_precision_score(y, p)))
                    if has_any.sum() > 20 and len(np.unique(y[has_any])) > 1:
                        covered_vals.append(float(average_precision_score(y[has_any], p[has_any])))
                if full_vals:
                    entry["methods"][method] = {
                        "full_test_mean_prauc": float(np.mean(full_vals)),
                        "covered_subset_mean_prauc": float(np.mean(covered_vals)) if covered_vals else None,
                    }
            covered_subset.append(entry)
            print(f"  covered-subset {task}/{phase}: frac_covered={frac_covered:.3f}", flush=True)

        # ---- sub-analysis 3: gap closure -----------------------------------
        gap_closure = []
        for task, phase in ALL_CELLS:
            sub = pcm_df[(pcm_df["task"] == task) & (pcm_df["phase"] == phase)]
            best_tabular = float(sub["tabular_mean"].max())
            best_codes_arm = float(sub["tabular_plus_codes_mean"].max())
            tfidf_mean, _ = _mean_over_seeds(BASELINE_RUNS, task, phase, "tfidf_logreg")
            denom = tfidf_mean - best_tabular
            ratio = (best_codes_arm - best_tabular) / denom if denom else float("nan")
            gap_closure.append({"task": task, "phase": phase, "best_tabular": best_tabular,
                                  "best_codes_arm": best_codes_arm, "tfidf_logreg": tfidf_mean,
                                  "gap_closure_ratio": ratio})
        median_gap_closure = float(np.nanmedian([g["gap_closure_ratio"] for g in gap_closure]))

        # ---- sub-analysis 4: codes-only vs full administrative view --------
        codes_only_wins = 0
        for task, phase in ALL_CELLS:
            sub = pcm_df[(pcm_df["task"] == task) & (pcm_df["phase"] == phase)]
            if sub["codes_only_mean"].notna().any() and float(sub["codes_only_mean"].max()) > float(sub["tabular_plus_codes_mean"].max()):
                codes_only_wins += 1

        # ---- fork: codes + best linear vs tfidf_logreg ----------------------
        fork_rows = []
        for task, phase in ALL_CELLS:
            sub = pcm_df[(pcm_df["task"] == task) & (pcm_df["phase"] == phase) & (pcm_df["method"].isin(LINEAR_FAMILY))]
            best_linear_codes = float(sub["codes_only_mean"].max()) if sub["codes_only_mean"].notna().any() else float("nan")
            tfidf_mean, _ = _mean_over_seeds(BASELINE_RUNS, task, phase, "tfidf_logreg")
            delta = delta_cells.get((task, phase), float("nan"))
            gap = abs(best_linear_codes - tfidf_mean) if not np.isnan(best_linear_codes) else float("nan")
            fork_rows.append({"task": task, "phase": phase, "best_linear_codes_only": best_linear_codes,
                                "tfidf_logreg": tfidf_mean, "gap": gap, "delta_cell": delta,
                                "within_delta": bool(gap <= delta) if not np.isnan(gap) else None,
                                "powered": (task, phase) in powered_cells})
        n_within_powered = sum(1 for r in fork_rows if r["within_delta"] and r["powered"])
        n_lags_powered = sum(1 for r in fork_rows if r["within_delta"] is False and r["powered"])
        n_powered = len(powered_cells)

    # ---- decision rules --------------------------------------------------
    primary_confirmed_families = [name for name, s in family_results_all.items()
                                    if s["n_cells_exceed_delta"] >= 5 and s["n_cells_sign_holds"] >= s["n_cells_exceed_delta"]]
    harmed_families = [name for name, s in family_results_powered.items() if s["mean_gain"] < 0]

    if primary_confirmed_families:
        primary_verdict = f"CONFIRMED for: {', '.join(primary_confirmed_families)} (>=5/20 cells exceed delta_cell, sign holds)."
    else:
        primary_verdict = "DIRECTIONAL: no family clears 5/20 cells exceeding delta_cell with sign holding. Sign is positive and usable; size unresolved."
    if harmed_families:
        primary_verdict += f" HARMED (negative mean gain on powered cells): {', '.join(harmed_families)} (hypothesis, not finding)."

    lin_gain, tree_gain = family_results_all["linear_margin"]["mean_gain"], family_results_all["trees_boosting"]["mean_gain"]
    lin_std, tree_std = family_results_all["linear_margin"]["std_across_cells"], family_results_all["trees_boosting"]["std_across_cells"]
    shape_supported = bool((lin_gain - tree_gain) > max(lin_std, tree_std))
    prediction_verdict = ("SUPPORTED: linear family's mean gain exceeds tree family's by more than the "
                           "between-cell spread of either." if shape_supported else
                           "CUT: linear vs tree gain difference is within spread -- channel is method-agnostic.")

    if n_within_powered >= 5:
        fork_verdict = (f"Text advantage is substantially coded disease content: codes+best-linear lands "
                         f"within delta_cell of tfidf_logreg on {n_within_powered}/{n_powered} powered cells. "
                         f"Part 3 should be redesigned around what remains.")
    elif n_lags_powered >= 8:
        fork_verdict = (f"Text carries something the controlled vocabulary doesn't: codes+best-linear lags "
                         f"tfidf_logreg by more than delta_cell on {n_lags_powered}/{n_powered} powered cells. "
                         f"The text-methods bake-off is justified.")
    else:
        fork_verdict = (f"Neither fork threshold reached ({n_within_powered}/{n_powered} within, "
                          f"{n_lags_powered}/{n_powered} lagging) -- inconclusive on this split.")

    artifact = {
        "test_id": "T21",
        "claim_at_stake": "the tabular view is administrative by construction, not by necessity",
        "inputs": {"methods": METHODS, "seeds": SEEDS, "arm1_dir": ARM1_DIR, "arm2_dir": ARM2_DIR,
                   "baseline_runs": BASELINE_RUNS},
        "null_check_majority_max_abs_gain": majority_max_abs_gain,
        "null_check_passed": majority_null_check_passed,
        "per_cell_method": per_cell_method,
        "family_results_all_20_cells": family_results_all,
        "family_results_11_powered_cells": family_results_powered,
        "covered_subset_scoring": covered_subset,
        "gap_closure_per_cell": gap_closure, "gap_closure_median": median_gap_closure,
        "codes_only_beats_full_administrative_count": codes_only_wins, "codes_only_beats_full_administrative_of": len(ALL_CELLS),
        "fork_per_cell": fork_rows, "fork_n_within_delta_powered": n_within_powered,
        "fork_n_lags_powered": n_lags_powered, "fork_n_powered_cells": n_powered,
        "power_accounting_note": f"{n_powered}/20 cells are powered per T1; every count above is reported "
                                   f"both over all 20 and, where noted, over the {n_powered} powered cells.",
        "decision_rules": {
            "primary": "If any family's mean gain exceeds delta_cell on >=5/20 cells and holds sign "
                        "across seeds, CONFIRMED with a number. Else DIRECTIONAL. Negative mean gain "
                        "on powered cells -> HARMED (hypothesis).",
            "pre_registered_prediction": "If linear family's mean gain exceeds tree family's by more "
                        "than the between-cell spread of either, 'shape suits linear' is SUPPORTED. "
                        "Else CUT, method-agnostic.",
            "fork": "If codes+best-linear lands within delta_cell of tfidf_logreg on >=5/11 powered "
                    "cells, text advantage is substantially coded content (Part 3 redesign). If it "
                    "lags by >delta_cell on >=8/11 powered cells, bake-off is justified.",
        },
        "verdict": f"{primary_verdict} PREDICTION: {prediction_verdict} FORK: {fork_verdict} "
                    f"Gap closure median: {median_gap_closure:.2%}. Codes-only beats full administrative "
                    f"view on {codes_only_wins}/{len(ALL_CELLS)} cells. Null check "
                    f"(majority=0.0000): {'PASSED' if majority_null_check_passed else 'FAILED -- harness leak suspected'}.",
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}")
    print(artifact["verdict"])
    return artifact


if __name__ == "__main__":
    main()
