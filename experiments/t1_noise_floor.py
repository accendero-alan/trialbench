"""T1 -- Noise floor: multi-seed variance across the Tier A grid.

newsletter-part2-test-plan.md, T1. Every numeric claim in Part 2 rests on this:
no difference anywhere is real until it exceeds what re-seeding alone produces.

Reads the run JSONs `python -m src.run_benchmark --seeds 42 7 123 2024 5`
already wrote to `results/extracted/trialbench/results/runs/`, computes per
cell:

    delta_cell = max(1.96 * seed_sd_pooled, mean_bootstrap_half_width)

where ``seed_sd_pooled`` is the RMS of each method's across-seed standard
deviation of the point PR-AUC (pooling equal-n groups), and
``mean_bootstrap_half_width`` is the mean, across methods and seeds, of each
fit's own bootstrap CI half-width. Note per the test plan: `load_task_phase`
takes the run seed as its train/valid-split seed too, so this also captures
split-redraw variance, not just model-fit stochasticity -- that's the
variance we want here, and the artifact says so explicitly.

Usage: `python -m experiments.t1_noise_floor`
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from experiments._common import Timer, git_sha, write_artifact

METHODS = [
    "majority", "logreg_l2", "logreg_l1", "random_forest", "extra_trees",
    "hist_gbm", "knn", "svm_linear", "xgboost", "lightgbm", "catboost", "tfidf_logreg",
]
SEEDS = [42, 7, 123, 2024, 5]
RESULTS_DIR = "results/extracted/trialbench/results"
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
OUT_PATH = "results/experiments/t1_noise_floor.json"
UNDERPOWERED_THRESH = 0.05


def _load_records():
    recs = []
    for path in glob.glob(os.path.join(RUNS_DIR, "*.json")):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("method") in METHODS and rec.get("seed") in SEEDS and rec.get("status") == "ok":
            recs.append(rec)
    return recs


def main():
    with Timer() as t:
        recs = _load_records()

        # cell -> method -> seed -> {point_prauc, boot_halfwidth}
        cells = {}
        for r in recs:
            cell = (r["task"], r["phase"])
            cells.setdefault(cell, {}).setdefault(r["method"], {})[r["seed"]] = {
                "point_prauc": r["point"]["prauc"],
                "boot_halfwidth": (r["bootstrap"]["prauc"]["hi"] - r["bootstrap"]["prauc"]["lo"]) / 2.0,
            }

        expected_cells = None  # populated from whatever cells actually appear
        per_cell = []
        missing = []
        for (task, phase), by_method in sorted(cells.items()):
            method_sds, method_halfwidths, per_method_out = [], [], {}
            for method in METHODS:
                seed_vals = by_method.get(method, {})
                if len(seed_vals) < len(SEEDS):
                    missing.append({"task": task, "phase": phase, "method": method,
                                     "seeds_present": sorted(seed_vals.keys())})
                    continue
                points = np.array([seed_vals[s]["point_prauc"] for s in SEEDS], dtype=float)
                halfwidths = np.array([seed_vals[s]["boot_halfwidth"] for s in SEEDS], dtype=float)
                points = points[~np.isnan(points)]
                sd = float(np.std(points, ddof=1)) if len(points) > 1 else float("nan")
                method_sds.append(sd)
                method_halfwidths.append(float(np.nanmean(halfwidths)))
                per_method_out[method] = {
                    "seed_points": {int(s): float(seed_vals[s]["point_prauc"]) for s in SEEDS},
                    "seed_sd": sd,
                    "mean_boot_halfwidth": float(np.nanmean(halfwidths)),
                }

            method_sds = [s for s in method_sds if not np.isnan(s)]
            seed_sd_pooled = float(np.sqrt(np.mean(np.square(method_sds)))) if method_sds else float("nan")
            mean_boot_halfwidth = float(np.nanmean(method_halfwidths)) if method_halfwidths else float("nan")
            delta_cell = max(1.96 * seed_sd_pooled, mean_boot_halfwidth) if method_sds or method_halfwidths else float("nan")

            per_cell.append({
                "task": task, "phase": phase,
                "n_methods_with_all_seeds": len(per_method_out),
                "seed_sd_pooled": seed_sd_pooled,
                "mean_bootstrap_half_width": mean_boot_halfwidth,
                "delta_cell": delta_cell,
                "underpowered": bool(delta_cell > UNDERPOWERED_THRESH) if not np.isnan(delta_cell) else None,
                "per_method": per_method_out,
            })

    n_underpowered = sum(1 for c in per_cell if c["underpowered"])
    artifact = {
        "test_id": "T1",
        "claim_at_stake": "every numeric claim in Part 2",
        "inputs": {"methods": METHODS, "seeds": SEEDS, "results_dir": RESULTS_DIR,
                   "note": "load_task_phase uses the run seed for the train/valid "
                           "split too, so this variance includes split-redraw "
                           "variance, not just fit stochasticity -- intentional, "
                           "per the test plan."},
        "n_cells": len(per_cell),
        "delta_cell_table": per_cell,
        "missing_method_seed_combinations": missing,
        "decision_rule": "No later test may report an effect as real unless it "
                          "exceeds this cell's delta_cell and holds sign across "
                          "seeds. Cells with delta_cell > 0.05 PR-AUC are "
                          "underpowered and drop out of the claim set.",
        "verdict": f"CONFIRMED: noise floor computed for {len(per_cell)} cells; "
                    f"{n_underpowered} cell(s) exceed the 0.05 PR-AUC underpowered "
                    f"threshold and are excluded from later claims." if missing == [] else
                    f"PARTIAL: {len(missing)} (task,phase,method) combinations are "
                    f"missing one or more seeds -- see missing_method_seed_combinations; "
                    f"delta_cell for affected cells used only the methods that completed.",
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"wrote {OUT_PATH}: {len(per_cell)} cells, {n_underpowered} underpowered, "
          f"{len(missing)} missing combos")


if __name__ == "__main__":
    main()
