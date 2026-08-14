"""T8 -- Did the text column handicap the tabular pack?

newsletter-part2-test-plan.md, T8. Because `brief_summary/textblock` (and, on
trial-failure-reason-identification, `detailed_description/textblock`) wasn't
in `TEXT_COLS` before T7's repair, `TabularFeaturizer._is_raw_multimodal`
didn't exclude it from the tabular view -- it fell through to the categorical
branch and got smoothed target-encoded: near-unique free text treated as a
~1600-level categorical, separating classes in training and collapsing to a
near-constant encoding at test.

Compares the 12 Tier A methods x 20 cells x 5 seeds, refit after T7's repair
(`python -m src.run_benchmark --force`), against the frozen pre-repair
snapshot from T1 (`results/experiments/_snapshots/t1_baseline/runs`) --
paired per (task, phase, method, seed).

Usage: `python -m experiments.t8_tabular_after_repair`
(run *after* the --force rerun has completed)
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
POST_REPAIR_RUNS = "results/extracted/trialbench/results/runs"
BASELINE_RUNS = "results/experiments/_snapshots/t1_baseline/runs"
OUT_PATH = "results/experiments/t8_tabular_after_repair.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"


def _load_dir(runs_dir):
    out = {}
    for path in glob.glob(os.path.join(runs_dir, "*.json")):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("method") in METHODS and rec.get("seed") in SEEDS and rec.get("status") == "ok":
            out[(rec["task"], rec["phase"], rec["method"], rec["seed"])] = rec["point"]["prauc"]
    return out


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}

    with Timer() as t:
        post = _load_dir(POST_REPAIR_RUNS)
        pre = _load_dir(BASELINE_RUNS)

        cells = sorted({(k[0], k[1]) for k in post.keys()} | {(k[0], k[1]) for k in pre.keys()})
        per_cell = []
        for task, phase in cells:
            delta = delta_cells.get((task, phase), float("nan"))
            per_method = {}
            gains = []
            for method in METHODS:
                paired = []
                for seed in SEEDS:
                    key = (task, phase, method, seed)
                    if key in post and key in pre:
                        paired.append(post[key] - pre[key])
                if paired:
                    mean_gain = float(np.mean(paired))
                    per_method[method] = {"n_seeds_paired": len(paired), "mean_gain": mean_gain,
                                           "per_seed_gain": paired}
                    gains.append(mean_gain)
            cell_mean_gain = float(np.mean(gains)) if gains else float("nan")
            per_cell.append({
                "task": task, "phase": phase, "delta_cell": delta,
                "mean_gain_across_methods": cell_mean_gain,
                "gain_exceeds_delta_cell": bool(abs(cell_mean_gain) > delta) if not np.isnan(delta) and gains else None,
                "per_method": per_method,
            })
            print(f"  {task}/{phase}: mean_gain={cell_mean_gain:+.4f} delta_cell={delta:.4f}", flush=True)

    n_cells = len(per_cell)
    n_above = sum(1 for c in per_cell if c["gain_exceeds_delta_cell"] is True)
    if n_above >= 5:
        verdict = (f"CONFIRMED: mean gain across methods exceeds delta_cell on {n_above}/{n_cells} "
                    f"cells -- the tabular-pack handicap claim gets a number attached.")
    else:
        verdict = (f"the tabular pack was NOT measurably handicapped: mean gain exceeds delta_cell "
                    f"on only {n_above}/{n_cells} cells. Must be written as 'a smoothed label leak "
                    f"in training and dead weight at test, with no measurable effect on scores' -- "
                    f"the honest version, still worth a paragraph.")

    artifact = {
        "test_id": "T8",
        "claim_at_stake": "the tabular pack was quietly handicapped across 320 fits",
        "inputs": {"methods": METHODS, "seeds": SEEDS,
                   "post_repair_runs": POST_REPAIR_RUNS, "baseline_runs": BASELINE_RUNS},
        "n_cells": n_cells,
        "per_cell": per_cell,
        "n_cells_gain_above_delta_cell": n_above,
        "decision_rule": "If mean gain exceeds delta_cell in >=5/20 cells, CONFIRMED with a "
                          "number. Otherwise, write as leak-in-train/dead-weight-at-test with "
                          "no measurable score effect.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}: {n_above}/{n_cells} cells with gain > delta_cell")
    print(verdict)
    return artifact


if __name__ == "__main__":
    main()
