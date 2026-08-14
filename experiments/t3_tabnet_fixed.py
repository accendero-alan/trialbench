"""T3 -- Merge the TabNet fix and re-measure.

newsletter-part2-test-plan.md, T3. As shipped, TabNet scored within noise of
the majority prior on several cells (mean 0.5115 vs 0.4319). The fix (class
weighting, scaling, loss-based early stopping -- see
`src/methods/deep_tabular.py::TabNet`, ported from
`experiments/tabnet_fix_compare.py`) moved PR-AUC +0.054 across 3 seeds on a
real-data subset (rerun.md) and was never merged into the live method.

Compares fixed TabNet (rerun with `--force` after the fix landed) against the
majority-prior baseline and the best GBM, per cell, from the same
post-T8-repair run directory (results/extracted/trialbench/results/runs) so
everything being compared reflects the same TEXT_COLS-repaired tabular view.

Usage: `python -m experiments.t3_tabnet_fixed`
(run *after* `run_benchmark --methods tabnet --force` has completed)
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from experiments._common import Timer, git_sha, write_artifact

SEEDS = [42, 7, 123, 2024, 5]
GBM_CANDIDATES = ["xgboost", "lightgbm", "catboost"]
RUNS_DIR = "results/extracted/trialbench/results/runs"
OUT_PATH = "results/experiments/t3_tabnet_fixed.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"


def _load(method, task, phase, seed):
    path = os.path.join(RUNS_DIR, f"{task}__{phase}__{method}__seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    return rec if rec.get("status") == "ok" else None


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    cells = sorted({(r["task"], r["phase"]) for r in t1["delta_cell_table"]})
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}

    with Timer() as t:
        per_cell = []
        for task, phase in cells:
            tabnet_recs = [_load("tabnet", task, phase, s) for s in SEEDS]
            tabnet_recs = [r for r in tabnet_recs if r is not None]
            if not tabnet_recs:
                per_cell.append({"task": task, "phase": phase, "status": "no_tabnet_runs"})
                continue
            tabnet_praucs = [r["point"]["prauc"] for r in tabnet_recs]
            tabnet_mean = float(np.mean(tabnet_praucs))

            majority_recs = [_load("majority", task, phase, s) for s in SEEDS]
            majority_recs = [r for r in majority_recs if r is not None]
            majority_mean = float(np.mean([r["point"]["prauc"] for r in majority_recs])) if majority_recs else float("nan")

            best_gbm_name, best_gbm_mean = None, -np.inf
            for method in GBM_CANDIDATES:
                recs = [_load(method, task, phase, s) for s in SEEDS]
                recs = [r for r in recs if r is not None]
                if recs:
                    m = float(np.mean([r["point"]["prauc"] for r in recs]))
                    if m > best_gbm_mean:
                        best_gbm_name, best_gbm_mean = method, m

            delta = delta_cells.get((task, phase), float("nan"))
            within_prior_noise = bool(abs(tabnet_mean - majority_mean) <= delta) if not np.isnan(delta) else None

            per_cell.append({
                "task": task, "phase": phase, "n_seeds": len(tabnet_recs),
                "tabnet_fixed_mean_prauc": tabnet_mean, "majority_prior_mean_prauc": majority_mean,
                "best_gbm": best_gbm_name, "best_gbm_mean_prauc": best_gbm_mean if best_gbm_name else None,
                "delta_cell": delta,
                "tabnet_within_delta_cell_of_majority_prior": within_prior_noise,
                "tabnet_minus_majority": tabnet_mean - majority_mean if not np.isnan(majority_mean) else None,
                "tabnet_minus_best_gbm": (tabnet_mean - best_gbm_mean) if best_gbm_name else None,
            })
            print(f"  {task}/{phase}: tabnet={tabnet_mean:.4f} majority={majority_mean:.4f} "
                  f"best_gbm({best_gbm_name})={best_gbm_mean:.4f} delta={delta:.4f} "
                  f"within_prior_noise={within_prior_noise}", flush=True)

    resolved = [c for c in per_cell if c.get("tabnet_within_delta_cell_of_majority_prior") is not None]
    n_within_prior = sum(1 for c in resolved if c["tabnet_within_delta_cell_of_majority_prior"])
    if n_within_prior >= 4:
        verdict = (f"CONFIRMED (stronger version, with the fix applied): fixed TabNet still "
                    f"sits within delta_cell of the majority prior on {n_within_prior}/{len(resolved)} "
                    f"resolved cells. Quote the fixed number, never 0.5115.")
    else:
        verdict = (f"CUT and replaced: fixed TabNet clears the majority prior everywhere except "
                    f"{n_within_prior}/{len(resolved)} cells -- 'the shipped config was broken' is "
                    f"the story, not 'TabNet as a class is barely a model'. Quote the fixed number.")

    artifact = {
        "test_id": "T3",
        "claim_at_stake": "TabNet as shipped is barely a model",
        "inputs": {"seeds": SEEDS, "runs_dir": RUNS_DIR,
                   "fix": "class weighting (weights=1) + StandardScaler(train-only) + "
                          "eval_metric=['logloss'] early stopping, ported from "
                          "experiments/tabnet_fix_compare.py into src/methods/deep_tabular.py::TabNet"},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "n_cells_within_delta_of_prior": n_within_prior,
        "n_cells_resolved": len(resolved),
        "decision_rule": "If fixed TabNet still sits within delta_cell of the majority prior in "
                          ">=4 cells, CONFIRMED (stronger, with fix applied). If it clears the "
                          "prior everywhere, CUT and replaced with 'shipped config was broken'. "
                          "Either way, quote the fixed number.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}: {n_within_prior}/{len(resolved)} cells within delta_cell of prior")
    print(verdict)
    return artifact


if __name__ == "__main__":
    main()
