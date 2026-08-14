"""T4 -- Close the coverage hole.

newsletter-part2-test-plan.md, T4. FT-Transformer ran only 10 of 20 cells,
excluding all of failure_reason, all of patient_dropout_rate_yn, and 2 of 4
serious_adverse_rate_yn phases (confirmed by checking the run files directly
-- the "10 cells" in the plan's own cost estimate matches this exact gap,
not just "all of patient dropout and failure reason" as its claim text says).

Fills the gap: `run_benchmark --methods ft_transformer --seeds 42 7 123` on
the 10 missing cells (run separately from this script). This script re-ranks
FT-Transformer against the full Tier A pack + fixed TabNet (T3) on the
complete 20/20 shared cell set, using only the 3 seeds [42, 7, 123] common to
both the original 10 cells (which have 5 seeds from T1/T8) and the newly
filled 10 (which have 3), so the comparison is apples-to-apples.

Usage: `python -m experiments.t4_deep_full_grid` (after the fill-in run_benchmark calls)
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
DEEP_METHODS = ["ft_transformer", "tabnet"]
SEEDS = [42, 7, 123]
RUNS_DIR = "results/extracted/trialbench/results/runs"
OUT_PATH = "results/experiments/t4_deep_full_grid.json"

ORIGINAL_FT_CELLS = [
    ("outcome", p) for p in ["Phase1", "Phase2", "Phase3", "Phase4"]
] + [
    ("mortality_rate_yn", p) for p in ["Phase1", "Phase2", "Phase3", "Phase4"]
] + [("serious_adverse_rate_yn", "Phase1"), ("serious_adverse_rate_yn", "Phase2")]
ALL_CELLS = ORIGINAL_FT_CELLS + [
    ("serious_adverse_rate_yn", "Phase3"), ("serious_adverse_rate_yn", "Phase4"),
] + [("patient_dropout_rate_yn", p) for p in ["Phase1", "Phase2", "Phase3", "Phase4"]] + [
    ("failure_reason", p) for p in ["Phase1", "Phase2", "Phase3", "Phase4"]
]


def _load(method, task, phase, seed):
    path = os.path.join(RUNS_DIR, f"{task}__{phase}__{method}__seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    return rec["point"]["prauc"] if rec.get("status") == "ok" else None


def _mean_prauc(method, cells):
    vals = []
    for task, phase in cells:
        for seed in SEEDS:
            v = _load(method, task, phase, seed)
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _rank_of(method, all_methods_scores):
    ranked = sorted(all_methods_scores.items(), key=lambda kv: -kv[1] if not np.isnan(kv[1]) else np.inf)
    for i, (name, _) in enumerate(ranked, 1):
        if name == method:
            return i
    return None


def main():
    with Timer() as t:
        all_methods = METHODS + DEEP_METHODS
        scores_10cell = {m: _mean_prauc(m, ORIGINAL_FT_CELLS) for m in all_methods}
        scores_20cell = {m: _mean_prauc(m, ALL_CELLS) for m in all_methods}

        rank_10 = _rank_of("ft_transformer", scores_10cell)
        rank_20 = _rank_of("ft_transformer", scores_20cell)
        rank_shift = abs(rank_20 - rank_10) if rank_10 and rank_20 else None

        tabnet_rank_20 = _rank_of("tabnet", scores_20cell)

        per_cell_new = []
        for task, phase in ALL_CELLS:
            if (task, phase) not in ORIGINAL_FT_CELLS:
                ft_vals = [_load("ft_transformer", task, phase, s) for s in SEEDS]
                per_cell_new.append({"task": task, "phase": phase,
                                       "ft_transformer_mean_prauc": float(np.mean([v for v in ft_vals if v is not None]))
                                       if any(v is not None for v in ft_vals) else None,
                                       "n_seeds_present": sum(1 for v in ft_vals if v is not None)})

    n_missing = sum(1 for c in per_cell_new if c["n_seeds_present"] < len(SEEDS))
    verdict = (f"FT-Transformer 10-cell rank: {rank_10}/{len(all_methods)}. "
                f"Full 20-cell rank: {rank_20}/{len(all_methods)}. ")
    if rank_shift is not None and rank_shift > 2:
        verdict += (f"Rank differs by {rank_shift} positions (>2) -- Part 2 reports the "
                    f"full-grid rank ({rank_20}) and notes the partial-coverage number "
                    f"({rank_10}) as the artifact it is.")
    else:
        verdict += f"Rank differs by {rank_shift} position(s) (<=2) -- the 10-cell rank held up."

    artifact = {
        "test_id": "T4",
        "claim_at_stake": "every FT-Transformer comparison, based on 10 of 20 cells",
        "inputs": {"seeds": SEEDS, "methods": METHODS + DEEP_METHODS,
                   "original_ft_cells": ORIGINAL_FT_CELLS,
                   "note": "gap was serious_adverse_rate_yn Phase3-4 + all patient_dropout_rate_yn "
                           "+ all failure_reason (10 cells), confirmed directly from run files."},
        "scores_10cell_subset": scores_10cell, "scores_20cell_full": scores_20cell,
        "ft_transformer_rank_10cell": rank_10, "ft_transformer_rank_20cell_full": rank_20,
        "rank_shift": rank_shift, "tabnet_rank_20cell_full": tabnet_rank_20,
        "newly_filled_cells": per_cell_new, "n_newly_filled_cells_incomplete": n_missing,
        "decision_rule": "Re-rank against the full Tier A pack on the complete shared cell "
                          "set. If FT-Transformer's full-grid rank differs from its 10-cell "
                          "rank by more than 2 positions, report the full-grid rank and note "
                          "the partial-coverage number as the artifact it is.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}")
    print(verdict)
    return artifact


if __name__ == "__main__":
    main()
