"""T18 -- Does the gain track decorrelation?

newsletter-part2-test-plan.md, T18. The mechanism behind Surprise 4, currently
n=2: Spearman 0.481 on approval with a significant +0.030 lift, 0.790 on
mortality with a non-significant +0.018. With 16 points instead of 2, does
blend lift regress on tabular/text rank disagreement?

Per cell: mean (across 5 seeds) Spearman correlation between the tabular-consensus
test rank and the TF-IDF test rank (from T16's persisted predictions), regressed
against T17's mean blend lift across the 16 cells. Slope CI via a nonparametric
bootstrap over cells (resample the 16 cells with replacement, refit OLS each
time) -- appropriate given n=16 is small and no distributional assumption on
the lift/correlation relationship is safe to make.

Usage: `python -m experiments.t18_decorrelation_regression` (after T16 and T17)
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from experiments._blend_common import CELLS, SEEDS, load_cell_seed
from experiments._common import Timer, git_sha, write_artifact

RNG_SEED = 1
N_BOOT = 10000
OUT_PATH = "results/experiments/t18_decorrelation_regression.json"
T17_ARTIFACT = "results/experiments/t17_blend_cis.json"


def _ols_slope(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return 0.0, ym
    slope = np.sum((x - xm) * (y - ym)) / denom
    intercept = ym - slope * xm
    return float(slope), float(intercept)


def main():
    with open(T17_ARTIFACT) as f:
        t17 = json.load(f)
    lift_by_cell = {(r["task"], r["phase"]): r["mean_lift"] for r in t17["per_cell"]}

    with Timer() as t:
        per_cell = []
        for task, phase in CELLS:
            corrs = []
            for seed in SEEDS:
                _, tab_rank, txt_rank = load_cell_seed(task, phase, seed)
                rho, _ = spearmanr(tab_rank, txt_rank)
                corrs.append(rho)
            mean_corr = float(np.mean(corrs))
            per_cell.append({"task": task, "phase": phase, "mean_spearman": mean_corr,
                              "mean_lift": lift_by_cell[(task, phase)]})
            print(f"  {task}/{phase}: spearman={mean_corr:.4f} lift={lift_by_cell[(task, phase)]:+.4f}", flush=True)

        xs = [c["mean_spearman"] for c in per_cell]
        ys = [c["mean_lift"] for c in per_cell]
        slope, intercept = _ols_slope(xs, ys)

        rng = np.random.default_rng(RNG_SEED)
        n = len(per_cell)
        boot_slopes = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            s, _ = _ols_slope([xs[j] for j in idx], [ys[j] for j in idx])
            boot_slopes[i] = s
        slope_lo, slope_hi = float(np.percentile(boot_slopes, 2.5)), float(np.percentile(boot_slopes, 97.5))
        slope_clears_zero = bool(slope_lo > 0 or slope_hi < 0)

    if slope_clears_zero:
        verdict = (f"CONFIRMED: slope={slope:+.4f}, bootstrap CI [{slope_lo:+.4f}, {slope_hi:+.4f}] "
                    f"clears zero -- decorrelation is a usable rule of thumb.")
    else:
        verdict = (f"CUT: slope={slope:+.4f}, bootstrap CI [{slope_lo:+.4f}, {slope_hi:+.4f}] "
                    f"crosses zero -- mechanism handed to Part 3 explicitly as untested, "
                    f"the n=2 line removed.")

    artifact = {
        "test_id": "T18",
        "claim_at_stake": "gain tracks decorrelation (Spearman 0.481/approval vs 0.790/mortality, n=2)",
        "inputs": {"seeds": SEEDS, "n_boot": N_BOOT, "rng_seed": RNG_SEED,
                   "note": "OLS slope of mean_lift on mean_spearman across the 16 cells; "
                           "CI via nonparametric bootstrap resampling of cells."},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "slope": slope, "intercept": intercept,
        "slope_ci_lo": slope_lo, "slope_ci_hi": slope_hi, "slope_clears_zero": slope_clears_zero,
        "decision_rule": "If the slope CI clears zero, CONFIRMED -- becomes a usable rule of "
                          "thumb. If it crosses zero, CUT and handed to Part 3 as untested.",
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
