"""T17 -- Confidence intervals on every blend lift.

newsletter-part2-test-plan.md, T17. The repo computed no CI on any blend
number; a first-pass paired bootstrap gave approval [+0.0026, +0.0522]
(clears zero) and mortality [-0.0017, +0.0281] (does not), using only 7 of 10
tabular methods. This redoes it on all 16 cells from T16, with the complete
10-method tabular consensus, reading T16's persisted predictions (no refit).

Design choice, stated here because the plan's procedure is one level less
specific than this: T16 gives 5 independent seed-fits per cell (same test
set, different train/valid carve, per `load_task_phase`). Each of the 10,000
resamples draws one shared set of test-row indices, applies it to *all 5*
seeds' predictions (a genuine paired bootstrap -- blended and best-single-source
scores for a resample always come from the same rows), computes each seed's
lift on those rows, and averages the 5 seed-lifts into one combined draw. This
propagates both test-set sampling noise and seed variability into a single
per-cell CI, consistent with T1's "seed variance + bootstrap width" framing.
Which source (tabular-consensus vs text) counts as "best single source" per
(cell, seed) is fixed from the real (non-resampled) data, per standard
bootstrap practice -- only the metric values are recomputed under resampling.

Usage: `python -m experiments.t17_blend_cis` (after T16 has completed)
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score

from experiments._blend_common import CELLS, SEEDS, TEXT_METHOD, load_cell_seed
from experiments._common import Timer, git_sha, write_artifact
from src.eval import metrics as M
from src.eval.predictions import load_predictions

N_RESAMPLES = 10000
RNG_SEED = 0
RESULTS_DIR = "results"
OUT_PATH = "results/experiments/t17_blend_cis.json"
T16_ARTIFACT = "results/experiments/t16_blend_full.json"


def _tfidf_boot_halfwidth(task, phase, seed=42):
    df = load_predictions(RESULTS_DIR, task, phase, TEXT_METHOD, seed, split="test")
    y, p = df["y_true"].to_numpy(), df["y_proba"].to_numpy()
    boot = M.bootstrap(y, p, "binary", 2, n_resamples=1000, seed=seed)["prauc"]
    return (boot["hi"] - boot["lo"]) / 2.0


def main():
    with open(T16_ARTIFACT) as f:
        t16 = json.load(f)
    by_cell_seed = {(r["task"], r["phase"], r["seed"]): r for r in t16["per_cell_seed"]}

    rng = np.random.default_rng(RNG_SEED)
    per_cell = []
    with Timer() as t:
        for task, phase in CELLS:
            seed_data = []
            for seed in SEEDS:
                y_test, tab_rank, txt_rank = load_cell_seed(task, phase, seed)
                rec = by_cell_seed[(task, phase, seed)]
                w_star = rec["w_star_valid"]
                best_is_text = rec["prauc_text_only"] >= rec["prauc_tabular_only"]
                seed_data.append({"y": y_test, "tab": tab_rank, "txt": txt_rank,
                                   "w_star": w_star, "best_is_text": best_is_text})
            n = len(seed_data[0]["y"])

            draws = np.empty(N_RESAMPLES)
            for i in range(N_RESAMPLES):
                idx = rng.integers(0, n, size=n)
                seed_lifts = []
                for sd in seed_data:
                    y_r = sd["y"][idx]
                    blended = (1 - sd["w_star"]) * sd["tab"][idx] + sd["w_star"] * sd["txt"][idx]
                    blended_prauc = average_precision_score(y_r, blended)
                    best_src = sd["txt"][idx] if sd["best_is_text"] else sd["tab"][idx]
                    best_prauc = average_precision_score(y_r, best_src)
                    seed_lifts.append(blended_prauc - best_prauc)
                draws[i] = float(np.mean(seed_lifts))

            mean_lift = float(np.mean(draws))
            lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
            clears_zero = bool(lo > 0)
            tfidf_halfwidth = _tfidf_boot_halfwidth(task, phase)
            ci_width = hi - lo
            ratio = (ci_width / (2 * tfidf_halfwidth)) if tfidf_halfwidth else float("nan")

            per_cell.append({
                "task": task, "phase": phase, "mean_lift": mean_lift,
                "ci_lo": lo, "ci_hi": hi, "clears_zero": clears_zero,
                "tfidf_bootstrap_halfwidth": tfidf_halfwidth,
                "lift_ci_width_over_tfidf_ci_width": ratio,
            })
            print(f"  {task}/{phase}: lift={mean_lift:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
                  f"clears_zero={clears_zero}", flush=True)

    n_clears = sum(1 for c in per_cell if c["clears_zero"])
    if n_clears < 4:
        verdict = (f"CUT to a Part 3 hypothesis: only {n_clears}/16 cells clear zero.")
    else:
        cleared = [c for c in per_cell if c["clears_zero"]]
        verdict = (f"{n_clears}/16 cells clear zero -- quote only these lifts: "
                    + ", ".join(f"{c['task']}/{c['phase']} {c['mean_lift']:+.4f}" for c in cleared))

    mortality_p1 = next((c for c in per_cell if c["task"] == "mortality_rate_yn" and c["phase"] == "Phase1"), None)
    if mortality_p1 is not None and not mortality_p1["clears_zero"]:
        verdict += (" | Part 1 correction needed: mortality/Phase1's blend lift "
                    f"({mortality_p1['mean_lift']:+.4f}, CI [{mortality_p1['ci_lo']:+.4f}, "
                    f"{mortality_p1['ci_hi']:+.4f}]) does not clear zero -- the cell where "
                    "text supposedly dominates is the cell where the improvement is not significant.")

    artifact = {
        "test_id": "T17",
        "claim_at_stake": "approval +0.0302 and mortality +0.0176 blend lifts",
        "inputs": {"n_resamples": N_RESAMPLES, "rng_seed": RNG_SEED, "seeds": SEEDS,
                   "note": "paired bootstrap over shared test-row resample indices, "
                           "applied to all 5 seeds' fits and averaged per resample -- "
                           "see module docstring for the full design rationale."},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "n_cells_clearing_zero": n_clears,
        "decision_rule": "Quote only lifts whose CI clears zero. If <4/16 cells clear "
                          "zero, CUT to a Part 3 hypothesis.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}: {n_clears}/16 cells clear zero")
    print(verdict)
    return artifact


if __name__ == "__main__":
    main()
