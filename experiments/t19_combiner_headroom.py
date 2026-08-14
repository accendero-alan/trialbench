"""T19 -- Is tuning the combiner worth anything?

newsletter-part2-test-plan.md, T19. A fixed 50/50 blend reportedly captures
59% of approval's lift and 115% of mortality's (beating the tuned blend
outright on mortality); perfect weight selection is worth at most +0.0067 on
mortality, where every weight from 0.66 to 0.84 sits within 0.001 of the
validation peak -- 19 grid points validation cannot tell apart. Checks
whether that holds across all 16 cells, plus a genuine (not achievable)
oracle headroom figure.

Per cell (mean across 5 seeds, from T16's persisted curves):
  - fixed 50/50 lift (curve_test at w=0.5, the middle of T16's 101-point grid)
  - validation-tuned lift (T16's w*_valid selection)
  - test-argmax oracle lift (dishonest but recorded separately, per protocol)
  - width of the weight band within 0.001 of the validation-curve peak
  - a genuine oracle: HistGradientBoostingClassifier with monotonic
    constraints on [tabular_rank, text_rank], fit *on the test labels
    themselves* -- this is not achievable in practice and is reported purely
    as a theoretical combiner-headroom ceiling. (A true 1-D IsotonicRegression
    can't combine two input features and would trivially preserve whichever
    single ranking it's given -- a monotonic transform never changes PR-AUC,
    which only depends on rank order -- so this substitutes a 2-input
    monotonically-constrained model as the practical stand-in the plan's
    "isotonic stacker" language points at; flagged explicitly here.)

Usage: `python -m experiments.t19_combiner_headroom` (after T16)
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from experiments._blend_common import CELLS, SEEDS, load_cell_seed
from experiments._common import Timer, git_sha, write_artifact

OUT_PATH = "results/experiments/t19_combiner_headroom.json"
T16_ARTIFACT = "results/experiments/t16_blend_full.json"
BAND_TOL = 0.001


def _band_width(ws, curve_valid):
    ws, curve_valid = np.asarray(ws), np.asarray(curve_valid)
    peak = curve_valid.max()
    within = np.flatnonzero(curve_valid >= peak - BAND_TOL)
    return float(ws[within[-1]] - ws[within[0]])


def _oracle_headroom(y_test, tab_rank, txt_rank, best_source_prauc, seed):
    X = np.column_stack([tab_rank, txt_rank])
    clf = HistGradientBoostingClassifier(max_depth=2, max_iter=50, monotonic_cst=[1, 1],
                                          random_state=seed)
    clf.fit(X, y_test)  # deliberate oracle: fit on the test labels themselves
    proba = clf.predict_proba(X)[:, 1]
    stacker_prauc = average_precision_score(y_test, proba)
    return stacker_prauc - best_source_prauc


def main():
    with open(T16_ARTIFACT) as f:
        t16 = json.load(f)
    by_cell_seed = {(r["task"], r["phase"], r["seed"]): r for r in t16["per_cell_seed"]}

    with Timer() as t:
        per_cell = []
        for task, phase in CELLS:
            fixed_lifts, tuned_lifts, oracle_lifts, band_widths, stacker_headrooms = [], [], [], [], []
            for seed in SEEDS:
                rec = by_cell_seed[(task, phase, seed)]
                best_src = max(rec["prauc_tabular_only"], rec["prauc_text_only"])
                i_mid = len(rec["w"]) // 2  # w=0.5 exactly, N_GRID=101 -> index 50
                fixed_lifts.append(rec["curve_test"][i_mid] - best_src)
                tuned_lifts.append(rec["lift_over_best_single_source"])
                oracle_lifts.append(rec["prauc_test_at_argmax"] - best_src)
                band_widths.append(_band_width(rec["w"], rec["curve_valid"]))

                y_test, tab_rank, txt_rank = load_cell_seed(task, phase, seed)
                stacker_headrooms.append(_oracle_headroom(y_test, tab_rank, txt_rank, best_src, seed))

            fixed_mean, tuned_mean = float(np.mean(fixed_lifts)), float(np.mean(tuned_lifts))
            pct_captured = (fixed_mean / tuned_mean * 100) if tuned_mean else float("nan")
            per_cell.append({
                "task": task, "phase": phase,
                "fixed_5050_lift": fixed_mean, "validation_tuned_lift": tuned_mean,
                "test_argmax_oracle_lift": float(np.mean(oracle_lifts)),
                "pct_of_tuned_lift_captured_by_fixed": pct_captured,
                "mean_weight_band_width_within_0.001_of_valid_peak": float(np.mean(band_widths)),
                "stacker_oracle_headroom": float(np.mean(stacker_headrooms)),
            })
            print(f"  {task}/{phase}: fixed={fixed_mean:+.4f} tuned={tuned_mean:+.4f} "
                  f"({pct_captured:.0f}% captured) headroom={np.mean(stacker_headrooms):+.4f}", flush=True)

    valid_pct = [c["pct_of_tuned_lift_captured_by_fixed"] for c in per_cell
                 if not np.isnan(c["pct_of_tuned_lift_captured_by_fixed"])]
    mean_pct_captured = float(np.mean(valid_pct)) if valid_pct else float("nan")
    if mean_pct_captured >= 80:
        verdict = (f"Tuning the combiner is close to worthless: fixed 50/50 captures a mean "
                    f"{mean_pct_captured:.0f}% of tuned lift across cells -- recommend the fixed "
                    f"blend. Oracle headroom figures are what Part 3 is built around.")
    else:
        verdict = (f"Fixed 50/50 captures a mean {mean_pct_captured:.0f}% of tuned lift -- "
                    f"below the 80% bar, tuning the combiner is not obviously worthless across "
                    f"the full 16-cell grid.")

    artifact = {
        "test_id": "T19",
        "claim_at_stake": "fixed 50/50 captures 59%/115% of lift; perfect selection worth <=+0.0067",
        "inputs": {"seeds": SEEDS, "band_tolerance": BAND_TOL,
                   "oracle_note": "2-input monotonic HistGradientBoostingClassifier fit on test "
                                   "labels, substituting for a true 1-D isotonic stacker which "
                                   "can't combine two inputs and would trivially preserve rank "
                                   "order (see module docstring)."},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "mean_pct_of_tuned_lift_captured_by_fixed": mean_pct_captured,
        "decision_rule": "If fixed 50/50 captures >=80% of tuned lift on average, tuning is "
                          "close to worthless; recommend the fixed blend. Oracle headroom "
                          "figure becomes what Part 3 is built around.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}: mean {mean_pct_captured:.0f}% captured by fixed 50/50")
    print(verdict)
    return artifact


if __name__ == "__main__":
    main()
