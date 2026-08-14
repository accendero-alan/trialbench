# Report figures

Five figures for the write-up, each as **PNG** (300 dpi, print) + **SVG** (web) +
a **`_table.csv`** twin carrying every plotted value, so no number in the piece is
only reachable by reading pixels.

Four are free from the saved run records; one needed a local rerun.

| Figure | Says | Source |
|---|---|---|
| `fig1_coverage_grid` | 279 of 320 cells have a record; the four deep methods are absent from dropout and failure reason entirely | run JSONs |
| `fig2_forest_outcome_p1` | all 9 other top-10 methods overlap the leader's CI — the ordering is inside the noise | run JSONs |
| `fig3_blend_curve` | tabular+text fusion beats either alone, and the optimal mix flips between tasks | **rerun** (`blend_curve_probe.py`) |
| `fig4_ranking_swap` | the published #1 is an artifact of partial coverage; on shared cells the order flips | run JSONs |
| `fig5_skill_vs_prevalence` | most of the apparent difficulty spread is class imbalance | run JSONs + `data/` |

## Reproduce

The blend curve refits 11 methods on two cells (~2 min, CPU):

```bash
python -m experiments.blend_curve_probe --out figures/data/blend_curve.json
```

Per-cell class prevalence for fig 5 (reads `data/`, ~1 min):

```bash
python -m experiments.make_figures --write-prevalence figures/data/prevalence.json
```

Then render everything:

```bash
python -m experiments.make_figures --blend-json figures/data/blend_curve.json --prevalence-json figures/data/prevalence.json --out-dir figures
```

## Reading notes

- **fig 2 is one cell** (`outcome`/Phase 1), not a task average — a bootstrap CI is
  defined per cell, so pooling phases would not give an honest interval. That cell
  was chosen because it has the *most* overlap (9 of 9); other cells overlap less.
  `serious_adverse`/Phase 3 and `mortality`/Phase 2 are the tightest, with 1 and 0
  overlapping the leader — the "inside the noise" claim is not uniform across the
  grid and shouldn't be stated as if it were.
- **fig 3 marks the validation-selected `w*`, not the test argmax.** For mortality
  the two differ (w*=0.79 → 0.873; test peaks at w=0.57 → 0.880), so the marked dot
  deliberately sits below the visible peak and the test peak is marked with an `x`.
  Quoting the test peak as the result would be selecting on test.
- **fig 4's grey lines are labelled but de-collided** — labels are nudged apart and
  joined to their point by a hairline, so a label's vertical position is
  approximate. Exact values are in the CSV.
- **fig 5's x-axis is the random-classifier PR-AUC** (positive rate for binary; mean
  per-class prevalence over classes present in test for `failure_reason`, which is
  0.25 in all four phases). The diagonal is therefore chance, and the stem length
  is skill. Y is the *best* method in each cell.
- Figures use the validated two-hue categorical palette (blue `#2a78d6`, orange
  `#eb6834`); all-pairs CVD ΔE 24.7, normal-vision 33.6. Everything else is
  emphasis or recessive ink, so no figure needs a third hue.
