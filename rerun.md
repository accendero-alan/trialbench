# TabNet fix — validated on real data

_Follow-up to `report.md` §4 (TabNet vs. FT-Transformer investigation). That
report's mechanism check used a synthetic MLP proxy because its sandbox could
not reach real TrialBench data or install torch/pytorch-tabnet. This session
had both already available locally (from implementing Tier B), so the actual
experiment — `experiments/tabnet_fix_compare.py`, as that report left it —
was run for real instead of by proxy._
_Date: 2026-07-24_

---

## What was run

```bash
python -m src.data.download                                      # real data, local
python -m experiments.tabnet_fix_compare \
    --task mortality_rate_yn --phase Phase1 --seeds 42 7 123
```

Same task/phase the original report used for its gap analysis
(`mortality_rate_yn` / Phase1 is one of the 10 cells `tabnet` and
`ft_transformer` share). The script loads data through the repo's real
`load_task_phase` + `TabularFeaturizer` (identical, leakage-safe path used in
production), trains TabNet twice per seed — once matching the current
`src/methods/deep_tabular.py::TabNet` exactly ("original"), once with the
report's three proposed fixes ("fixed") — and scores both against the same
held-out test set. It writes nothing to `results/`.

The three fixes, applied together:
1. **Class weighting** — `weights=1` (pytorch-tabnet's inverse-frequency
   auto-balancing), previously unset.
2. **Feature scaling** — `StandardScaler` fit on train, applied to
   valid/test, before TabNet ever sees the data.
3. **Loss-based early stopping** — `eval_metric=["logloss"]` in place of
   pytorch-tabnet's default accuracy-ish metric, which rewards majority-class
   collapse on imbalanced data.

## Result (3 seeds: 42, 7, 123)

| Variant | PR-AUC | AUROC | frac. predicted positive @0.5 |
|---|---|---|---|
| majority / prior | 0.270 | — | — |
| **original** (current `deep_tabular.py`) | 0.500 ± 0.038 | 0.759 | 0.03 |
| **fixed** (scaled + class-weighted + logloss) | **0.555 ± 0.045** | **0.798** | 0.22 |

**delta PR-AUC (fixed − original) = +0.054. delta AUROC = +0.039.**

True test positive prevalence is ~0.27. Unfixed TabNet predicts positive on
only 3% of test cases — it is collapsing toward the majority class, exactly
as the original report diagnosed from the aggregate real-run numbers (tabnet
sitting within 0.05 of the majority prior in 4 of its 10 completed cells).
The fixed variant's predicted-positive rate (0.22) sits much closer to the
true rate, and both PR-AUC and AUROC improve.

## How this compares to the report's synthetic check

| | PR-AUC delta | Basis |
|---|---|---|
| Report §4.1 (synthetic MLP proxy) | **+0.33** | sklearn MLP, synthetic data matched to the mortality/Phase1 profile |
| This run (real data, real TabNet) | **+0.054** | `pytorch_tabnet.TabNetClassifier`, real `mortality_rate_yn`/Phase1 |

The real effect is smaller than the proxy predicted, which is expected and
not a contradiction: the proxy was a stylized MLP built specifically to
isolate the two failure modes (no built-in class weighting, no built-in
scaling) with nothing else going on. Real TabNet on real data has its own
architecture-specific dynamics, noise, and a much smaller sample than a
synthetic generator can produce on demand. What the proxy correctly predicted
was the **direction and mechanism** — scaling + rebalancing help, via
reduced majority-class collapse — and that holds up on real data. The report
itself flagged this as "directional evidence for the mechanism, not a
measurement"; this run supplies the measurement, on one representative cell.

## Caveats

- **Single task×phase.** `mortality_rate_yn`/Phase1 is one of the 10 cells
  `tabnet`/`ft_transformer` share, but the report's aggregate gap (0.512 vs
  0.730 PR-AUC) is measured across all 10. The fix should be re-validated
  on at least a couple more of those cells before treating +0.054 as
  representative of the full gap — this run does not claim the fixed TabNet
  now matches FT-Transformer everywhere, only that the fix measurably helps
  on real data in the cell tested.
- **3 seeds.** Enough to see the direction and a rough spread (±0.04), not
  enough for a tight confidence interval.
- Even fixed, TabNet (0.555) remains well below FT-Transformer's reported
  0.730 average on shared cells — the fix closes part of the gap, not all of
  it. Whether the residual gap is architectural or further-fixable
  implementation detail is still open.

## Recommended next step

Apply the same three changes to `src/methods/deep_tabular.py::TabNet` for
real, then re-run just that method across the full grid (`--methods tabnet
--force`) so the leaderboard reflects the fixed version rather than the one
this comparison shows is measurably worse. Not done in this session — this
file only documents the validation result, per what was asked.
