# Where ft_transformer's accuracy actually comes from

_Follow-up to `report.md` §3.2 (the "Overall" ranking is not apples-to-apples).
That finding showed `ft_transformer`'s #1 ranking was an artifact of partial
grid coverage. This is the natural next question: even on the cells it did
run, is its strong score genuine architectural capability, or something more
mundane? Investigated locally with real data — no EC2 rerun needed._
_Date: 2026-07-24_

---

## What was run

```bash
python -m experiments.ft_transformer_importance --task mortality_rate_yn --phase Phase1
```

New standalone script (`experiments/ft_transformer_importance.py`, alongside
`experiments/tabnet_fix_compare.py` — same pattern: real loader/featurizer,
writes nothing to `results/`), on the same cell used for the TabNet
comparison in `rerun.md`. Two checks:

1. **Permutation importance** — shuffle one feature column at a time in the
   test set, measure the PR-AUC drop. Names the columns the model actually
   relies on.
2. **Cross-method prediction correlation** — Spearman correlation between
   `ft_transformer`'s test-set scores and `extra_trees`, `random_forest`, and
   `tfidf_logreg` on the identical split. High correlation means it's
   recovering the same signal as everyone else; low correlation would mean
   it's finding something distinctive.

`ft_transformer` baseline on this cell: **test PR-AUC = 0.8088**.

## Finding 1: accuracy is concentrated in one feature

| Feature | PR-AUC drop when shuffled (mean of 8 repeats) |
|---|---|
| `eligibility/healthy_volunteers__te` | **+0.2788** |
| `eligibility/maximum_age__te` | +0.1058 |
| `enrollment` | +0.0946 |
| `MaskingType-Investigator` | +0.0651 |
| `study_design_info/primary_purpose__te` | +0.0495 |
| `study_design_info/masking_num` | +0.0429 |
| `Other intervention Number` | +0.0073 |
| `eligibility/minimum_age` | +0.0051 |
| (35 more features) | each <0.004; 30 of 43 total are <0.001 |

**One feature — whether the trial accepts healthy volunteers — accounts for
more than a third of the model's entire PR-AUC** (0.279 of 0.809). Adding the
next five features explains most of the rest. The remaining 30 of 43 features
(70% of the input) contribute almost nothing individually. This is not the
signature of a model exploiting rich multimodal feature interactions — it's
one dominant tabular signal carrying the score, with the transformer
architecture along for the ride.

`eligibility/healthy_volunteers__te` is target-mean encoded (smoothed,
fit-on-train per `TabularFeaturizer`, so not a test-leakage issue in the
strict sense) — but for a low-cardinality Yes/No field, that encoding is
close to "the historical outcome rate for trials with this flag," which is a
blunt, easily-recovered signal, not something requiring a transformer to
find.

## Finding 2: every other strong method finds the same signal

| Method | Test PR-AUC | Spearman corr. vs. ft_transformer |
|---|---|---|
| extra_trees | 0.8168 | 0.728 |
| random_forest | 0.8039 | 0.805 |
| **tfidf_logreg** | **0.8554** | 0.782 |

All three are highly correlated with `ft_transformer`'s predictions
(0.73–0.81 Spearman) — and **`tfidf_logreg` scores *higher* than
`ft_transformer`** on this cell, using plain TF-IDF + logistic regression on
free text, no deep learning at all. That's consistent with the eligibility
text itself ("healthy volunteers," age ranges) carrying the same signal the
target-encoded tabular column carries — every method with any access to it,
tabular or text, finds it and clusters in the same 0.80–0.86 band.

## What this means

`ft_transformer`'s accuracy is not evidence of a real capability edge from
the transformer architecture. It is dominated by one strongly-predictive,
easily-encoded eligibility flag that classical trees, linear text models, and
the transformer all recover similarly — which is exactly why they all land
in the same narrow PR-AUC range regardless of method sophistication. This
sharpens `report.md`'s conclusion from a different angle: not only is
`ft_transformer`'s #1 overall rank an artifact of incomplete grid coverage
(§3.2), but its strong score on the cells it *did* complete isn't really
measuring what a "Tier B deep-tabular method" comparison is meant to
measure. The honest read is still: **`tfidf_logreg` is the strongest,
most-complete, and — per this check — least mysteriously-strong method.**

## Caveats

- **Single task×phase, single seed.** `mortality_rate_yn`/Phase1 only. Feature
  reliance could differ on other cells (e.g. `failure_reason`, where
  `ft_transformer` has no completed cells at all to check).
- **Permutation importance measures reliance, not causality.** It says the
  model's output depends heavily on this column's value; it doesn't by
  itself prove the column is spuriously predictive rather than genuinely
  informative (healthy-volunteer Phase 1 trials plausibly do have a
  different mortality profile than patient trials — that could be real
  signal, just not one that needed a transformer to find).
- **Correlation, not identity.** 0.73–0.81 Spearman is high but not 1.0 —
  the methods agree substantially, not completely; some of each method's
  score likely does come from smaller, method-specific signal.

## Related artifacts this session

- `rerun.md` — real-data validation of the TabNet fix (class weighting +
  scaling + logloss early stopping), same investigative thread.
- `experiments/ft_transformer_importance.py` — the script behind this
  writeup; reusable on any other task/phase.
- `experiments/tabnet_fix_compare.py` — original-vs-fixed TabNet comparison
  (from the earlier `report.md` investigation).
