# Is clinical_embeddings' compute cost buying anything? No.

_Follow-up to `text_vs_tabular_signal.md`, which found this benchmark's text
signal is substantially keyword-driven (specific disease names, "healthy
volunteers" boilerplate). Natural next question: does a frozen clinical BERT
encoder actually do anything TF-IDF doesn't, given it costs roughly
100–1,000x more compute per cell? Checked against the existing pulled-down
results — no rerun needed._
_Date: 2026-07-24_

---

## What was checked

`results/extracted/trialbench/results/results_long.csv` (the archive pulled
from EC2), every cell where **both** `tfidf_logreg` and `clinical_embeddings`
have a completed (`status: ok`) result. Per `report.md`, `clinical_embeddings`
has only ever completed 9 of the 20 task×phase cells it would need for full
coverage — this table is **all 9**, not a selected subset.

## Result: 0 wins out of 9

| Task / Phase | tfidf_logreg | clinical_embeddings | delta (BERT − TF-IDF) |
|---|---|---|---|
| serious_adverse_rate_yn / Phase1 | 0.909 | 0.824 | −0.085 |
| mortality_rate_yn / Phase1 | 0.857 | 0.752 | −0.105 |
| mortality_rate_yn / Phase2 | 0.853 | 0.773 | −0.079 |
| mortality_rate_yn / Phase3 | 0.852 | 0.751 | −0.101 |
| outcome / Phase1 | 0.662 | 0.520 | −0.142 |
| outcome / Phase2 | 0.598 | 0.497 | −0.101 |
| outcome / Phase3 | 0.824 | 0.712 | −0.112 |
| outcome / Phase4 | 0.621 | 0.504 | −0.117 |
| mortality_rate_yn / Phase4 | 0.504 | 0.294 | −0.210 |

**`clinical_embeddings` loses to `tfidf_logreg` in all 9 of 9 cells.** Mean
delta = **−0.117 PR-AUC**; best case for BERT is still a loss (−0.079,
`mortality_rate_yn`/Phase2); worst case is −0.210
(`mortality_rate_yn`/Phase4). There is no cell, anywhere in the current
results, where the frozen-BERT path outperforms plain TF-IDF + logistic
regression.

## Cost side of the ledger

Per `report.md`'s timing data, `clinical_embeddings` averages **~51 minutes
per cell, peaking at 2h10m** (Bio_ClinicalBERT CPU inference), against
`tfidf_logreg`'s ~1–2 **seconds**. That's on the order of 1,000–5,000x the
wall-clock cost for a method that is, so far, never the winner. It was also
the direct cause of the Tier B/C grid getting cut off mid-run on EC2 several
times this session — its cost is not a rounding error, it materially limited
how much of the benchmark could complete.

## Why: this looks like a mismatch between the method and the task, not a fluke

Three things point the same direction:

1. **The task's signal is keyword-driven, not contextual.**
   `text_vs_tabular_signal.md` found `tfidf_logreg`'s strongest learned terms
   are specific, discriminative tokens — disease names (`leukemia`,
   `metastatic`), boilerplate phrases (`healthy`, `single dose`, `bmi`).
   TF-IDF is built exactly for "does this document contain these
   informative words," and lets `LogisticRegression` assign each one a
   large, task-specific coefficient directly.
2. **The embeddings are frozen, and mean-pooled.**
   `clinical_embeddings` never fine-tunes `Bio_ClinicalBERT` on this task
   (`src/methods/text_nlp.py`) — it mean-pools every token in the
   (often-long) eligibility/summary text into one fixed 768-dim vector, then
   fits only a linear head on top. Averaging over a full document likely
   dilutes a handful of salient keywords among a lot of administrative
   boilerplate, in a way TF-IDF's explicit per-term weighting doesn't.
3. **Domain and data-size mismatch.** `Bio_ClinicalBERT` was pretrained on
   MIMIC-III ICU discharge notes — stylistically distant from
   clinicaltrials.gov eligibility criteria. And these are small training
   sets (hundreds to a few thousand rows per cell) — exactly the regime
   where simple bag-of-words methods often hold their own against
   frozen, un-fine-tuned embeddings.

## Recommendation

As currently implemented, `clinical_embeddings` isn't earning its place in a
default run: it's slower than every other method by orders of magnitude and
has not once produced the best score. Options, in order of effort:

1. **Cheapest:** drop it from the default `configs/benchmark.yaml` grid, or
   mark it explicitly "opt-in / experimental" rather than on-by-default,
   so a normal run isn't gated behind hour-plus cells for a method that
   isn't winning.
2. **Medium:** try a lighter/faster embedding model (a MiniLM-scale
   sentence-transformer, per `PLAN.md`'s original Tier C menu) to see if
   the cost/accuracy trade improves — still frozen, but much cheaper per
   cell, so at least the cost side of the ledger improves even if it
   doesn't start winning.
3. **Real fix, most effort:** the task calling for a frozen encoder was
   itself the design choice most likely to hurt here — fine-tuning
   end-to-end (PLAN.md already flags this as Tier D / GPU-preferred) would
   let the model learn to weight exactly the keywords TF-IDF is finding by
   hand, and might be the only way for a transformer-based approach to
   actually beat it on this benchmark.

## Caveats

- **9 cells, 3 tasks.** No data at all for `patient_dropout_rate_yn` or
  `failure_reason` (that pass was cut off before reaching them — see
  `report.md` §3.1), so "0/9" describes `outcome`, `mortality_rate_yn`, and
  `serious_adverse_rate_yn`/Phase1 only, not a verdict on every task.
  `serious_adverse_rate_yn` itself only has one comparable cell (Phase1)
  since `clinical_embeddings` never reached its later phases.
- **One configuration tested.** This indicts the *specific* setup (frozen
  `Bio_ClinicalBERT`, mean pooling, linear head, `MAX_LENGTH=256`), not
  clinical embeddings as a category. A different base model, pooling
  strategy, or fine-tuning approach could behave differently.
- **Single seed throughout** — no variance estimate on any of these deltas,
  though at −0.08 to −0.21 with zero wins across 9 independent cells, the
  direction is not in doubt even without confidence intervals.

## Related artifacts this session

- `text_vs_tabular_signal.md` — the keyword-driven-signal finding this
  builds on.
- `healthy_volunteers.md`, `rerun.md` — the earlier permutation-importance
  and TabNet-fix investigations in the same thread.
- `report.md` — original coverage/timing data for `clinical_embeddings`
  (51min avg / 2h10m peak per cell).
