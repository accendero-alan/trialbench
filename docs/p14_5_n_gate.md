# P14.5 — the fresh-slice n-gate (wave2-start-plan.md)

Run 2026-08-27 against the pinned AACT snapshot (2026-08-26, see
`docs/aact_label_rule.md`'s header for the checksum). Per §5: "P14.5's count
is therefore a launch gate on T28, not a caveat on T29. Run it first; it is
cheap and it is a query, not a model." This is that query, run before any
Wave 2 grid cell bills.

**Endpoints:** mortality (`mortality_rate_yn`) and SAE
(`serious_adverse_rate_yn`) — T29's scope per the test plan. A trial has a
"computable label" for an endpoint if `reported_event_totals` carries a row
for it (deaths / serious) at all — in practice this is every trial with any
posted results, since AACT emits all three `event_type` rows per group
whenever a group's results exist (verified: `n_trials_computable_label`
equals `n_trials_in_date_slice` in every row below).

**Two nested slices, per model, per `configs/bedrock_prices.yaml`'s pinned
`cutoff` field** (month-end reading — e.g. `2025-07` → 2025-07-31):

- **(a) registered pre-cutoff, results posted post-cutoff** — `study_first_posted_date < cutoff` and `results_first_posted_date > cutoff`.
- **(b) fully post-cutoff** — both dates after cutoff. This is T29's **primary** slice ("fresh for every model"), computed at the **latest** cutoff among the five ladder models (Claude Haiku 4.5, `2025-07`) so it is simultaneously post-cutoff for all five.

## Per-model counts

| Model | Cutoff | (a) registered-pre/results-post | (b) fully post-cutoff |
|---|---|---|---|
| Claude Opus 4.5 | 2025-03 | 9,046 | 208 |
| Claude Haiku 4.5 | 2025-07 | 6,644 | 105 |
| DeepSeek V3.2 | 2025-03 | 9,046 | 208 |
| Llama 4 Maverick | 2024-08 | 13,471 | 467 |
| Amazon Nova Lite | 2024-10 | 12,152 | 367 |

(Counts are per endpoint — mortality and SAE draw from the same date-sliced
trial set, since results-posting triggers both event-type rows together;
`n_trials_computable_label` = `n_trials_in_date_slice` for both endpoints in
every row.)

## The primary slice

**Fully post-cutoff at the latest ladder cutoff (2025-07-31), fresh for
every model: n = 105 trials, both endpoints.**

This is the number that answers T29's cross-model question. Cross-checked
directly against `studies.txt`: 6,749 trials in the whole AACT snapshot have
`results_first_posted_date` after 2025-07-31; of those, only 105 were also
*registered* after that date — the other 6,644 are older trials that
recently posted results, which is exactly slice (a)'s count. That agreement
confirms the join, not a bug.

## Reading against the plan's own expectation

`wave2-start-plan.md` T29 estimated **"a few thousand trials with computable
mortality/SAE labels per year of post-cutoff window."** The measured primary
slice, over a 13-month post-cutoff window, is **105** — roughly 20–60x
smaller than that estimate, and the plan's own §8 risk 3 named exactly this
possibility ("Thirteen months of post-cutoff window against results-posting
lag could leave too few labelled trials for sign-level replication").

Slice (a) is not thin at all (6,644–13,471 depending on model) but is *not*
the primary slice per T29's own text, precisely because it is not
simultaneously fresh for a model whose cutoff is later — a trial in slice
(a) could have been registered (and possibly discussed pre-results) before
an earlier-cutoff model's training data closed.

**This is the launch-gate finding P14.5 exists to produce.** n=105 pooled
across two endpoints is thin for a *pooled sign-level* check (not
impossible — T29 asks only for sign agreement on `L1−L2` and `L6−L1`, not
per-cell magnitude), but it is well below the plan's own comfort estimate
and should be weighed explicitly before T28 is funded, per §5: **"without
T29 nothing from T28 is falsifiable."** Options, not decided here:

1. **Proceed as designed**, reporting the primary slice's result at n=105
   with its width stated plainly, and lean on slice (a) — 6,644 trials,
   pre-cutoff registration but post-cutoff results — as a secondary,
   larger-n replication check even though it is not "fresh for every
   model" in the strict sense (it is fresh for the *label*, since the
   result could not have been in any model's training data regardless of
   whether the trial's existence was known).
2. **Widen the primary window** by using each model's *own* cutoff for its
   own T29 comparison (slice (b) per-model: 105–467) rather than insisting
   on one shared cross-model slice — trades cross-model comparability for
   n.
3. **Re-scope T28** before spending against it — e.g. drop models whose
   marginal contamination-vs-cost tradeoff is weakest, or treat T29 as
   directional-only evidence rather than a hard gate, stated as such in
   Part 3.

## Data

Full per-model, per-slice, per-endpoint table (this document's source data):
`data/external/aact_20260826/` (pinned snapshot) queried via
`src/data/aact.py`'s `load_table`. Raw output not checked into the repo —
reproducible in under a minute from the snapshot already on disk.
