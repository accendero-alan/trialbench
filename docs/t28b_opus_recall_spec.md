# T28b — Is Opus 4.5's predictive signal recall or reading?

Spec written 2026-08-28, before any call. Runs after T28a, before the T28
grid. Artifact: `results/experiments/t28b_opus_recall.json`.

## Claim at stake

T28a found Opus 4.5 discriminating well above chance on three of four tasks
with the full serialized trial in the prompt: mortality balanced accuracy
0.829 (Fisher p=0.00016, n=34), SAE 0.769 (p=0.00032, n=34), outcome 0.748
(p=0.00021, n=59), all clearing the Bonferroni-corrected alpha of 0.0025.
Title recall did not beat its shuffled-ID floor (1/200, Fisher p=1).

**The gap this test closes.** Title recall asks for a title from a bare NCT
ID. The outcome probe hands the model the *whole trial*, including
`brief_summary/textblock`. Recognising a fully-described trial is a far
easier retrieval task than producing a title from an opaque accession
number, so a null on title recall does **not** exclude
description-recognition followed by outcome recall. T28a cannot separate
"read the trial and predicted well" from "recognised the trial and
remembered the answer," and Opus is the rung where that distinction decides
whether the campaign's headline is usable.

## The design

The discriminator is **when the trial's results became public**, not whether
the trial existed. Three arms, one prompt, one endpoint set.

| Arm | Definition | Trial known to Opus? | Outcome memorisable? |
|---|---|---|---|
| **A** | TrialBench trials, results posted **before** 2025-03 | yes | **yes** |
| **B** | P14 slice (a): registered **pre**-cutoff, results posted **post**-cutoff | yes | **no** |
| **C** | P14 slice (b): registered and resulted **post**-cutoff | no | no |

Opus's cutoff is `2025-03` per `configs/bedrock_prices.yaml` (Bedrock card).
Per `docs/p14_5_n_gate.md`, slice (a) is **9,046** trials at that cutoff and
slice (b) is **208**.

> **On 208 versus the n=105 quoted elsewhere.** Both are correct and they are
> different cuts. 105 is the cross-model primary slice, computed at the
> *latest* ladder cutoff (Haiku, 2025-07) so it is simultaneously
> post-cutoff for every rung. 208 is Opus's own earlier cutoff (2025-03)
> alone, and is larger for exactly that reason. T28b is a single-model test,
> so Opus's own cutoff is the right one. Dropping Llama does not move the 105
> figure, since Haiku still holds the latest cutoff.

### Implementation gap — blocking

**Arm A needs a results-posting date, and no code reads one.**
`results_first_posted_date` appears in `src/data/aact.py:56`'s schema
whitelist and nowhere else in the codebase; the only place it has ever been
queried is the ad-hoc script behind `docs/p14_5_n_gate.md`, whose raw output
was never checked in (that document's own Data section says so). T28a's
`_join_registration_dates` reads `study_first_posted_date` — a registration
date, not a results date.

So Arm A's definition ("results posted before 2025-03") requires a new
loader function alongside the existing one, not a data pull. It should
return `results_first_posted_date` per `nct_id` from the pinned snapshot,
null-safe for trials with no posted results, and it is what partitions all
three arms. Write it, unit-test it against the snapshot, and commit it
before any call is issued.

**A → B is the decisive contrast, and it is the reason this test is worth
running.** It holds trial-identity knowledge constant while removing outcome
knowledge. A drop there cannot be explained by unfamiliarity or by the trial
being odd; the only thing that changed is whether the answer was published
in time to be read. This is `docs/p14_5_n_gate.md`'s own option 1 framing
("fresh for the *label*, since the result could not have been in any model's
training data regardless of whether the trial's existence was known"),
applied as an instrument rather than as a fallback.

A → C is secondary and ambiguous by construction, since it removes trial
familiarity *and* outcome availability at once. It is reported to separate
the two effects, not to carry the verdict.

**Predictions, stated before the run.**

- Recall hypothesis: A high, B drops sharply, C drops at least as much.
- Reading hypothesis: A ≈ B ≈ C, net of distribution shift.

## The reference arm — two configurations, both explicit

Every arm is also scored by `tfidf_logreg` fit on the TrialBench training
split and applied frozen. It learned from the training split and has no
memory of anything, so it calibrates how much of any A→B drop is
distribution shift rather than lost recall. This is T29's pre-registered
"while the classical arms hold" clause, run narrowly.

> **`text_cols` MUST be passed explicitly in both configurations below.**
> `TEXT_COLS` in `src/data/features.py` defaults to **ten** columns, and
> three of them — `brief_title`, `brief_summary/textblock`, `condition` —
> are on the repo's own `DISEASE_LEAK_COLS` list
> (`src/data/serialize.py:124`), the set `render_arm` refuses to put in a
> prompt body. Taking the default silently changes what the reference reads
> in a way that matters for R2 below. Relying on the default is the bug;
> naming the columns is the fix.

### R1 — the A→B / A→C reference (distribution-shift calibration)

`text_cols = ("condition", "brief_summary/textblock",
"eligibility/criteria/textblock")`.

Restricted to these three because `docs/p14_4_schema_slice.md` records
`phase` at 78.1% missing on the fresh slice against 0.0% in train,
`responsible_party` at 100% missing, and `icdcode` at 51% reconstructed. A
tabular or code-based reference would be reading those reconstruction gaps.
These three come straight from AACT's `brief_summaries.description`,
`eligibilities.criteria` and `conditions.name`, so they do not inherit the
problem.

**Disease leakage is deliberately not a concern for R1.** This arm only has
to be a predictor with no memory; reading the disease is exactly what
`tfidf_logreg` does in T22 and T25. The `DISEASE_LEAK_COLS` overlap matters
for R2, not here.

### R2 — the disease-swap reference (sensitivity calibration)

`text_cols = ("condition",)` — the swapped slot only.

The swap rewrites the disease slot. It does **not** rewrite the disease out
of `brief_summary/textblock` or `brief_title`, both of which routinely name
the condition in prose. A reference reading those fields therefore still
sees the *original* disease after the swap, and its measured sensitivity is
attenuated for a reason that has nothing to do with recall. That would
corrupt the one comparison the swap arm exists to make.

This is the same swap-consistency argument that makes L1 rather than L7 the
correct arm for the model side, applied to the reference.

If the swap also rewrites `condition_browse/mesh_term`, add that column and
say so in the artifact. Do not add any other `DISEASE_LEAK_COLS` member, and
do not add `keyword`, which is not on that list but frequently carries
condition terms.

### Why this test is runnable now while T29 is not

An LLM-only probe over text needs none of the 41 `TabularFeaturizer` columns
that block T29's frozen classical arms, and R1's three fields reconstruct
verbatim from AACT.

## Endpoints

`mortality_rate_yn` and `serious_adverse_rate_yn` only. These are the two
P14 recomputes labels for (`docs/aact_label_rule.md`). Opus's third
significant task, `outcome` (trial approval), has no AACT recomputation rule
and is **out of scope for this test** — a stated limitation, covering 2 of
its 3 significant tasks.

**Labels come from P14's recomputed rule on all three arms**, including A,
so the label definition is constant across the contrast rather than
switching from TrialBench's binarisation to ours at the arm boundary. The
rule of record is `docs/aact_label_rule.md` §1–§3; the confusion tables are
in `docs/p14_6_label_audit.md`, which reports disagreement against
TrialBench of 0.128% on mortality and 0.061% on SAE (agreement 99.87% and
99.94%). So this costs almost nothing on A and buys exact comparability.

## Sample

- Arm A: 500 trials, stratified by endpoint and label
- Arm B: 500 trials, stratified the same way
- Arm C: all 208

Identical prompt, elicitation, temperature and parser across arms. Prompts
byte-identical outside the trial content itself.

## Pre-flight checks — zero model calls, all blocking

1. **Text identity.** For 50 trials present in both TrialBench and the AACT
   snapshot, compare TrialBench's `brief_summary/textblock` against AACT's
   `brief_summaries.description` verbatim. Both derive from the same
   upstream record, so they should be identical. If they are, the
   prompt-content confound between arms is dead and this test gets much
   stronger. **If they differ systematically, stop** — the A→B contrast
   would be reading a reconstruction difference.
2. **n and label balance** per arm per endpoint, recorded before scoring.
3. **Covariate distributions** across arms: phase, enrollment, sponsor
   class, summary token length. Report; do not match on them (the reference
   arm absorbs shift). Flag any that separate the arms starkly.
4. **Power.** Compute the smallest A→B balanced-accuracy drop detectable at
   the chosen n by paired bootstrap, and record it. If a drop from 0.83 to
   0.75 is not detectable, raise n on A and B before running — slice (a)
   has 9,046 trials, so n is cheap here in a way it never is for T29.

## Elicitation

Verbalized 0–100 probability, temperature 0, JSON, per P13 — **not** T28a's
Yes/No. The probability form is what makes the disease-swap arm's "moves
less than 0.05" threshold meaningful, and it matches what T28 will actually
run, so a signal confirmed here is a signal in T28's own units.

Parse failures → 0.5 and logged, per P13. **Record the parse-failure rate
per arm.** Llama was dropped from the ladder for a 30.5% non-answer rate on
a strictly easier contract; Opus answered 200/200 on T28a, but the JSON
contract is harder and this is where that would surface.

## Secondary arm — disease swap

200 Arm A trials re-run as L1 with a different ICD chapter's disease
substituted. Report the insensitivity rate (predictions moving < 0.05)
against the test plan's pre-registered 20% threshold, and run the identical
swap through the frozen reference so the model's sensitivity is read against
a predictor that cannot be recalling.

Weakness to state in the artifact: T22 came back INCONCLUSIVE on disease
share, so we have no strong prior that disease is highly predictive for
these endpoints. Low sensitivity is therefore not recall on its own — it is
only interpretable against the reference's sensitivity on the same rows.

## Decision rules — written before the data

**Primary, on the A→B contrast, mortality and SAE pooled, paired bootstrap
on `nct_id`:**

| Result | Reading | Consequence |
|---|---|---|
| Opus drops by more than its paired CI, reference's drop does not clear its CI | **Outcome recall demonstrated** | Opus's TrialBench figures are recall-inflated. No absolute Opus number from TrialBench is quoted. T28's Opus arm is reported on fresh data or not at all. |
| Both drop comparably | Distribution shift, not recall | Report both drops. Opus's arm orderings stand, with the shift noted. |
| Neither drops | **No outcome recall detected** | Opus's signal is reading. T28 proceeds with Opus as rung 1 as designed. |
| Opus drops, reference drops *more* | Opus is more robust than the reference | Report as-is; not a contamination finding. |

**Secondary:** if C drops substantially more than B, the excess is
trial-familiarity effect on top of whatever B shows, reported separately and
not folded into the primary verdict.

**Tertiary:** disease-swap insensitivity above 20% quarantines Opus's arm
ordering to an appendix, per the test plan's existing rule.

**The upside is pre-registered too.** If Opus holds on Arm B, the finding is
that a frontier model predicts mortality and SAE at roughly 0.8 balanced
accuracy on trials whose results it demonstrably could not have read. That
is a stronger and more publishable result than the representation story it
was meant to protect, and it should not be buried as a passed safety check.

## Cost

At Opus's realised T28a rate of $0.0037 per call: 1,208 trials plus the
200-trial swap arm is roughly **$5.20**, plus the frozen reference at no API
cost. If the power check raises Arm A and B to 1,000 each, about $8.20.

Negligible against the $234 grid this gates, which is the point.

## What this test does not settle

- `outcome`/trial approval, Opus's third significant task, for want of an
  AACT label rule.
- The other four rungs. If Opus is clean, that is evidence about the rung
  most likely to be contaminated (capacity scaling was the plan's own
  prediction), not proof about the rest. Haiku and Nova showed no signal at
  all on T28a, so there is nothing to attribute for them; DeepSeek's single
  mortality result would warrant the same test if its arm ordering ends up
  load-bearing.
- T29. This is not a substitute. T29 replicates *arm orderings*; T28b asks
  only whether one model's absolute skill survives the removal of outcome
  knowledge.
