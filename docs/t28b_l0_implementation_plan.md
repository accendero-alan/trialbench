# T28b-L0 — disease-sensitivity null: implementation plan

Written 2026-08-28 against `experiments/t28b_opus_recall.py` @ `a38744ea`.

## Why this exists — the swap arm is invalid as run

`elicit_row` (line 584) hardcodes the arm:

```python
rendered = render_arm(row, "L7", str(row["nct_id"]))
```

Every arm goes through it, including the swap. Meanwhile `swap_disease`
(line 464) replaces **only** `row["condition"]`:

```python
swapped["condition"] = repr([donor_name])
```

And L7's filler (`src/data/serialize.py:243`) is:

```python
names = _parse_terms(row, "condition")
summary = _fmt(row.get("brief_summary/textblock"))
return f"{name_str}. {summary}"
```

So a swapped prompt carries the **donor** condition followed by the trial's
**original, unswapped** brief summary, which still describes the real
disease. The prompt contradicts itself, and the summary wins.

`opus_move_exactly_zero_rate = 0.81` is therefore the expected result of a
rendering bug, not evidence of memorisation. `quarantine_ordering: true`
must be withdrawn.

The spec (`docs/t28b_opus_recall_spec.md`) always said L1 for the swap arm.
Verification marked "L1 vs L7: confirmed correct" — it checked the spec, not
whether the code honours it. Worth noting as a review-process gap: a spec
assertion was verified against the spec.

---

## P1 — Parameterise the arm (blocking prerequisite)

`elicit_row` and `score_arm` take an `arm` argument; `"L7"` becomes the
default so the existing primary A/B/C contrasts are byte-identical to what
already ran and stay cache-valid.

```python
def elicit_row(client, results_dir, model_id, endpoint, row, meter, arm="L7"):
    rendered = render_arm(row, arm, str(row["nct_id"]))
```

**Regression guard:** re-run the primary contrasts after the change and
assert `meter.calls == 0, cache_hits == 1616`. Any nonzero `calls` means a
prompt changed and the A/B/C results are no longer the ones already
reported.

Add `arm` to every per-trial record so the artifact is self-describing.

---

## P2 — The three-point disease-sensitivity curve

Same 200 rows, same body, three disease slots. `_render_body` is identical
across arms by construction and `assert_body_identical` (standing rule 9)
already enforces it, so these three differ **only** in the disease slot:

| Arm | Disease slot |
|---|---|
| **L1** | the trial's own condition name(s) — the baseline |
| **L0** | `[condition withheld]` |
| **L1-swap** | a donor condition from a different ICD chapter |

L1 is the common baseline for both comparisons. Do **not** compare against
the existing L7 Arm A scores: L7 adds the whole brief summary, so an L7→L0
contrast conflates removing the disease with removing the summary.

**Note the body scrubber.** `_render_body` derives `mask_terms` from
`row["condition"]` and `row["condition_browse/mesh_term"]`. For the swap row,
`condition` is already the donor, so the scrubber masks the *donor's* terms
and leaves the original disease intact anywhere it appears in `_SCRUB_COLS`.
Build the swap row so mask terms come from the **original** condition, and
assert post-render that no original-disease term survives in the swapped
text.

---

## P3 — A reference with matched input composition

My earlier `r2_text_cols = ("condition",)` was wrong: it made the reference a
pure disease model, so a swap changed 100% of its input while changing one
line of Opus's. That asymmetry, not memorisation, produced most of the
0.84-vs-0.135 gap.

**Fix: vectorise the rendered prompt text itself.** Fit `tfidf_logreg` on
`render_arm(row, "L1").text` over the TrialBench training split, then score
the same three renderings for the 200 eval rows. Input parity is then exact,
byte for byte, because the reference reads the same string Opus reads.

---

## P4 — Repeat the curve on Arm B. This is the actual discriminator

**P2 alone cannot separate "Opus ignores disease" from "Opus recalls the
trial."** Both hypotheses predict full invariance across L1/L0/L1-swap. Only
the reference tells you whether there was disease signal available to ignore,
and even then the two readings stay entangled.

Running the identical three-point curve on **Arm B rows** (registered
pre-cutoff, results posted post-cutoff — outcome not memorisable) breaks the
tie:

- invariant on TrialBench, **sensitive** on Arm B → memorisation
- invariant on both → Opus does not use the disease slot for these endpoints,
  full stop

Arm B rows are already sampled and their labels already computed.

---

## Statistics

Per arm pair, per endpoint and pooled:

- movement distribution of `|p_baseline − p_variant|`, reported as a
  histogram, not a single rate
- **exactly-zero rate** and **nonzero-but-below-threshold rate**, separately
  (the existing `disease_swap_granularity` already splits these — reuse it)
- **rank-based movement** alongside absolute. Opus produced 22 distinct
  values across 500 predictions with a largest tie block of 20.2%, so a fixed
  0.05 threshold on that scale is coarse. Report mean absolute rank shift and
  Spearman correlation between baseline and variant scores.
- AUROC at L1, L0 and L1-swap, with paired CIs, so sensitivity is measured in
  predictive terms and not only in movement terms
- the same four for the P3 reference on identical rows

The L1 vs L0 AUROC delta is also T28's own L0 null on one cell, which makes
this a preview of PR-5's within-experiment null at negligible cost.

---

## Reading matrix

Interpretation requires the reference. Written before the run:

| Reference L1→L0 | Opus L1→L0 | Reading |
|---|---|---|
| insensitive | insensitive | Disease carries little signal for these endpoints on these rows. Opus's invariance is innocent, quarantine lifted. Consistent with T22 coming back INCONCLUSIVE on disease share. |
| sensitive | insensitive | Opus is not using predictive disease information. Either weak reading or substituted recall — **not separable here**; P4 decides it. |
| sensitive | sensitive | Normal behaviour. The swap probe works, and the 0.81 invariance was the L7 bug. |

P4 overrides: sensitivity on Arm B with invariance on TrialBench is the
memorisation signature regardless of which row above fires.

---

## Cost and sequencing

New prompts, so these miss the cache and bill:

| Block | Calls | Est. cost |
|---|---|---|
| P2 on TrialBench (L1, L0, L1-swap × 200) | 600 | ~$3.75 |
| P4 on Arm B (same three × 200) | 600 | ~$3.75 |
| **Total** | **1,200** | **~$7.50** |

Basis: T28b realised $10.09 over 1,609 calls = $0.0063/call. Use that, not
T28a's $0.0037 — T28a's prompts were much shorter, and my earlier estimate
was 2x low for exactly this reason.

Order: P1 → P2 → P3 → P4. Stop after P2/P3 if the reference is insensitive,
since row 1 of the matrix resolves it and P4 adds nothing.

---

## Artifact and test changes

- New artifact `results/experiments/t28b_l0_null.json`, not an overwrite of
  `t28b_opus_recall.json`.
- Dump per-trial predictions for all three arms, per the same rule that made
  this reanalysis free.
- **Withdraw `quarantine_ordering`** from `t28b_opus_recall.json` and replace
  it with a note pointing at the L7 bug and this document. Do not silently
  delete the field.
- Test: render L1 and L1-swap for a fixture row and assert the swapped text
  contains no original-disease term. That is the assertion whose absence
  cost the original swap arm.
- Test: assert `elicit_row` renders the arm it is passed. A one-line test
  that would have caught the hardcoded `"L7"`.
