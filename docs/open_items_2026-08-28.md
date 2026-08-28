# Open items — review of the 2026-08-28 overnight run

Reviewed: `t25_dense_vs_sparse.json`, `t26_unseen_disease.json`,
`t27_mapping_noise.json`, `t27_disagreement_sample.csv`, `src/splits/`
(P12), and the tracked diffs to `src/methods/text_nlp.py` and
`requirements-extended.txt`.

Companion document: `docs/t28a_fixes_before_full_run.md` (F1–F9, written
2026-08-27, none landed yet — see A5).

---

## Actions

### A1 — Commit the tree and stamp the artifact SHAs
**Standing artifact rule, now missed four times in a row.**

| Artifact | Recorded SHA |
|---|---|
| `t28a_probe_gate.json` | `80c90a0bf1581dc2f8eb2eeeb39c0f50c15f4424-dirty` |
| `t25_dense_vs_sparse.json` | `f4970dda94d6e122a29d9083b6ebf768ee0f5801-dirty` |
| `t26_unseen_disease.json` | `f4970dda94d6e122a29d9083b6ebf768ee0f5801-dirty` |
| `t27_mapping_noise.json` | `f4970dda94d6e122a29d9083b6ebf768ee0f5801-dirty` |

None of these results can currently be tied to the code that produced them.
T25 alone cost 19,568 seconds (5.4 h), so **do not re-run for provenance.**
Commit the current tree, then add a `provenance_note` to each of the four
artifacts recording that it was produced from that commit's tree. That is
honest and costs nothing.

- [ ] Commit `src/splits/`, `experiments/t25*.py`, `t26*.py`, `t27*.py`, the
      `text_nlp.py` / `requirements-extended.txt` diffs, and both docs
- [ ] Add `provenance_note` to the four artifacts above

### A2 — T27 claims an error estimate it does not have
**Blocking on the T27 write-up. Do not let this reach Part 3 as written.**

`hand_adjudication_status` reads `"NOT PERFORMED"`. The decision rule still
describes the agreement table as *"the first published error estimate for the
mapping pipeline that HINT, TOP, and TrialBench all share."*

It is not an error estimate. It is a **disagreement rate between two mappers**,
and disagreement is silent on which one is wrong. Full-code Jaccard of 0.036
to 0.073 is equally consistent with:

- the shipped lexical mapping being very noisy (the plan's prior), **or**
- the SapBERT nearest-neighbour remapper being bad.

The score re-run is weak evidence for the second reading, not the first:
`rerun_directions` gives `patient_dropout_rate_yn/Phase1: "remap_worse"` and
`indistinguishable` elsewhere. A remap that hurts is not a cleaner mapping.

Illustrative only, from row 1 of the disagreement sample: NCT03746522
(Alström Syndrome; Bardet-Biedl Syndrome) is mapped by the shipped mapper to
`D68, D69` (coagulation defects) and by the remap to `Q87` (other congenital
malformation syndromes). Q87 is the more defensible reading. One row proves
nothing, which is precisely why (c) exists in the procedure.

Two acceptable resolutions, pick one:

- [ ] **Do the adjudication** (plan §T27(c): "an evening"). This converts a
      disagreement rate into an error rate with a taxonomy, and it is the only
      route to the claim as written.
- [ ] **Or soften the artifact and the write-up** to "disagreement rate," state
      that direction is unresolved, and drop the "first published error
      estimate" framing until (c) is done.

Also confirm: the plan asks for **100** stratified disagreements,
`disagreement_sample_n` is **75**. Stratification constraint or truncation?

- [ ] Resolve the 75-vs-100 discrepancy in the artifact's `inputs`

### A3 — `t27_disagreement_sample.csv` is not UTF-8
One non-ASCII byte, `0xf6` at position 233 (cp1252 `ö` in "Alström"). Breaks
any UTF-8 CSV reader, in the one file destined for a human reviewer.

- [ ] Re-emit with `encoding="utf-8"` before anyone opens it for A2

### A4 — The verdict layer needs the review the metrics layer already gets
**Three bugs in three days, all the same shape**, all in the code that turns
correct numbers into a conclusion:

1. `t28a` — SHRINK threshold of 0.30 compared against raw accuracy on a
   sample with a 0.615 base rate. Fires for any model that answers.
2. `t26` — verdict compared SapBERT against **all** arms including
   `presence`, a trivial floor with near-zero transfer penalty by
   construction. Self-caught, see `verdict_correction_note`.
3. `t27` — verdict declared the remap the new campaign default **without
   checking the sign** of the effect; the effect is negative. Self-caught,
   see `verdict_correction_note`.

In all three the underlying measurements were fine. Two of three were caught
within a day, which is the right instinct, but the pattern is systematic
rather than coincidental.

- [ ] Add a test per experiment that feeds the verdict function a null /
      no-effect input and asserts the verdict comes back negative
- [ ] House rule: every `verdict` string names its **baseline** and its
      **direction** explicitly

### A5 — T28a fixes and the five-model run
Unchanged from 2026-08-27 and still gating every dollar of Wave 2 spend
(`wave2-start-plan.md` §5's order of work). F1–F6 are blocking; see
`docs/t28a_fixes_before_full_run.md` for detail and for the
pre-registration amendment that F1/F2 require.

- [ ] F1–F6 land
- [ ] Amendment written and dated **before** the run
- [ ] All five models in one invocation (`--n-trials 200 --seed 42`)

### A6 — Re-read §6.7 in light of T25
The amendment's §6.7 fallback ("if the rung ladder (T23–T25) ties within CI at
the pooled level, PR-6's classical side narrows to T24's ancestor expansion on
the rare-disease half") was written against a state where T25 had not run.

T25 has now produced a pooled result that clears its interval on two clinical
tasks, so **the antecedent no longer holds** and the fallback should not be
invoked. PR-6's classical side is the direct result, not the consolation.

- [ ] Confirm this reading and record it, so the fallback is not applied out
      of habit when Part 3 is drafted

---

## Notes, not actions

### What landed
- **T25, T26, T27** written and run — the Tier 2 block `wave2-start-plan.md`
  §8 had declared out of scope.
- **P12** (`src/splits/disease_holdout.py`) built, with an honest docstring
  about the longest-processing-time bin-packing heuristic not being a hard
  guarantee on the 5–15% fold band, and `fold_fracs` recorded so it is
  visible rather than hidden. Good.
- **SapBERT GPU path fixed** — real `.to(device)` placement with
  `torch.cuda.is_available()`, replacing a docstring that claimed GPU while
  the code was CPU-only with a bare `.numpy()` that would have raised on a
  CUDA tensor (`wave1-preflight-review.md` L1).
- **`requirements-extended.txt`** torch install comment corrected: a bare
  `pip install torch` resolves to the `+cpu` wheel on Windows even with a GPU
  present.

### Results
**T25.** Grouped multi-hot codes beat SapBERT dense vectors on the clinical
tasks by more than the pooled CI. Outcome: mean delta −0.0543, CI [−0.0690,
−0.0399]. Mortality: −0.0342, CI [−0.0521, −0.0166]. Fusion adds nothing —
`d_vs_a` on outcome is +0.0004, CI [−0.0021, +0.0029]. **PR-3 first half CUT.**
Arm (e) `DEFERRED` by user decision, so **PR-7 is unevaluated**.

**T26.** Task-dependent. SapBERT clears both multi-hot arms on
`mortality_rate_yn` (non-overlapping CIs) and on neither for
`serious_adverse_rate_yn`. Zero single-class folds skipped. PR-3's second half
is neither SUPPORTED nor CUT.

**T27.** Shipped-vs-remapped agreement is far lower than the plan's 13%
literature figure implies: full-code Jaccard 0.036–0.073, char3 0.23–0.39,
chapter 0.45–0.60. Re-running scores moves them not at all or in the wrong
direction. Shipped mapping stays the campaign default. Interpretation blocked
on A2.

### Where Wave 1 stands
One clean positive (T24, rare-half ancestors), one clean negative (T25, dense
loses and fusion is dead), one task-dependent (T26), one tie (T23), one
inconclusive (T22), and T27 pending an evening of human work. Enough for the
classical half of Part 3, and it depends on nothing in Wave 2.
