# T28a F4 — detector-AUROC arm: pool with P14, or drop it

Open decision, flagged for team discussion 2026-08-28. Not yet resolved;
F1/F2/F3/F5/F6 do not depend on it and are landing separately. See
`docs/t28a_fixes_before_full_run.md` F4 for the original writeup this
summarizes.

## The problem

`experiments/t28a_contamination_probes.py`'s three detector instruments
(n-gram coverage, tabular memorization, guided prompting) and the blind
metadata baseline all score a model's ability to discriminate trials
registered before vs. after that model's training cutoff. On TrialBench
alone this is structurally dead, not a bug fixable by re-running: every
TrialBench trial was registered before 2024-02-16, and the earliest ladder
model's cutoff (llama4-maverick) is 2024-08. The pre/post-cutoff label is
therefore single-class for all five models. `_safe_auroc` and
`blind_baseline_auroc` both correctly return `None` rather than a fabricated
number — confirmed offline against the pinned AACT snapshot at both n=40
and n=200 (registration dates span 2003–2021, all pre-cutoff for all five
models).

Two options.

## Option (a) — pool with P14's 105-trial slice

Pool TrialBench (all pre-cutoff) test rows with P14's fully post-cutoff
fresh slice (n=105, 54 with ICD codes — see `docs/p14_5_n_gate.md`) to get
a genuine two-class label.

- Makes `detector_aurocs` and `blind_baseline_auroc` computable — both
  return real numbers instead of `None`.
- Exercises the blind-baseline temporal-drift check the arm was
  specifically built to catch (per the plan's framing: if the blind
  baseline matches the detectors, the detectors are reading temporal drift,
  not memory).
- Both data sources already exist; no new data collection.
- Costs: more code (a pooling function that keeps the two sources'
  provenance distinguishable in the artifact — P14 rows are a different
  sampling frame from TrialBench test-split rows). Couples T28a to P14's
  slice, so any future change to P14's slice composition changes T28a's
  detector numbers too. Sample composition shifts — part of the 200-trial
  sample becomes P14 rows rather than pure TrialBench test-split rows,
  which is a change to what `_pool_trials` currently guarantees (stratified
  sampling across TrialBench (task, phase) cells only).
- Keeps the full 9-calls-per-trial battery (4 detector instruments + blind
  baseline + 2 recall probes + cross-instrument agreement).

## Option (b) — drop the detector arm

Keep only the two recall probes (`outcome_recall_hit`, `title_recall_hit`)
that F1's corrected branch logic actually depends on.

- Removes `ngram_coverage_score`, `guided_prompting_delta`,
  `tabular_memorization_score`, `blind_baseline_auroc`, `_safe_auroc`,
  `detector_aurocs`, `recognition_uninformative`, and
  `cross_instrument_agreement` from the script.
- `_run_one_model` shrinks to outcome recall + title recall per trial.
- Roughly a 78% cost reduction (2 calls/trial vs. 9).
- Costs: the detector instruments (n-gram coverage, tabular memorization,
  guided prompting) and the blind-baseline temporal-drift control are gone
  entirely, not merely unused — rebuilding them later (e.g. once a dataset
  with a genuine two-class cutoff label is in scope) means writing them
  again from scratch rather than re-enabling dormant code.

## What happens either way

F1's decision rule is built on `title_recall_hit` and per-task outcome
discrimination (F2), neither of which needs the detector arm. Whichever
option is chosen, F1/F2/F3/F5/F6 land as planned; only the presence and
shape of `detector_aurocs` / `blind_baseline_auroc` /
`recognition_uninformative` / `cross_instrument_agreement` in the artifact
depends on this call.
