# T28a — fixes before the five-model run

Written 2026-08-27, after the two-model run (`amazon.nova-lite`,
`deepseek.v3-2`, n=200, seed 42) whose artifact is now at
`results/experiments/t28a_probe_gate.json`. Three rungs remain:
`anthropic.claude-opus-4-5`, `anthropic.claude-haiku-4-5`,
`meta.llama4-maverick-17b`.

Everything below is derived from the two-model artifact and from the code,
not from the smoke test. Where a fix requires choosing a number after having
seen results, that is flagged explicitly in §Pre-registration, because it is
the one place this document could quietly damage the campaign's integrity.

---

## What the two-model run actually showed

Both models branched `SHRINK_TO_UNRECOGNIZED_STRATUM`. Both are false
positives of a mis-specified threshold.

| | outcome_recall | pooled majority-class | margin | title_recall |
|---|---|---|---|---|
| nova-lite | 0.600 | 0.615 | **−0.015** | 0.000 |
| deepseek v3.2 | 0.620 | 0.615 | **+0.005** | 0.005 |

`outcome_recall_hit` is raw accuracy against the true label. The pooled
sample is 61.5% positive, so a model answering "Yes" to everything scores
0.615 — more than twice the 0.30 SHRINK threshold, with no memorization
whatever. The gate cannot currently return `PROCEED_AS_DESIGNED` for any
model that answers at all.

Rescored per task on balanced accuracy (base-rate invariant), with Fisher
exact on each 2×2:

| Task | n | nova bal.acc (p) | deepseek bal.acc (p) |
|---|---|---|---|
| mortality_rate_yn | 34 | 0.604 (0.291) | **0.792 (0.002)** |
| patient_dropout_rate_yn | 73 | 0.500 (1.000) | 0.676 (0.013) |
| outcome | 59 | 0.532 (0.493) | 0.516 (1.000) |
| serious_adverse_rate_yn | 34 | 0.524 (1.000) | 0.500 (1.000) |

Pooled across tasks both models look significant (nova p=0.009, deepseek
p<0.001). That pooled signal is Simpson's paradox: four tasks with base
rates from 0.525 to 0.836 pooled into one 2×2. Per task, nova has nothing
anywhere, and deepseek has mortality (survives Bonferroni at 0.05/8 =
0.00625) and dropout (does not).

Title recall is 0/200 and 1/200. By the rule of three that bounds the true
rate below roughly 1.5% at 95%.

**Reading:** no detectable recall on either rung; one real predictive signal
(deepseek/mortality, n=34, on the most information-rich input in the design).
Both rungs are the cheapest two of five, and the campaign predicted recall
scales with capacity, so this does not generalize to Opus.

---

## Blocking fixes

### F1 — the outcome-recall gate cannot separate prediction from recall

This is the substantive defect, not the arithmetic one. `outcome_recall_hit`
asks whether the model got the answer right. A model can get the answer right
by remembering it *or* by predicting it, and the current branch treats both
as contamination. Deepseek/mortality is exactly this case: a significant
result that the gate would read as recall and that the title probe says is
not.

`title_recall_hit` has no predictive route — nothing about a trial's clinical
content lets you infer its registered title from an NCT ID alone. It is the
discriminator, and the branch logic should be built on it:

- title recall significantly above zero → recall demonstrated → `SHRINK` /
  `STRATIFY` per the existing thresholds
- outcome discrimination significant **and** title recall null → predictive
  signal, `PROCEED_AS_DESIGNED`, and record the effect size
- neither → no signal, `PROCEED_AS_DESIGNED`

### F2 — score outcome recall per task, base-rate invariantly

Replace raw pooled accuracy with per-task balanced accuracy plus Fisher exact,
Bonferroni-corrected across (models × tasks). Retain the raw rate as a
descriptive field explicitly labelled base-rate-inflated, so the old number
stays readable against this run's artifact without being actionable.

Per-task is not optional: pooling produced a significant result for a model
(nova) that has no signal on any individual task.

### F3 — `recognition_uninformative` is a silent false negative

```python
recognition_uninformative = (
    best_detector_auroc is not None and blind_auroc is not None
    and abs(best_detector_auroc - blind_auroc) <= BLIND_BASELINE_NOISE_BAND
)
```

When both are `None` this evaluates to `False`, which reads as "the blind
baseline control ran and the detectors are citable." In the current artifact
both models carry `recognition_uninformative: false` and *no control was
computed at all*. Make it tri-state: `None` when either input is missing.

### F4 — the detector AUROC arm is structurally inert on TrialBench

All three detector AUROCs and the blind baseline came back `None` for both
models. This is not a bug and not fixable by re-running. Every TrialBench
trial registered before 2024-02-16; the earliest ladder cutoff is
llama4-maverick at 2024-08. The pre/post-cutoff label is therefore
single-class for all five models, `_safe_auroc` returns `None`, and
`blind_baseline_auroc` returns `None`.

Confirmed offline against the pinned AACT snapshot: at both n=40 and n=200,
registration dates span 2003–2021 and all 200 trials are pre-cutoff for all
five models.

Two options, and the choice is a real one:

- **(a) Make it computable.** Pool TrialBench trials (pre-cutoff) with P14's
  fully post-cutoff slice (105 trials) to get a two-class label. Both sides
  are already built. This also makes the blind baseline run, which is the
  temporal-drift trap the plan built it to catch.
- **(b) Drop it.** Keep only the two recall probes. That cuts the battery
  from 9 calls per trial to 2, roughly a 78% cost reduction.

Do not leave it as-is emitting nulls that read as absent evidence rather than
as an uncomputable statistic.

### F5 — the artifact overwrites

`OUT_PATH` is a single file holding all models under `per_model`, and `main()`
exposes no `--out-path`. Running the remaining three models will clobber the
two already in the tree.

The response cache did **not** come back in the EC2 tarball (`results/cache/`
contains only `clinical_embeddings`, no `bedrock/`), so re-running nova and
deepseek re-bills. Total realized cost for both was **$0.37** against a $5–15
estimate. Re-run all five in one invocation: one artifact, one SHA, provably
identical rows, and the re-bill is noise.

### F6 — commit before running

Current artifact records
`git_sha: 80c90a0bf1581dc2f8eb2eeeb39c0f50c15f4424-dirty`. The standing
artifact rule requires a clean SHA; a dirty tree means the code that produced
these numbers is not identified.

---

## Non-blocking, worth doing in the same pass

### F7 — store the model's raw answer

`per_trial` records only `outcome_recall_hit` (a bool). Reconstructing what
the model actually said required inferring it as `hit XOR label`. Store the
parsed Yes/No and the raw response text so the 2×2 is readable directly from
the artifact.

### F8 — label-stratify the sample

`_pool_trials` stratifies by *task*, not by label (its docstring is explicit:
the largest-remainder technique from `src/data/subset.py` "applied here to
task strata"). The result is per-task n of 34 to 73 with base rates from
0.525 to 0.836 — the imbalance that broke the threshold and the small n that
leaves everything except deepseek/mortality unresolvable.

Label-stratifying within task strata makes raw accuracy interpretable and
raises per-task power at no additional call cost.

### F9 — report the null's bound, not just its point estimate

Title recall of 0/200 should be reported with its interval. Same for any
per-task balanced accuracy that lands at 0.500: at n=34 that is a wide
interval, not a demonstrated absence.

---

## Pre-registration

**F1 and F2 change a pre-registered decision rule after seeing results from
two models.** That is the thing pre-registration exists to prevent, and it
needs to be recorded as an amendment rather than a silent edit.

The defence, which should be written into the amendment verbatim:

1. The defect is derivable **without reference to the results**. The pooled
   sample's 61.5% positive rate is a property of the sample, computable
   before any model is called. A 0.30 threshold on raw accuracy against a
   0.615 base rate is mis-specified on its face.
2. The correction is in the campaign's own prior art.
   `LLM_CONTAMINATION_PLAN.md` §4 already specified prevalence baselines
   ("the comparison is against those, not against 0.5") and shuffled-ID and
   fabricated-ID controls. None of that survived into
   `t28a_contamination_probes.py`. This is restoring a control the campaign
   already wrote down, not inventing a new one.
3. The corrected rule is **not** a numeric threshold chosen to land on one
   side of the observed data. It is a significance test with a stated
   correction. Any bare number chosen now would be fitted to two models'
   results and should be refused.

Record the amendment dated, before the run, with the two-model numbers
attached as the motivating evidence, and note that nova and deepseek were
scored under both the original and corrected rules.

---

## Recommended invocation

After F1–F6 land and the tree is committed:

```
python -m experiments.t28a_contamination_probes \
    --n-trials 200 --seed 42 --data-root data --results-dir results \
    --models amazon.nova-lite deepseek.v3-2 anthropic.claude-haiku-4-5 \
             meta.llama4-maverick-17b anthropic.claude-opus-4-5
```

`--n-trials 200` matches the prose plan's "200 stratified trials" and the
completed run; the script's default of 40 does not, and 40 gives per-task n
in single digits.

Expected cost: under $5 for all five including the nova/deepseek re-bill,
based on $0.37 realized for the two cheapest rungs and the §7 price ladder.
`price_verified: false` on every model, so treat that as an estimate.

---

## What this does not fix

T29's n=105 (54 with ICD codes) is untouched by any of the above and remains
open regardless of how the probe comes out. A clean contamination result does
not retire the fresh-slice requirement, which is unconditional for LLM claims.
