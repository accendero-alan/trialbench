# T28b — reanalysis plan

Written 2026-08-28 against `results/experiments/t28b_opus_recall.json`
(git_sha `1f68a6d7`, clean). Every item below is **re-analysis, not
re-measurement**: with the response cache pulled, all four run at $0.

## Why

T28b returned `OUTCOME_RECALL_DEMONSTRATED`. That verdict is not supported by
the analysis that produced it, and two further defects plus one open
reconciliation sit alongside it. Nothing here requires new model calls.

## Prerequisite — pull the response cache

`deploy/fetch_wave2_results.sh:34` excludes `cache/bedrock/` unless
`FETCH_CACHE` is set. That is why the cache was absent from the last pull.

```
FETCH_CACHE=1 EC2_HOST=… ./deploy/fetch_wave2_results.sh
```

`RESULTS_DIR` defaults to `results_wave2`, but T28b wrote to
`results/experiments/`. Confirm which tree the cache actually landed in
before assuming the default is right.

**Verify the cache is complete before trusting the $0 claim.** Re-run the
scoring once and check the meter: `calls` should be 0 and `cache_hits`
should be 1,616 (1,609 + 7 from the original run). Any nonzero `calls` means
the cache is partial and you are re-billing silently.

---

## R1 — Threshold-free metrics. Run this first; it may dissolve all three findings

`src/eval/pooled_bootstrap.py:45` computes balanced accuracy by thresholding
the verbalized probability at a hard **0.5**:

```python
"balanced_accuracy": lambda y_true, proba:
    balanced_accuracy_score(y_true, (np.asarray(proba) >= 0.5).astype(int))
```

Verbalized probabilities cluster on round numbers and are usually badly
calibrated — that is disqualifier 3 in `LLM_CONTAMINATION_PLAN.md` §1. If
Opus's outputs sit mostly below 0.5, a fixed 0.5 threshold predicts "No"
almost everywhere, and **a single artifact explains all three of T28b's
headline results**:

- **The 0.185 gap to the reference.** Opus 0.634 against TF-IDF 0.819. The
  reference is a fitted logistic model whose probabilities are calibrated to
  the training base rate, so it thresholds sensibly. Opus is not fitted to
  anything and has no reason to.
- **The 84% swap insensitivity.** Predictions that never cross the threshold
  cannot move more than 0.05 across it.
- **The A→B drop itself.** Arm B's prevalence is lower than Arm A's on both
  endpoints (mortality 0.396 → 0.324, SAE 0.676 → 0.476). Balanced accuracy
  is base-rate invariant *for a fixed classifier*; it is **not** invariant to
  how a fixed threshold interacts with a score distribution that has shifted.

`roc_auc_score` and `average_precision_score` are already imported in the
same module, so this needs no new dependency.

**Do:** rescore every arm and both references on **AUROC and PR-AUC**
alongside balanced accuracy. Report all three. Record the threshold used for
the balanced-accuracy figures explicitly in `inputs` — it is currently
implicit.

**Reading:** if Opus's AUROC is flat across A and B while its thresholded
balanced accuracy drops, the primary result is a calibration artifact and
there is no recall finding. If AUROC drops too, the finding survives R1 and
proceeds to R2.

---

## R2 — Paired difference-in-differences. This decides the verdict

`_branch` (line 615) fires on:

```python
opus_drops = primary["lo"] > 0
ref_drops  = reference["lo"] > 0
if opus_drops and not ref_drops: return "OUTCOME_RECALL_DEMONSTRATED"
```

That tests each drop for significance separately. It never tests whether the
two drops **differ from each other**, which is the actual claim. This is the
Gelman–Stern fallacy, and it is the fourth verdict-layer bug in this campaign
after T28a's threshold, T26's floor arm and T27's direction check.

Observed: Δ_opus = 0.0552 (sd 0.0200), Δ_ref = 0.0209 (sd 0.0250),
difference = 0.0343. Significance depends entirely on the correlation
between the two bootstrap delta series, which is not stored:

| ρ | SE of difference | 95% CI | Verdict |
|---|---|---|---|
| 0 (independent) | 0.0321 | [−0.029, 0.097] | **not significant** |
| 1 (perfect) | 0.0050 | [0.024, 0.044] | significant |

Both are scored on the same rows so ρ > 0, but the two predictors behave very
differently (0.634 against 0.819), so ρ near 1 is implausible.

**Do:** compute `D = Δ_opus − Δ_ref` **within each bootstrap resample**,
using the same resamples for both, paired on `nct_id`. Report D's CI and the
observed ρ. Run it on whichever metric survives R1.

**Reading:** the verdict stands only if D's CI excludes zero. Otherwise the
branch is `INCONCLUSIVE`, not `OUTCOME_RECALL_DEMONSTRATED`.

---

## R3 — Disease swap at matched granularity

Opus insensitivity 0.84 against the reference's 0.135, with
`quarantine_ordering: true`. The two predictors have incomparable output
resolution: Opus emits coarse verbalized integers, the reference is
continuous and essentially never moves less than 0.05 by chance. That
asymmetry alone could produce the gap with no memorisation.

**Do:**

1. Report Opus's **distinct-value count, largest tie-block size, and full
   score histogram per arm** — already required by
   `LLM_CONTAMINATION_PLAN.md` §4 and never implemented.
2. Split the insensitivity rate into **exactly unchanged (move == 0)** versus
   **moved but < 0.05**. The first is "gave the same answer"; the second is
   "moved a little." They mean different things and are currently pooled.
3. Discretise the reference to Opus's observed value set, then recompute both
   insensitivity rates.

**Reading:** `quarantine_ordering` stands only if the gap survives matched
granularity.

---

## R4 — Per-endpoint split, and reconcile against T28a

T28b Arm A is 0.634 pooled at n=250 per endpoint. T28a gave Opus 0.829 on
mortality and 0.769 on SAE at n=34 each. Same model, same data source, and
the T28a figures are what justified funding T28.

Two candidate explanations, and they are separable:

- **Small-sample optimism** at n=34 — the leading hypothesis.
- **Elicitation.** T28a used Yes/No; T28b uses a verbalized probability
  thresholded at 0.5. R1 isolates this half.

**Do:** split T28b's Arm A by endpoint, report balanced accuracy with CIs at
n=250, and state explicitly whether T28a's point estimates fall inside those
intervals.

---

## Sequencing, and what re-bills

R1 → R2 → R3 → R4, all from cache, all $0. R1 first because it determines
whether the quantity R2 tests is meaningful.

**What is *not* free:** normalising the phase-string format across arms
(Arm A carries TrialBench title-case `Phase2`, Arms B and C carry AACT
uppercase `PHASE2` plus categories absent from A entirely — `PHASE1/PHASE2`,
`EARLY_PHASE1`, 15.4% of Arm B). That changes prompt text, changes the
prompt hash, misses the cache, and re-bills at roughly $10. Defer it. If R1
or R2 dissolves the verdict, the re-run may not be worth doing at all.

Also unresolved and free to settle: the preflight text-identity check
returned **47/50**, not 50/50, and the spec called a systematic difference
blocking. The three mismatched NCT IDs are recorded in the artifact
(`NCT01248455`, `NCT00640159`, `NCT01844765`). Adjudicate them and write the
decision down rather than leaving the gate silently passed.

---

## Pre-registered readings

| R1 | R2 | Verdict |
|---|---|---|
| Opus AUROC flat A→B | — | **No recall finding.** The primary was a calibration artifact. Report the threshold sensitivity as the result. |
| AUROC drops | D's CI excludes 0 | **Outcome recall demonstrated**, as originally branched. |
| AUROC drops | D's CI includes 0 | **Inconclusive.** Underpowered to separate Opus's drop from the reference's. Report both drops and the difference with its CI. |

Whatever R1 returns, the threshold-free numbers become the reported figures
and the balanced-accuracy ones move to a secondary column with their
threshold stated.

---

## Artifact changes to land at the same time

- **Dump per-trial predictions** alongside the summary: `nct_id`, `arm`,
  `endpoint`, `p14_label`, `opus_score`, `reference_score`, `swap_score`.
  `t28b_opus_recall.py` currently writes only the summary via
  `write_artifact`, which is why this reanalysis needs the cache at all. It
  is the same gap as F7 on T28a and it has now cost twice.
- Record the balanced-accuracy threshold in `inputs`.
- Record ρ and D in `primary_a_vs_b`.
- Note in the artifact that Arm C's prevalence (0.096 / 0.125 against Arm A's
  0.396 / 0.676) leaves roughly 20 positives, so its secondary contrast is
  uninformative by construction rather than null.
