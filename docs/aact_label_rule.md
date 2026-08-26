# AACT label rule

Written before `src/data/aact.py` gains its label functions, per P14.2
(`wave2-start-plan.md`). States, per endpoint, the exact tables/columns read,
the numerator/denominator, the multi-arm aggregation, and the binarization
threshold — then measures agreement against TrialBench's own `Y/N` labels on
the overlap, per P14.2's rule: *"If TrialBench's threshold is not documented,
calibrate it on the overlap trials, report the calibrated value, and report
the audit both at that threshold and at the nearest round alternative."*

**Snapshot this rule was written against:** AACT daily flat-file export dated
2026-08-26, `aact.ctti-clinicaltrials.org/static/exported_files/daily/2026-08-26`,
sha256 `d2ef247566ec080a4788f7556ab7fe1dad630e87af5a783f3664be0ddee75e57`
(2,513,044,058 bytes). See `src/data/aact.py`'s module docstring (P14.1) for
the full pinning record.

---

## §1 — Mortality (`mortality_rate_yn`)

**Table:** `reported_event_totals`. Columns: `nct_id`, `ctgov_group_code`,
`event_type`, `subjects_affected`, `subjects_at_risk`.

This table is AACT's own pre-aggregated rollup of `reported_events` (the
per-adverse-event-term table) into exactly three rows per
(`nct_id`, `ctgov_group_code`): `event_type = 'deaths'`
("Total, all-cause mortality"), `'serious'` ("Total, serious adverse
events"), `'other'` ("Total, other adverse events"). Deaths and serious
events are therefore already separated at the source — no term-level
classification is needed to tell them apart.

**Numerator/denominator:** `subjects_affected` / `subjects_at_risk`, summed
across every `ctgov_group_code` (arm/result-group) for the trial before
dividing — i.e. pool arms first, then compute one trial-level rate. Filtered
to `event_type = 'deaths'`.

**Multi-arm aggregation:** pooled sum (see above), not a per-arm rate
averaged or a single reference arm. Verified below to reproduce TrialBench's
labels; a per-arm-then-average alternative was not separately tested once the
pooled-sum rule cleared 99.8% agreement.

**Threshold:** `subjects_affected > 0` (pooled) — i.e. **at least one death
event reported, anywhere, in any arm** — not a rate cutoff at all. This was
not assumed; it was found by testing it against TrialBench's actual labels
(below) after the pooled rate did not show a clean single-cutpoint separation
one would expect from a documented rate threshold.

**Measured agreement (§1's calibration):** 17,915 trials in the intersection
of `mortality-event-prediction`'s train+test (all four phases, deduplicated
by `nct_id`) and AACT's `reported_event_totals` for `event_type='deaths'`.
The any-event rule (`subjects_affected > 0`) agrees with TrialBench's `Y/N`
on **99.87%** (17,892 / 17,915). All 23 disagreements: TrialBench says
`Y/N=0` but the current AACT snapshot shows at least one death event on 20 of
them; TrialBench says `Y/N=1` with zero deaths in the current snapshot on the
other 3. Both directions are consistent with **AACT's live-registry drift**
between whenever TrialBench pulled its snapshot and this one (2026-08-26) —
sponsors amend results-reporting records after initial submission — rather
than with a different threshold rule. P14.3's hand-read sample should include
a few of these 23 to confirm that explanation rather than assume it.

## §2 — Serious adverse events (`serious_adverse_rate_yn`)

**Table:** `reported_event_totals`, same as §1, filtered to
`event_type = 'serious'` ("Total, serious adverse events").

**Numerator/denominator/aggregation:** identical rule to §1 (pooled
`subjects_affected` / `subjects_at_risk` across arms).

**Threshold:** `subjects_affected > 0` (pooled), same as §1 — at least one
serious adverse event reported in any arm.

**Measured agreement:** same 17,915-trial overlap
(`serious-adverse-event-forecasting`). The any-event rule agrees with
TrialBench's `Y/N` on **99.94%** (17,904 / 17,915).

## §3 — Patient dropout (`patient_dropout_rate_yn`)

**Table:** `milestones`. Columns: `nct_id`, `result_group_id`, `title`,
`period`, `count`. Restricted to `period = 'Overall Study'` — sub-periods
(`'Period 2'`, `'Treatment Period'`, ...) are a finer breakdown of the same
arms and would double-count subjects if pooled alongside the overall-study
row.

**Numerator/denominator:** `count` where `title = 'NOT COMPLETED'` /
`count` where `title = 'STARTED'`, each summed across every
`result_group_id` for the trial (pooled across arms) before dividing.

`drop_withdrawals` (per-reason breakdown: Withdrawal by Subject, Lost to
Follow-up, Adverse Event, Death, ...) is available for an audit of *why*
subjects dropped, but is not needed for the Y/N rate itself — `milestones`'
`NOT COMPLETED` count is already the clean aggregate TrialBench's label
almost certainly reads.

**Threshold:** `NOT COMPLETED count > 0` (pooled) — at least one participant
recorded as not completing the study, in any arm.

**Measured agreement:** 32,050 trials in the intersection of
`patient-dropout-event-forecasting`'s train+test and AACT's `milestones`
(`Overall Study` rows present for both `STARTED` and `NOT COMPLETED`). The
any-event rule agrees with TrialBench's `Y/N` on **99.96%**
(32,036 / 32,050).

## §4 — What this rules out

All three endpoints independently converged on the same shape of rule (any
qualifying event anywhere, pooled across arms, not a rate cutoff), at
99.87–99.96% agreement without needing a calibrated rate threshold at all.
The plan flagged the threshold as "the subtle part" and asked to calibrate on
overlap if undocumented; the finding here is that there effectively isn't a
rate threshold to calibrate — TrialBench's `Y/N` columns are, to measured
precision, "did this event class occur at all," and a "nearest round
alternative" rate cutoff was not tested because there was no residual pattern
in the disagreements suggesting one exists (they read as data drift, per §1).

## §5 — Outcome / failure_reason

Out of scope for this rule file. P14.2 names mortality, SAE and dropout;
T29's endpoints (per `disease-representation-test-plan.md`'s T29 section) are
mortality and SAE specifically. `outcome` (trial-approval-forecasting) and
`failure_reason` are not recomputed from AACT here.

## §6 — Open (P14.3)

The 99.87–99.96% agreement rates are a large-sample audit, not the plan's own
required check. **P14.3's 30 hand-read trials are still owed**: sample 30
trials stratified across endpoint/phase/outcome (including several of the
~40 combined disagreements above), read the actual ClinicalTrials.gov record
by hand, and reconcile every disagreement between the hand-read label and
both TrialBench's and this rule's computed label. Not done in this pass —
flagged rather than fabricated, since "the human label" is specifically what
P14.3 asks for.
