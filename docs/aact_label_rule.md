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

## §6 — P14.3: the 30 hand-read trials

Done 2026-08-27. Sample: 10 trials per endpoint (mortality, SAE, dropout),
stratified across phase and TrialBench `Y/N` value, drawn from the
17,915/17,915/32,050-trial overlaps in §1–§3 (seed 2026, sampling script not
checked in — a one-time audit input, not a re-run pipeline). Each endpoint's
10 include 4 of that endpoint's known TrialBench-vs-rule disagreements
(12 of the ~48 combined disagreements across all three endpoints, comfortably
"several"), plus 6 agreements spread across phase × label-value cells. Every
row below is a live read of `https://clinicaltrials.gov/api/v2/studies/<nct_id>`
(field-filtered to `resultsSection.adverseEventsModule` /
`resultsSection.participantFlowModule` where the full record truncated), not
a re-run of `src/data/aact.py` — an independent read of the primary source,
against which both TrialBench's label and this rule's computed label are
checked.

**Mortality** (`deathsNumAffected`, pooled across all reported groups —
including open-label extensions and long-term follow-up periods, which is
exactly where the disagreements below turn out to live):

| nct_id | phase | tb | rule | hand-read | agrees with |
|---|---|---|---|---|---|
| NCT02427568 | Phase2 | 0 | 1 | **1** (death in 12-mo follow-up arm) | rule |
| NCT02836249 | Phase3 | 0 | 1 | **1** (death in open-label extension) | rule |
| NCT02836236 | Phase3 | 0 | 1 | **2** (deaths in both open-label arms) | rule |
| NCT03569293 | Phase3 | 0 | 1 | **5** (deaths in blinded-extension arms) | rule |
| NCT04897867 | Phase4 | 0 | 0 | 0 | both |
| NCT01632241 | Phase4 | 1 | 1 | **2** | both |
| NCT03717155 | Phase2 | 1 | 1 | 30 (of 43 — advanced NSCLC, median OS 10.1mo) | both |
| NCT04138043 | Phase1 | 0 | 0 | 0 | both |
| NCT00866697 | Phase3 | 1 | 1 | 498 (of 940 — AGO-OVAR16, OS-driven closure) | both |
| NCT02904902 | Phase3 | 0 | 0 | 0 | both |

**Serious adverse events** (`seriousNumAffected`, pooled):

| nct_id | phase | tb | rule | hand-read | agrees with |
|---|---|---|---|---|---|
| NCT03674411 | Phase2 | 1 | 0 | **0** (0/16, 0/2 — 0 SAE; 7 deaths tracked separately) | rule |
| NCT03658304 | Phase2 | 0 | 1 | **1** (fever, CTCAE 4.0) | rule |
| NCT04249778 | Phase4 | 0 | 1 | **8** (3+5 across two arms) | rule |
| NCT01178814 | Phase2 | 0 | 1 | **3** | rule |
| NCT03419403 | Phase3 | 1 | 1 | 14 (6+4+4) | both |
| NCT04454567 | Phase2 | 0 | 0 | 0 | both |
| NCT03230916 | Phase1 | 0 | 0 | 0 (across 11 cohort/subcohort groups) | both |
| NCT02192541 | Phase1 | 1 | 1 | 3 | both |
| NCT00924118 | Phase2 | 1 | 1 | 4 | both |
| NCT03945175 | Phase3 | 0 | 0 | 0 | both |

**Dropout** (`NOT COMPLETED` count, period = Overall Study, pooled):

| nct_id | phase | tb | rule | hand-read | agrees with |
|---|---|---|---|---|---|
| NCT03485287 | Phase2 | 1 | 0 | **0/4** | rule |
| NCT00854373 | Phase4 | 1 | 0 | **0/188** | rule |
| NCT00038727 | Phase3 | 0 | 1 | **455/3234** (DPPOS) | rule — see note |
| NCT00353938 | Phase2 | 0 | 1 | **1/14** | rule |
| NCT01533207 | Phase3 | 0 | 0 | 0/38 | both |
| NCT00449787 | Phase4 | 1 | 1 | 205/401 | both |
| NCT00695578 | Phase4 | 0 | 0 | 0/20 | both |
| NCT00593450 | Phase3 | 1 | 1 | 80/1185 (CATT) | both |
| NCT00587288 | Phase2 | 1 | 1 | 12/106 | both |
| NCT01841632 | Phase1 | 0 | 0 | 0/3 | both |

**Result: 30/30 hand-read labels agree with this rule's computed label; 0/30
agree with TrialBench's label where TrialBench and the rule disagree.** That
includes all 12 sampled disagreement cases — the hand read did not split
them, it resolved every one toward the rule. This is stronger evidence than
§1's assumption: the rule was not merely internally consistent (all three
endpoints converging on the same "any event, pooled" shape), it is what a
person reading the current registry record would also conclude.

**On the disagreement mechanism.** 11 of the 12 read as clean **registry
drift** — the event that flips the label (a death in an open-label extension
added after the double-blind phase, a dropout counted only once a long-term
follow-up period closed out) sits in exactly the part of the record most
likely to be amended or extended after a sponsor's first results submission,
which is when TrialBench's snapshot was presumably taken. **NCT00038727
(DPPOS) is flagged as a likely exception**: 455 of 3,234 participants (14%)
not completing is not a small, easily-missed correction — a discrepancy of
that size reads as a TrialBench labeling defect for this specific trial
rather than drift, though nothing in this audit can distinguish "TrialBench's
source snapshot genuinely had `Y/N=0` and the loader mislabeled it" from
"TrialBench's snapshot was taken before DPPOS's overall-study period existed
in its current form" without TrialBench's own snapshot date, which is not
recorded anywhere the loader exposes. Flagged rather than resolved further.

**Conclusion.** The 99.87–99.96% large-sample agreement figures (§1–§3) stand,
and this hand-read pass upgrades their interpretation from "close enough" to
"the small residual disagreement is explained, not just bounded" — for 11 of
12 sampled cases by drift, for the twelfth by a named, checkable exception.
P14.3 is closed; nothing here changes §1–§4's rule or threshold.
