# P14.6 — the label audit (confusion tables, feeding T29's 5% rule)

Run 2026-08-27. Formalizes `docs/aact_label_rule.md` §1–§3's agreement
figures as the confusion table P14.6 asks for, and states the number T29's
pre-registered rule actually gates on.

**Decision rule used (per `docs/aact_label_rule.md` §1–§3):** any qualifying
event anywhere, pooled across arms — `subjects_affected > 0` (pooled) for
mortality/SAE from `reported_event_totals`, `NOT COMPLETED count > 0`
(pooled, `period='Overall Study'`) for dropout from `milestones`. Not a rate
threshold; see that document for why.

## Confusion tables (TrialBench `Y/N` × this rule's computed label)

**Mortality** (n=17,915):

| | Rule=0 | Rule=1 |
|---|---|---|
| **TB=0** | 10,875 | 20 |
| **TB=1** | 3 | 7,017 |

**Serious adverse events** (n=17,915):

| | Rule=0 | Rule=1 |
|---|---|---|
| **TB=0** | 5,932 | 9 |
| **TB=1** | 2 | 11,972 |

**Patient dropout** (n=32,050):

| | Rule=0 | Rule=1 |
|---|---|---|
| **TB=0** | 6,450 | 6 |
| **TB=1** | 8 | 25,586 |

## Against T29's pre-registered 5% rule

`wave2-start-plan.md` T29: *"If the label audit disagrees with TrialBench on
more than 5% of overlap trials, that is a dataset finding that outranks
everything else in this wave and gets reported to the TrialBench
maintainers."*

Disagreement rates: **mortality 0.128%** (23/17,915), **SAE 0.061%**
(11/17,915), **dropout 0.044%** (14/32,050). All three are roughly **40–100x
below** the 5% threshold — the rule does not trigger, by a wide margin.

## What backs this beyond the large-sample count

`docs/aact_label_rule.md` §6 (P14.3) hand-read a stratified 30-trial sample
that deliberately over-included disagreements (12 of the ~48 total across
the three endpoints) against live ClinicalTrials.gov records, independent of
this rule's own code path. Result: 30/30 hand-read labels agreed with the
rule; 0/30 agreed with TrialBench where the two disagreed. 11 of the 12
sampled disagreements read as registry drift (a death/SAE/dropout added to
the record after TrialBench's snapshot, typically in an open-label extension
or long-term follow-up period); one (NCT00038727, DPPOS) is flagged as a
likely TrialBench labeling defect for that specific trial given the size of
the discrepancy (455/3,234 non-completers is not a small correction).

**Conclusion: P14.6 is closed, the 5% gate does not fire, and nothing here
is reported to the TrialBench maintainers** — the small residual
disagreement is explained by drift, not by a flaw in either label source at
scale.
