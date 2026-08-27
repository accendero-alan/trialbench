# P14.4 — fresh-slice schema reconstruction and the featurizer pass

Run 2026-08-27 against `docs/p14_5_n_gate.md`'s primary slice (105 trials,
fully post-cutoff, fresh for all five ladder models). Implementation:
`src/data/aact_slice.py`'s `emit_trialbench_schema` — one row per `nct_id`,
TrialBench's own ~60-column schema, reconstructed from the 17 extracted
AACT tables plus one live external call per distinct condition string for
`icdcode` (see below). `missing_rate_report` compares the result against
TrialBench's own training data, per-column.

## Acceptance check: PASSED

> "A slice row passes both featurizers fitted on TrialBench train without
> raising, with a per-column missing-rate report against the training
> distribution."

`TabularFeaturizer` (fit on all 4 phases of `mortality-event-prediction`
train, 14,368 rows → 41 live features) and `CodeFeaturizer` (`char3`
granularity, 1,624 features) both fit on train and `transform()` the
105-trial slice with **no exception**. Both featurizers are already
defensively coded against a missing source column (`X.get(c)` /
`col not in X.columns` checks throughout `src/data/features.py`), so this
check does not by itself say the reconstruction is *good* — the missing-rate
report below is where the real signal is.

## What's reconstructed, what's approximated, what's missing outright

Of TrialBench's ~60 columns, `TabularFeaturizer` only actually *uses* 41 of
them after its own >50%-train-missing drop and its raw-multimodal exclusion
(`icdcode`, `smiless`, `condition`/MeSH/text columns feed `CodeFeaturizer`
and the raw view instead) — that's the set that matters for T29's frozen
classical arms, and it's evaluated separately from the full 60 below.

**Of the 41 live `TabularFeaturizer` columns, 39 reconstruct cleanly** (delta
between slice and train missing-rate under 5 points — mostly under 1). Two
do not:

| Column | Train missing | Slice missing | Cause |
|---|---|---|---|
| `phase` | 0.0% | 78.1% | **Real, structural shift, not a defect.** TrialBench's train rows are already phase-filtered (that's the task's own split key — `Phase3/train_x.csv`'s `phase` column is the constant `"Phase 3"` for every row). The general AACT population feeding the fresh slice is 63% non-interventional trials with no phase at all (`PHASE` is null for 126,493 of ~200k sampled AACT studies) — a fresh slice that isn't pre-filtered to interventional trials will always look like this on `phase`. **Follow-up, not done here:** filter the slice to trials with a phase value before scoring T29's per-phase cells, the way TrialBench's own split implicitly does. |
| `responsible_party/responsible_party_type` | 0.1% | 100% | **Reconstruction gap.** AACT's `responsible_parties` table was never extracted into `data/external/aact_20260826/` (`src/data/aact.py`'s docstring lists the 17 tables that were). This is a genuinely used, genuinely low-missing feature in training, so its total absence in the slice is the one finding here worth fixing before T29 runs for real — extracting one more AACT table is a small addition to P14.1, not new design work. |

The remaining 19 always-missing or approximate columns
(`intervention_browse/mesh_term`, the 5 `ipd_info_type-*` flags, `smiless`)
are **not** in the 41 live columns — `TabularFeaturizer` already drops or
excludes every one of them (`smiless`/`intervention_browse/mesh_term` are
raw-multimodal by construction; the `ipd_info_type-*` flags are 89.7%
missing in *training itself* and would be dropped by the >50%-missing rule
regardless of what the slice provides). Their 100% slice-missingness is
therefore inert for `TabularFeaturizer`'s actual output, though
`intervention_browse/mesh_term` does feed `CodeFeaturizer`'s `mesh_int`
block directly — that block will read as all-empty on this slice, which is
a real (if lower-stakes) gap for anyone reading MeSH-intervention features
specifically off T29.

**`icdcode` (feeds `CodeFeaturizer`, not `TabularFeaturizer`) — 51%
reconstructed, and the reproduction method matters.** Per P14.4's own
instruction ("reproduce that mapper... with the re-mapped codes as a second
column for sensitivity" is T27's fuller job, out of scope this wave),
`reproduce_icdcode_column` looks up each trial's AACT `conditions.name`
string against the same public NLM Clinical Tables ICD-10-CM lexical search
API TrialBench's own mapping is built on
(`clinicaltables.nlm.nih.gov/api/icd10cm/v3/search`). **This is a best-effort
reproduction of the mapper's *inputs and endpoint*, not a validated match to
TrialBench's exact selection rule** (how many top hits it keeps per
condition, tie-breaking, etc.) — that validation is T27's job and stays out
of scope per the plan's own Tier-2 deferral. 54 of 105 trials got at least
one code; the other 51 had no lexical match returned (short/unusual
condition strings, or conditions the search API doesn't recognize) and are
correctly emitted as `icdcode = NaN`, which `CodeFeaturizer` already handles
as "no ICD terms" rather than raising.

## Reading this against T29

Two fixes before T29 runs for real, both small relative to what's already
built: (1) extract AACT's `responsible_parties` table and wire it in, and
(2) filter the emitted slice to trials with a non-null AACT `phase` before
assigning them to a T29 per-phase cell (or accept a `phase`-blind pooled
comparison, stated as such). Neither changes this pass's headline: **the
slice-emission and featurizer pipeline works end-to-end on real data**,
and the one genuinely concerning gap (`responsible_party`) is a one-table
extraction away from closed, not a design problem.

## Data

`src/data/aact_slice.py` (code, committed). Full per-column report:
`missing_rate_report()`'s output, not checked in (reproducible from the
pinned snapshot in well under a minute, minus the ~30s of rate-limited NLM
API calls for `icdcode`).
