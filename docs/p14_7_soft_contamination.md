# P14.7 — soft contamination: earliest public mention

Run 2026-08-27, scope as decided: PubMed literature only (NCBI E-utilities,
free, no API key — press releases and news coverage have no comparable free
structured API and are not checked here), against the P14.5 primary slice
(105 trials, fully post-cutoff for all five ladder models, cutoff
2025-07-31).

**Method.** Per trial: `ESearch` against PubMed for
`"<nct_id>"[Secondary Source ID]` (the field PubMed indexes a paper's
associated trial-registry number in), `ESummary` on every returned PMID for
`sortpubdate`, take the minimum. A trial is flagged if its earliest PubMed
mention predates the cutoff in question — the same shape of check
`docs/aact_label_rule.md`/T28a's contamination work uses elsewhere, applied
to literature instead of model behavior.

## Result: 0/105 flagged (0.00% drop rate)

- 0 lookup failures.
- 8 of 105 trials have any PubMed hit at all (1 each); all 8 are dated
  2026-03 through 2026-10, comfortably after the 2025-07 cutoff. (One date,
  2026-10-01, is nominally after "today" in this repo's clock — an
  ahead-of-print / epub-ahead-of-issue cover date, a known PubMed quirk, not
  a data error worth chasing further here.)
- The other 97 have zero PubMed presence under this NCT ID at all.

## Why this result is closer to a structural guarantee than a discovery

**For this specific slice, the check could not have come back any other
way, and that is worth stating plainly rather than presenting 0.00% as a
surprising clean bill of health.** Slice B is defined as
`study_first_posted_date > cutoff` — every trial in it was *registered*
after the cutoff. A PubMed paper citing an NCT ID as a secondary source ID
cannot predate that trial's own registration (nothing to cite before the
registry entry exists), so it categorically cannot predate the cutoff
either, since registration itself is already after the cutoff for every row
in this slice. The 0.00% is real and correctly measured, but it is a
confirmation of the slice's construction, not independent evidence that
these 105 trials are free of soft contamination in the sense CT Open's
methodology cares about (a model's training data containing outcome
information from *some* source before its cutoff).

**Where this check would actually be informative — not done here, flagged
as the natural extension:** `docs/p14_5_n_gate.md`'s **slice (a)**
("registered pre-cutoff, results posted post-cutoff," 6,644 trials at the
latest cutoff) is where soft contamination is structurally *possible* — a
trial registered before a model's cutoff, whose AACT results posting came
later, could still have had its outcome leak earlier through a conference
abstract, preprint, or news report the AACT `results_first_posted_date`
field doesn't capture. That check needs either a much larger PubMed sweep
(thousands of trials, ~35 minutes of rate-limited API calls at this
script's pace — mechanical, not a design change) or a stratified sample of
it, and, for full CT Open-grade rigor, a press-release/news search this
free API doesn't cover. Out of scope for this pass by explicit choice, not
an oversight.

## Data

`data/external/aact_20260826/` (nct_id list) + live NCBI E-utilities calls,
2026-08-27. Raw output not checked into the repo — reproducible in about two
minutes at PubMed's public rate limit (3 req/s, no key).
