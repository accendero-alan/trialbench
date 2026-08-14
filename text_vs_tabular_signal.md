# Does text carry the same signal as tabular, or something the tabular view can't reach?

_Follow-up to `healthy_volunteers.md`, which found that `ft_transformer`'s
accuracy on `mortality_rate_yn`/Phase1 was dominated by one tabular feature
(`eligibility/healthy_volunteers__te`) that every strong method — including
the text-only `tfidf_logreg` — converges on. The natural next question: is
`tfidf_logreg` just re-deriving that same tabular fact through text, or is it
finding something the structured features can't represent at all? Both,
depending on the cell — investigated locally with real data, no EC2 rerun
needed._
_Date: 2026-07-24_

---

## What was run

```bash
python -m experiments.tfidf_signal_probe --task mortality_rate_yn --phase Phase1
python -m experiments.tfidf_signal_probe --task mortality_rate_yn --phase Phase2
```

New standalone script (`experiments/tfidf_signal_probe.py`, same
non-invasive pattern as the other `experiments/` scripts — real
loader/featurizer, writes nothing to `results/`). For a fitted
`tfidf_logreg`, it prints:

1. The top TF-IDF terms by learned `LogisticRegression` coefficient (most
   positive → pushes toward class 1; most negative → toward class 0).
2. Whether the raw text actually contains the word "healthy," and whether
   `tfidf_logreg`'s predicted score differs systematically by the raw
   `eligibility/healthy_volunteers` column value — a direct test of whether
   the text channel is redundant with that tabular flag.

## Finding 1 — on Phase1, text mostly re-derives the same tabular fact (but more specifically)

`mortality_rate_yn`/Phase1 — where `healthy_volunteers` is the dominant
tabular feature (`healthy_volunteers.md`):

**Top terms toward mortality (class 1):** `tumor`, `solid`, `advanced`,
`therapy`, `tumors`, `metastatic`, `chemotherapy`, `metastases`, `cancer`,
`cell` — oncology language.

**Top terms away from mortality (class 0):** `healthy` (by far the
strongest, −1.208), `screening`, `single dose`, `alcohol`, `bmi`, `blood
pressure`, `body mass index` — classic healthy-volunteer pharmacokinetic
(PK) study boilerplate.

**Direct check:** mean `tfidf_logreg` predicted score by the raw column value
(test set):

| `eligibility/healthy_volunteers` | n | mean predicted score |
|---|---|---|
| Accepts Healthy Volunteers | 217 | **0.177** |
| No | 202 | **0.584** |

Over 3x apart — text tracks the same underlying distinction almost exactly.
181 of 419 test rows' concatenated text literally contains the word
"healthy." This explains the high Spearman correlation (0.78–0.87) between
`tfidf_logreg` and the top tabular/deep methods found in
`healthy_volunteers.md`: on this cell, text isn't adding new information, it's
re-deriving the same fact through richer, more specific vocabulary (not just
*whether* it's a healthy-volunteer study, but recognizably identifying
*advanced/metastatic solid tumor* as the specific high-risk population).

## Finding 2 — on Phase2, text finds signal the tabular view structurally cannot represent

`mortality_rate_yn`/Phase2 has the largest tfidf-vs-deep-tabular gap in the
current results (`tfidf_logreg` 0.853 vs. `ft_transformer` 0.712, `tabnet`
0.549 — a 0.14 PR-AUC gap to the best deep method). Probing why:

**Top terms toward mortality:** `metastatic`, `metastases`, `advanced`,
**`leukemia`**, **`transplant`**, `progression`, `lung`, `chemotherapy`,
**`covid`/`covid 19`**, **`mechanical`/`ventilation`**, `inhibitor`.

**Top terms away from mortality:** `topical`, `medications`, `pain`,
`history`, `abnormal`, `prostatectomy`, `bmi` — low-acuity, outpatient/
elective-procedure language.

**And critically, the `healthy_volunteers` flag barely applies here:** only
85 of 1600 test rows (5.3%) mention "healthy" in the text, vs. 181/419 (43%)
in Phase1. That's exactly why it didn't appear in `ft_transformer`'s top
permutation-importance features for this cell in the earlier comparison
(`enrollment`/`masking_num` dominated instead) — the feature that carried
Phase1 is nearly absent from Phase2's population.

The real driver in Phase2 is **which specific disease or acute condition** a
trial is treating — leukemia, transplant recipients, COVID-19 patients on
mechanical ventilation are plausibly high-mortality populations, versus
topical treatments or elective prostatectomy studies. This is not a richer
restatement of a tabular fact; it's information with **no tabular
counterpart at all**. The 42–43 columns `TabularFeaturizer` produces
(`src/data/features.py`) are entirely administrative/logistical — arm
counts, masking flags, ages, sponsor/allocation type. There is no column
anywhere for the disease being studied. `condition`, `brief_summary`,
`brief_title`, and `keyword` — the fields where "leukemia" or "COVID-19"
actually appear — are declared `TEXT_COLS` and explicitly excluded from the
tabular view (`_is_raw_multimodal`), consumed only by text/multimodal
methods.

So `ft_transformer` and `tabnet` don't underperform on this cell because of
an architecture or training deficiency (per `rerun.md`, TabNet's gap is
partly fixable implementation issues) — on *this* cell they are **structurally
blind** to the dominant signal, regardless of architecture, because it was
never in their input.

## What this means

The two views are not redundant in general — the relationship is
cell-dependent:

- **Where a structured field captures the key distinction** (Phase1's
  binary healthy-volunteer flag), text mostly re-derives it, just with finer
  granularity (identifying the specific high-risk population, not only the
  yes/no fact).
- **Where the key distinction is the disease/condition itself** (Phase2),
  text carries information the tabular featurizer has no way to represent,
  and any tabular-only method — however sophisticated — cannot close that
  gap without access to text.

This is a concrete, evidence-backed case for **Tier D's `fingerprint_fusion`
stub** (or any tabular+text fusion approach): combining views should help
most exactly where this analysis shows they're complementary rather than
redundant, and the current single-view leaderboard has no way to distinguish
"text and tabular agree" cells from "text sees something tabular can't"
cells — both currently just look like "text_logreg wins."

## Caveats

- **Two cells, one task.** Both examples are `mortality_rate_yn`; whether
  the same Phase1-redundant / Phase2-complementary pattern holds for
  `outcome`, `serious_adverse_rate_yn`, `patient_dropout_rate_yn`, or
  `failure_reason` is untested.
- **Coefficient inspection, not causal attribution.** Top TF-IDF terms show
  what the linear model weights most heavily, which is a reasonable but
  indirect proxy for "what the text is encoding" — collinear or
  correlated terms could share credit in ways a single coefficient list
  doesn't fully disentangle.
- **`clinical_embeddings` untested here.** This probe only covers
  `tfidf_logreg`, since its linear coefficients are directly inspectable;
  whether frozen BERT embeddings pick up the same condition-level signal
  (they likely do, being higher-capacity) is a natural follow-up.

## Related artifacts this session

- `healthy_volunteers.md` — the original permutation-importance finding this
  builds on.
- `rerun.md` — real-data validation of the TabNet fix, same investigative
  thread.
- `experiments/tfidf_signal_probe.py` — the script behind this writeup.
- `experiments/multi_method_signal_compare.py`,
  `experiments/ft_transformer_importance.py` — the earlier permutation/
  correlation tooling that surfaced the Phase1-vs-Phase2 contrast.
