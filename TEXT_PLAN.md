# TEXT_PLAN.md — benchmarking text-modeling methods on TrialBench classification

An extension to `PLAN.md`. That plan asked "which predictive method wins on
TrialBench's classification tasks?" and answered it with a mostly-tabular menu
plus one text baseline. The text baseline won. This plan asks the follow-up
question properly: **among ways of modeling the trial text, which one actually
predicts best, and what does the winner's margin come from?**

_Status: proposed, not yet executed. Date: 2026-08-11._

---

## 0. Why this is worth doing (the numbers that motivate it)

From the pulled-down EC2 results (`results/extracted/trialbench/results/`,
279 run records across the 20 cells: 269 `ok`, 10 `skipped` for a missing
`tabpfn` license token):

- **`tfidf_logreg` has the best PR-AUC in 17 of 20 task×phase cells**, the other
  three going to `lightgbm`, `xgboost`, and `ft_transformer` (one each) — **but
  that headline is partly a coverage artifact.** Only 9 of the 20 cells ran all
  three deep methods (`ft_transformer`, `tabnet`, `clinical_embeddings`) before
  the Tier B/C pass was cut off (`report.md` §3.1). On those 9 fully-covered
  cells `tfidf_logreg` wins **6 of 9**; on the 11 partially-covered cells it
  wins **11 of 11**. And where `ft_transformer` did run, it is the best tabular
  method in 6 of 10 cells. The incumbent is still the champion, but by a
  narrower and less certain margin than "17 of 20" suggests.
- **It beats the best tabular method by up to +0.141 PR-AUC**
  (`mortality_rate_yn`/Phase2) and loses by up to −0.112 (`outcome`/Phase4). The
  sign flips by cell, which means "text vs tabular" is not one question.
- **The incumbent is a ~40-line method with one hand-picked config**, not a
  library default: bigrams, `min_df=2`, `sublinear_tf=True`,
  `max_features=50_000`, `class_weight="balanced"`, `max_iter=2000`
  (`src/methods/text_nlp.py`). Only `C=1.0` is a default. So the honest question
  is not "does tuning beat the untuned default" but "is one plausible config
  near the ceiling of what lexical modeling can do here" — which nobody has
  checked.
- **The only non-lexical text method tried loses 9 for 9.**
  `clinical_embeddings` (frozen `Bio_ClinicalBERT`, mean-pooled, `MAX_LENGTH=256`)
  averages −0.117 PR-AUC against `tfidf_logreg` (`bert_vs_tfidf.md`) and costs
  ~51 min/cell peaking at 2h10m (`report.md` §3.1) against `tfidf_logreg`'s
  6.5s mean / 16.0s max (recomputed from `runs/*.json`).
- **`text_vs_tabular_signal.md` showed the text channel carries signal the
  tabular view structurally cannot hold** (disease and acuity terms: `leukemia`,
  `transplant`, `covid`, `mechanical ventilation`), because none of the tabular
  columns encodes the condition being studied. (The tabular view is 38–63
  columns depending on the cell — 42–43 on mortality, 56–63 on
  `failure_reason`. `text_vs_tabular_signal.md` reports 42–43 for the mortality
  cells it analyzes, which is correct there but does not generalize.)

So the benchmark's strongest signal source has had exactly **two** modeling
approaches tried on it: one bag-of-words configuration, and one frozen encoder
in a configuration that is measurably handicapped (see §5.2). That is a thin
basis for a leaderboard whose headline finding is "text wins."

Three facts found while writing this plan sharpen the case:

1. **The current text view truncates a lot of its input, unevenly across
   cells.** Median concatenated document length is 531 words on
   `mortality_rate_yn`/Phase1 and 454 on Phase2 (p90 1,245 and 1,060; max
   ~2,700), but 232 on `outcome`/Phase4 and 254 on `failure_reason`/Phase3.
   Across all four mortality phases the medians run 255–531. At roughly 1.3–1.5
   BERT tokens per word, `MAX_LENGTH=256` covers only ~35% of a typical
   mortality-Phase1 document but ~79% of a typical outcome-Phase4 one. So the
   truncation objection to `clinical_embeddings` is strong on the long-document
   cells and weak elsewhere — which is itself a testable prediction (§5.2), not
   a blanket excuse. **These are pre-fix lengths**; once fact 3 below is fixed
   the same coverage falls to roughly 30% and 58%, so truncation gets worse
   everywhere and the spread between cells narrows.
2. **`concat_text` is nondeterministic across processes.**
   `src/data/features.py:TEXT_COLS` is a `set`, and `concat_text` iterates it,
   so field concatenation order varies run to run under Python's per-process
   string-hash randomization (verified: three fresh interpreters, three
   different orders). With `ngram_range=(1,2)`, cross-field boundary bigrams
   change with the order, so **re-running `tfidf_logreg` today will not exactly
   reproduce its own existing leaderboard numbers.** This must be fixed before
   any claim of comparability or any field-level ablation (§7, item 0).
3. **`TEXT_COLS` has a column-name bug, and every text method has been reading
   a truncated field set because of it.** `src/data/features.py` lists
   `brief_summary` and `detailed_description`. The actual columns are
   **`brief_summary/textblock`** and **`detailed_description/textblock`**. The
   names never match, so `concat_text` silently drops them.

   `brief_summary/textblock` is present and **100% populated in all 20
   task/phase folders** (median 43–62 words). `detailed_description/textblock` is
   present in all four `failure_reason` folders. Only
   `eligibility/study_pop/textblock` is genuinely absent, along with
   `intervention/description` and `study_design_info/masking_description` on some
   tasks.

   Effect on the text actually seen, median words per training document:

   | Cell | current | with the fix |
   |---|---|---|
   | `mortality_rate_yn`/Phase1 | 531 | 616 |
   | `mortality_rate_yn`/Phase2 | 454 | 531 |
   | `outcome`/Phase4 | 232 | 316 |
   | `failure_reason`/Phase3 | 254 | **463** |

   So `tfidf_logreg` won 17 of 20 cells while never reading the trial's summary,
   and on `failure_reason` it never read 45% of the available text. This is a
   bigger finding than anything else in this plan, it invalidates nothing on the
   leaderboard (all methods were handicapped equally) but it changes what the
   leaderboard *means*, and it must be fixed before any text comparison is run
   (§7 item 0). `PLAN.md` §2 was right to list these fields;
   `text_vs_tabular_signal.md` was right to name `brief_summary` as a place
   disease terms appear. The featurizer was wrong.

---

## 1. The questions this extension answers

**Q1 — Ranking.** Among text-modeling methods, which achieves the best PR-AUC
per cell, and by how much over a *search-selected* TF-IDF control (§4 T1)?

**Q2 — Attribution of the margin.** Is the text advantage (a) vocabulary
membership, i.e. which disease words appear, (b) document context and word
order, or (c) simply reading more of the document than 256 tokens? Each has a
different implication for what to build next.

**Q3 — Regime dependence.** Does the winner change with training-set size, class
prevalence, phase, or binary vs multiclass? The six diagnostic cells span
n_train **1,276–7,984** and test positive rates **0.27–0.46**. The benchmark's
extremes (10,086 rows on `patient_dropout_rate_yn`/Phase2; prevalence 0.175 on
`mortality_rate_yn`/Phase4 and 0.910 on `patient_dropout_rate_yn`/Phase3) are
outside that span by deliberate exclusion (§3), so regime claims are scoped to
the mid-range and the two extremes stay untested unless a method is promoted to
the full grid at step 6.

A result counts as an answer only if it comes with a **paired** interval on the
delta against the control (§6.2), not two overlapping independent CIs.

---

## 2. Scope: text-only, strictly

**In scope.** Every method receives exactly the fields `concat_text` draws from
(`src/data/features.py:TEXT_COLS` ∩ columns present), and nothing else. Methods
may re-split, re-weight, re-order, or per-field vectorize those fields; they may
not read numeric, categorical, multi-hot, SMILES, ICD, or MeSH columns.

**Explicitly out of scope.**

- **Tabular+text fusion**, including `fingerprint_fusion` and the rank-blend
  from `signal-analysis.md`. Deliberate: adding a fusion axis would confound the
  representation comparison, and the fusion question deserves its own run once
  the best text representation is known. Mixing *two text representations*
  (e.g. TF-IDF ⊕ embeddings) stays in scope, since both sides are text.
- **Every LLM-based method** (`llm_zeroshot`, `llm_fewshot`, `llm_knn_fewshot`,
  `llm_featurizer`). Quarantined to `LLM_CONTAMINATION_PLAN.md`; reasoning in
  §4 T4. This is a harder exclusion than the others: LLM methods are not
  deferred to a later run of this benchmark, they are disqualified from it, and
  only one narrow route exists for LLM-*derived* features to return (see the
  quarantine doc's E5).
- Regression and generation tasks (unchanged from `PLAN.md`).
- The HINT baseline (`hint_reference`) — multimodal by construction.

**Enforcement.** Add a `feature_view = "text"` view to the runner (§7.1) that
hands methods a `DataFrame` containing only the present text columns (methods
that want one string per row call `concat_text` themselves; field-wise methods
use the columns directly) rather than the full raw frame. A text method that
cannot see the tabular columns cannot accidentally use them, which is a
stronger guarantee than a convention.

---

## 3. The diagnostic cell set

Six cells, chosen from the real per-cell results to span the regimes that
matter, not by convenience. `delta` = `tfidf_logreg` − best tabular method;
`lift` = `tfidf_logreg` − `majority`.

| Cell | n_train | n_test | tfidf | best tabular | delta | lift | Why this cell |
|---|---|---|---|---|---|---|---|
| `mortality_rate_yn`/Phase1 | 1,276 | 419 | 0.857 | 0.817 (`extra_trees`) | +0.040 | +0.588 | The **redundant-signal** case. Text re-derives the `healthy_volunteers` flag (`text_vs_tabular_signal.md`). Small n, long documents. |
| `mortality_rate_yn`/Phase2 | 5,212 | 1,600 | 0.853 | 0.712 (`ft_transformer`) | **+0.141** | +0.394 | The **complementary-signal** case and the largest text advantage anywhere. If richer text modeling helps at all, it should show here. |
| `outcome`/Phase2 | 7,984 | 2,514 | 0.598 | 0.606 (`xgboost`) | −0.008 | +0.230 | **Largest cell of the six** and low absolute score (0.598) — real headroom with plenty of training data. Tests whether text ties here for lack of capacity. |
| `outcome`/Phase4 | 2,923 | 892 | 0.621 | 0.733 (`ft_transformer`) | **−0.112** | +0.221 | The **text-loses** case. Does better text modeling close a 0.11 gap, or is this cell just tabular? Short documents (median 232 words), so truncation is not the excuse here. |
| `failure_reason`/Phase3 | 2,672 | 815 | 0.415 | 0.335 (`random_forest`) † | +0.080 | +0.165 | **Multiclass** (macro metrics), in the task family with the benchmark's lowest absolute scores. Exercises the multiclass path of every new method. |
| `serious_adverse_rate_yn`/Phase4 | 1,889 | 584 | 0.698 | 0.696 (`svm_linear`) † | +0.003 | +0.314 | **Smallest n, hardest phase-4 cell.** The small-data regime where contrastive fine-tuning (`setfit`) and the lexical baselines are expected to hold up better than large fine-tunes. |

† **These two "best tabular" values exclude the deep tabular methods**, which
never ran on these cells (12 of 15 methods completed). Since `ft_transformer` is
the best tabular method on `serious_adverse_rate_yn`/Phase1 *and* /Phase2, the
+0.003 on Phase4 could plausibly flip sign once it runs. The six deltas are
therefore not mutually comparable as stated, and closing that gap is a
prerequisite, not an optional extra: **step 1 of §8 re-runs `ft_transformer` and
`tabnet` on these two cells** so every diagnostic cell has a like-for-like
tabular reference. `clinical_embeddings` also has no record on either cell, and
at ~51 min/cell those two runs are budgeted in §9's step-1 backfill row.

Deliberately excluded: `patient_dropout_rate_yn`/Phase3 (`tfidf` 0.965 —
no headroom, everything ties at the ceiling) and `mortality_rate_yn`/Phase4
(`tfidf` 0.504 with delta +0.006 — near-noise, would mostly measure variance).

**Promotion rule.** A method runs the full 20 cells only if, on these six, it
either (a) beats the TF-IDF control on ≥3 cells with a paired-bootstrap delta
whose 95% interval excludes 0, or (b) matches it within 0.01 mean PR-AUC at
≥10x lower cost. Everything else is reported as a six-cell result and stops
there. This is the direct fix for what happened to the Tier B/C pass: one
expensive method with no wins consumed the budget for the rest of the grid.

**This rule gates on test-set performance, which is in tension with golden rule
2** ("do all model selection on validation"). The tension is real and is
accepted deliberately, with two mitigations: the gate decides only *where a
method runs next*, never a hyperparameter or a threshold (those stay on
validation), and every promoted method's 20-cell numbers are reported alongside
the fact that its promotion was earned on 6 of those 20 cells. The alternative —
gating on validation PR-AUC — was considered and rejected because the diagnostic
cells are small enough (419–2,514 test rows) that validation deltas would be
noisier than the test deltas they are meant to protect. Any published ranking
must carry the selection-bias caveat.

---

## 4. Method menu

Cost figures are per cell, order-of-magnitude, extrapolated from the measured
timings in `runs/*.json` (`tfidf_logreg` 6.5s mean; `clinical_embeddings`
51 min mean at `MAX_LENGTH=256`). **The instance type behind those timings is
not recorded anywhere in the repo**, and `deploy/README.md` separately estimates
`clinical_embeddings` at "4–8 minutes per cell" — an order of magnitude below
what the run records show, unreconciled. Every cost number below inherits that
uncertainty; treat them as ratios against `tfidf_logreg` on the same box rather
than as absolute wall-clock, and re-measure on the actual instance at step 2.

### Tier T0 — references (already implemented, re-used as-is)

| Method | Role |
|---|---|
| `majority` | Prior floor; keeps `lift` interpretable. |
| `tfidf_logreg` | The incumbent, **unchanged**, so new numbers stay comparable to the existing leaderboard. |
| `clinical_embeddings` | The known-bad frozen encoder, kept as the "what we already tried" anchor. |

### Tier T1 — sparse lexical, CPU, seconds to a few minutes

The incumbent is one hand-picked lexical config (§0), so the question this tier
answers is not "does tuning beat a default" but **how much lexical headroom is
left above the config already in the repo**. Whatever `tfidf_logreg_tuned` gains
over `tfidf_logreg` is the bar every T2–T3 method must clear to count as an
improvement in *representation* rather than in search effort.

| Method | Sketch | Cost |
|---|---|---|
| `tfidf_logreg_tuned` | **The control.** Grid over `ngram_range` {(1,1),(1,2),(1,3)}, `min_df` {1,2,5}, `sublinear_tf` {T,F}, `max_features` {20k,50k,None}, `C` {0.1,1,10}, and `class_weight` {`"balanced"`, `None`} selected on **validation** PR-AUC only. `class_weight` is in the grid deliberately: the incumbent fixes it to `"balanced"`, and a control that silently dropped it could score *below* the incumbent on these imbalanced cells, making the §5.3 headroom number uninterpretable. Report the chosen config per cell. | ~2–5 min |
| `tfidf_svm` | Linear SVC + `CalibratedClassifierCV` (probabilities needed for PR-AUC). | ~1 min |
| `tfidf_gbm` | TF-IDF → TruncatedSVD(256) → LightGBM. Tests whether non-linearity over lexical features buys anything. | ~2 min |
| `char_tfidf_logreg` | `analyzer="char_wb"`, n-grams 3–5. Robust to the morphology of drug and condition names (`metastatic`/`metastases` share stems). | ~3 min |
| `bm25_logreg` | BM25 term weighting (k1/b tuned on validation) instead of TF-IDF, same linear head. Better length normalization, which matters given a p90/p10 length spread of ~6–7x within a single cell. | ~2 min |
| `nbsvm` | NB log-count ratio features → linear model. The classic strong small-data text baseline; this benchmark's cells are exactly that regime. | ~1 min |
| `fieldwise_tfidf_logreg` | **One vectorizer per text field**, hstacked, so `condition` terms are not diluted by a 2,000-word eligibility block. Directly tests whether `concat_text`'s single blob is throwing away field structure. | ~3 min |
| `lsa_logreg` | TF-IDF → SVD(300) → logreg. Cheap dense-topic reference for the embedding tier. | ~2 min |

### Tier T2 — frozen neural encoders, CPU, minutes

All embeddings cached to `{results_dir}/cache/text_embeddings/<model>/<hash>.npy`
(`results_dir`-relative, per §6.1, not the hardcoded `results/` the existing
`ClinicalEmbeddings.CACHE_DIR` uses)
following the existing `clinical_embeddings` pattern, so re-runs and
overlapping cells pay once. Encoders are pretrained (no fitting), so the
leakage rule applies to the **head** only — but the cache key must still be
split-aware to avoid silently embedding test rows during `fit`.

| Method | Model | Why |
|---|---|---|
| `minilm_logreg` | `all-MiniLM-L6-v2` | Purpose-built sentence embeddings, ~10x cheaper than BERT-base. `bert_vs_tfidf.md` recommendation #2. |
| `bge_small_logreg` | `BAAI/bge-small-en-v1.5` | Current-generation retrieval encoder; stronger than MiniLM at similar cost. |
| `pubmedbert_logreg` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | **Domain fix.** `Bio_ClinicalBERT` was pretrained on MIMIC-III ICU notes; PubMedBERT is closer to the register of clinicaltrials.gov text. |
| `clinical_embeddings_chunked` | `Bio_ClinicalBERT`, 256-token windows, mean+max pooled over chunks | **The truncation fix.** Same model as the 0-for-9 loser, now reading the whole document. Isolates truncation from representation. |
| `e5_base_logreg` | `intfloat/e5-base-v2` | Instruction-prefixed encoder, strong general baseline. Optional if T2 is already saturated. |

Head for every T2 method: `StandardScaler` → `LogisticRegression(class_weight="balanced")`,
identical to `clinical_embeddings`, so the comparison is encoder-to-encoder.
Add `tfidf_plus_minilm` (sparse ⊕ dense, hstacked) as the one intra-text
combination — it tests complementarity between lexical and semantic views
without leaving the text-only scope.

### Tier T3 — fine-tuned encoders, GPU

`bert_vs_tfidf.md` §Recommendation #3 argues this is the only route by which a
transformer plausibly beats TF-IDF here: fine-tuning lets the model learn the
same discriminative keywords TF-IDF weights explicitly, with context on top.
Target instance: one `g5.xlarge` (A10G 24GB) or equivalent; a base-size encoder
on ≤8k rows for 3–5 epochs is minutes, not hours.

| Method | Sketch |
|---|---|
| `bert_finetune_clinical` | `Bio_ClinicalBERT`, 512 tokens, class-weighted loss, early stop on **validation** PR-AUC, 3 seeds. The like-for-like answer to the frozen version. |
| `bert_finetune_pubmed` | Same recipe, PubMedBERT encoder. Isolates pretraining domain under fine-tuning. |
| `deberta_finetune` | `microsoft/deberta-v3-base`. Strong general-purpose encoder; tests whether clinical pretraining matters once fine-tuned. |
| `longdoc_finetune` | `ModernBERT-base` (8k context) or `Clinical-Longformer` (4,096). **No truncation at all** — the p90 document (~1,250 words) fits whole. |
| `lora_finetune` | LoRA adapters on the best T3 backbone. Cheaper and better-behaved on 1.3k-row cells; report against full fine-tuning. |
| `setfit` | Contrastive fine-tune of a MiniLM-class encoder + logistic head. Built for the small-label regime; the natural method for `serious_adverse_rate_yn`/Phase4. CPU-feasible, GPU faster. |

Fine-tuning is stochastic and small-n runs are high-variance, so **T3 methods
run 3 seeds (42, 7, 123) from the start** and report mean ± std across seeds in
addition to the within-cell bootstrap. A single-seed fine-tune number is not
reportable.

### Tier T4 — removed. LLM methods are quarantined.

**No LLM method appears on this leaderboard.** `llm_zeroshot`, `llm_fewshot`,
`llm_knn_fewshot`, and `llm_featurizer` are all excluded, and
`src/methods/llm.py` stays a stub as far as this benchmark is concerned. Three
independent reasons, any one of which would be disqualifying:

1. **Not version-pinnable.** Every other method here is reproducible from
   pinned dependencies plus a seed, give or take a downloaded HF checkpoint and
   the multi-seed variance T3 already reports. A model behind an API endpoint is
   not reproducible in that sense at all, so its
   number cannot be re-derived later and does not belong in a grid whose purpose
   is a stable comparison.
2. **Training-data contamination cannot be ruled out.** These are public
   clinicaltrials.gov records, many with published outcomes, registered years
   before any model's cutoff (across the `outcome` cells, NCT ordinals run from
   172 on Phase3 to 6,161,792 on Phase2, implying ~1999 to ~2024 registrations;
   no single cell spans that whole range). A model may be recalling rather than
   predicting.
3. **PR-AUC penalizes the score format.** Anthropic's API exposes no token
   logprobs, so the score must be verbalized, and verbalized probabilities
   cluster on round numbers. `average_precision_score` lumps ties at one
   threshold, so an LLM would score below its true discriminative ability for
   reasons unrelated to clinical knowledge.

There is also a framing mismatch: this benchmark asks which method learns best
from a given training split, and a model that has read the internet is not
answering that question, so even a clean win would not mean what a leaderboard
column implies.

**The LLM work is not abandoned, it is relocated.** See
`LLM_CONTAMINATION_PLAN.md`, which turns problem 2 from an unfalsifiable
objection into a measured quantity, and defines the one conditional route by
which LLM-derived features could ever re-enter this benchmark.

---

## 5. Ablations that pay off regardless of the ranking

Unlike a leaderboard, these produce findings that stay true when the model menu
changes. §5.1, §5.3, and §5.4 are T1-priced (CPU-minutes). §5.2 is **not** — its
BERT arm is the single most expensive block in this plan (see §9), and is worth
running anyway because it decides whether the existing 0-for-9 result means what
it appears to mean.

### 5.1 Field ablation — where is the text signal?

Run `tfidf_logreg_tuned` on each field subset, all six cells:

`condition` only · `keyword` only · `brief_title` only ·
`eligibility/criteria/textblock` only · **`brief_summary/textblock` only** (newly
reachable after §7 item 0, and the largest single narrative field) ·
`condition` + `keyword` (the "disease names only" hypothesis) · all fields.

`text_vs_tabular_signal.md` suggests the whole text advantage may be
disease-name membership. If `condition`+`keyword` alone recovers most of the
margin, that is a stronger and more surprising result than any ranking in §4 —
it says the benchmark's best method is a disease lookup, and it tells the
tabular side exactly which column it is missing.

### 5.2 Truncation ablation — how much document is needed?

`clinical_embeddings` at `MAX_LENGTH` ∈ {128, 256, 512, chunked-full}, plus
`tfidf_logreg_tuned` on the first-N-words prefix for N ∈ {64, 128, 256, 512, all}
(the prefix arm is CPU-cheap and covers the same hypothesis at 1/1000 the cost,
so run it first). This separates "BERT lost because it is the wrong model" from
"BERT lost because it read a third of the input." One of those is a modeling
conclusion and the other is a bug.

The plan's own length data makes a **falsifiable prediction**: truncation should
matter on the long-document cells (mortality Phase1/Phase2, where 256 tokens is
~35–45% of a median document) and barely at all on the short ones
(`outcome`/Phase4 at median 232 words, ~79% covered). If the prefix curve is
flat everywhere, truncation is exonerated and the frozen-encoder deficit is
about representation. If it rises only on the long cells, the 0-for-9 result was
partly an artifact of `MAX_LENGTH`.

**Blocking implementation note.** `ClinicalEmbeddings._cache_path` keys the
embedding cache on `(MODEL_NAME, sorted NCT ids)` and **nothing else** — not
`MAX_LENGTH`, not the pooling strategy. Any variant that keeps the same backbone
name and cache directory will load the existing 256-token `.npy` and return
byte-identical vectors, producing a perfectly flat ablation with no error and no
warning. Fix the cache key (§7.2) **before** running this ablation, or its result
is guaranteed to be meaningless.

### 5.3 Tuning-headroom control

Report `tfidf_logreg_tuned` − `tfidf_logreg` per cell. This number is the bar
every method in T2–T3 must clear to count as an improvement in representation
rather than in search effort. Since the incumbent is already a deliberate config
(§0), the honest expectation is a *small* headroom; a large one would mean the
existing leaderboard understates lexical methods, and a negative one would mean
the grid or the `class_weight` handling is wrong (§T1) rather than that tuning
hurts.

### 5.4 Train/test near-duplicate audit

Not a method, a validity check, and it gates everything else. Compute
near-duplicate text pairs across the train/test boundary per cell (MinHash or
TF-IDF cosine ≥ 0.9) and report the rate. Trial registries contain
sibling protocols with near-identical eligibility blocks; if those straddle the
split, every text method's score is inflated and the text-beats-tabular finding
is partly an artifact. Run this **first** (~1 hour of work), because a bad
result here changes the interpretation of the entire benchmark.

---

## 6. Protocol

Inherits `PLAN.md` §5 and the golden rules in `CLAUDE.md`, with **two explicit
amendments** flagged up front rather than buried:

- **Scope.** `CLAUDE.md` opens with "on **CPU only**" and `deploy/README.md`
  repeats it. Tier T3 requires a GPU instance. This plan *amends* that scope for
  T3 only; T0–T2 and the ablations stay CPU-only, and T3 is isolated to one step
  with its own instance that is terminated at the end of it (§8 step 5).
- **Golden rule 2.** The §3 promotion rule gates on test-set PR-AUC. See the
  discussion under that rule; hyperparameters, thresholds, early stopping, and
  and early stopping all remain validation-only.

Changes and additions:

### 6.1 Persist predictions and artifacts

`signal-analysis.md` §2 found that saved runs are metrics-only, so answering
"what did the model use?" required a full refit. Fix it here: each run record
additionally writes test-set predicted probabilities (plus the fitted
vectorizer's top coefficients, or the chosen hyperparameters, where cheap) to
`{results_dir}/preds/{task}__{phase}__{method}__seed{seed}.npy`. Path is
**`results_dir`-relative**, not hardcoded to `results/`, so the text grid writes
under `results_text/` and cannot touch the existing results tree. (The same
latent bug exists today in `ClinicalEmbeddings.CACHE_DIR`, a hardcoded relative
`results/cache/...` that ignores `results_dir`; fix it in the same change.) This
is what makes §6.2, agreement analysis, and post-hoc ensembling possible without
re-running a GPU fine-tune.

### 6.2 Paired bootstrap on deltas

Comparing two methods by whether their independent 95% CIs overlap is
underpowered and was already noted as a weakness of the current reporting.
Instead: resample test indices **once** per resample draw and score both
methods on the same draw, giving a distribution of the *delta*. Report
`delta_mean [lo, hi]` against `tfidf_logreg_tuned` as the headline comparison
for every new method. Requires §6.1. Add to `src/eval/metrics.py` as
`paired_bootstrap(y, proba_a, proba_b, ...)`.

### 6.3 Per-method wall-clock budget and hard timeout

New config key `budget_secs` (per method, per cell). On overrun the cell is
recorded as `status: "budget_exceeded"` with elapsed time, and the grid moves
on. This is the mechanical version of the `clinical_embeddings` lesson: no
single method may consume the run.

### 6.4 Seeds

T1/T2 (deterministic or near-deterministic): seed 42, as now. T3 fine-tunes and
`setfit`: seeds 42, 7, 123, mean ± std across seeds reported alongside the
bootstrap.

### 6.5 Cost as a first-class metric

Every leaderboard gains a `secs` column and the writeup includes a
PR-AUC-versus-log-cost scatter. `bert_vs_tfidf.md` established that cost belongs
in the ledger; make it structural rather than a footnote.

### 6.6 Unchanged and non-negotiable

PR-AUC headline, macro-averaged for `failure_reason`. All model selection,
threshold choice, hyperparameter grids, and early stopping use
**train/validation only**. Test is scored once per method per
cell. Vectorizers and heads fit on train only.

---

## 7. Harness work items

Roughly ordered, with the smallest enabling changes first.

0. **Fix `TEXT_COLS`** (`src/data/features.py`), two defects in one edit, and
   nothing else starts until this lands:
   - **Names.** `brief_summary` → `brief_summary/textblock`,
     `detailed_description` → `detailed_description/textblock` (§0.3). Drop
     `eligibility/study_pop/textblock`, which exists in no folder, or leave it
     harmlessly since `concat_text` intersects with present columns.
   - **Determinism.** Replace the `set` with an ordered tuple (§0.2).

   Then re-run `tfidf_logreg` and `clinical_embeddings` on the six cells and
   **report how far the existing numbers move.** Both changes alter the input
   text, so the old leaderboard values are not the new baseline. Expect
   `failure_reason` to move most (its documents nearly double) and treat any
   claim of comparability to the pre-fix leaderboard as void until measured.
   Note the second-order effect: with more text per document, `MAX_LENGTH=256`
   now truncates *more* aggressively, which strengthens the §5.2 hypothesis.
1. **`feature_view = "text"`** in `src/run_benchmark.py:run_cell` and
   `src/methods/base.py`. **Payload: a DataFrame of the present text columns**
   (not a `list[str]`) — field-wise methods need the columns, and concatenating
   methods call `concat_text` themselves. Touch points beyond the runner:
   `src/methods/base.py` docstring, `README.md` (documents the two views),
   `src/data/features.py` module docstring, `CLAUDE.md`'s method-contract
   section, and `experiments/multi_method_signal_compare.py`, which branches on
   `feature_view == "tabular"` and would otherwise silently treat a text method
   as raw. ~60 lines across six files.
2. **`src/data/text_views.py`**: `field_subsets()`, `prefix_words(texts, n)`,
   and a shared `TextCache`. **Cache key must be
   `(model, max_length, pooling, sorted NCT ids)`** — the existing
   `ClinicalEmbeddings._cache_path` omits `max_length` and pooling, which would
   silently invalidate §5.2. Fix that method too, or version its `CACHE_DIR`, so
   old 256-token vectors cannot be served to a 512-token request.
3. **`src/eval/metrics.py`**: `paired_bootstrap`. **`src/run_benchmark.py`**:
   write `{results_dir}/preds/*.npy`; honor `budget_secs`; record
   `max_test_rows_used` and `n_seeds` in the record.
4. **`src/methods/text_sparse.py`** (new): all of Tier T1.
5. **`src/methods/text_frozen.py`** (new): Tier T2, sharing `TextCache`, plus
   the `MAX_LENGTH`/pooling variants §5.2 needs as thin subclasses with distinct
   cache keys. `text_nlp.py` itself stays untouched so the original
   `clinical_embeddings` remains the reference — noting that "byte-identical"
   only holds on a cache hit, since item 0 changes the text it would embed on a
   miss.
6. **`src/methods/text_finetune.py`** (new): Tier T3. Import `torch` lazily
   inside `fit`, **and gate on `torch.cuda.is_available()`, raising
   `ImportError` when there is no GPU** — not `RuntimeError`. Torch *is*
   installed on the existing CPU box (`bootstrap_ec2.sh --extended` installs the
   CPU wheel for `tabnet`/`ft_transformer`), so a bare lazy import would happily
   start an hours-long CPU fine-tune, and any non-`ImportError` exception is
   recorded as `error` rather than `skipped` by the runner's handler.
7. **`src/methods/llm.py`**: **leave as a stub.** Add a one-line docstring
   pointer to `LLM_CONTAMINATION_PLAN.md` so the next reader does not implement
   it into this grid by mistake.
8. **`configs/text_benchmark.yaml`** (new, do not disturb `benchmark.yaml`):
   the six diagnostic cells, the text method list, `budget_secs`, seeds,
   `results_dir: results_text` so the text grid cannot overwrite the existing
   leaderboard. No `llm_*` entries, commented or otherwise.
9. **`requirements-text.txt`**: `sentence-transformers`, `transformers`,
   `datasets`, `accelerate`, `peft`, `setfit`, `rank_bm25`, `datasketch`.
   Grouped by tier with the same "install only what you need" comments as
   `requirements-extended.txt`. **No `anthropic`** — that dependency belongs to
   the quarantined study and its own requirements file.
10. **`tests/test_text_smoke.py`**: every registered text method fits and
    predicts on a 50-row toy slice with the right output shape for both binary
    and multiclass, and no method can see a non-text column. Add an assertion
    that the enabled method list contains no `llm_*` name, so the quarantine is
    enforced by a test rather than by memory.

---

## 8. Execution order and gates

| Step | Work | Gate before proceeding |
|---|---|---|
| **0** | §5.4 near-duplicate audit + harness item 0 (deterministic `concat_text`) | If the cross-split near-dup rate is material (>2–3%), stop and reinterpret the existing benchmark before adding methods. |
| **1** | Harness items 1–3 + smoke tests; backfill `ft_transformer`/`tabnet` on `failure_reason`/Phase3 and `serious_adverse_rate_yn`/Phase4 | `tests/test_smoke.py` and `test_text_smoke.py` both pass; existing `benchmark.yaml` behavior unchanged; all six cells have a full tabular reference (§3 †). |
| **2** | Tier T1 on 6 cells (~8 methods, minutes each, CPU); re-measure `tfidf_logreg` and one `clinical_embeddings` cell to pin down actual per-cell cost on this instance | Control established, §5.3 headroom reported, §4 cost estimates replaced with measurements before any expensive tier is authorized. |
| **3** | §5.1 field ablation + §5.2 prefix arm (CPU-cheap); §5.2 BERT arm only if the prefix curve is non-flat | The ranking-independent findings are banked, and the expensive BERT sweep is conditional on the cheap version showing something. |
| **4** | Tier T2 on 6 cells (CPU, embeddings cached once): cheap encoders first, then BERT-class **only if** a cheap encoder clears the T1 control | Apply the §3 promotion rule. Anything that fails it stops here. The internal gate keeps ~20 CPU-hours of BERT-class work conditional (§9). |
| **5** | Tier T3 on 6 cells × 3 seeds (GPU instance up, then down) | GPU instance is terminated the moment the grid finishes. Promotion rule applies. |
| **6** | Promote survivors to all 20 cells; rebuild `results_text/leaderboard.md` | — |
| **7** | Writeup (§11) | — |

The LLM tier that used to sit at step 6 is gone; it runs on its own track under
`LLM_CONTAMINATION_PLAN.md` with no dependency in either direction, so neither
schedule blocks the other.

Steps 2–4 are CPU-only and can run on the existing EC2 setup with
`deploy/run_forever.sh`; the resume-by-JSON-existence behavior already handles
preemption. Step 5 needs a separate GPU instance and is the only step with a
new deployment surface: extend `deploy/bootstrap_ec2.sh` with a `--gpu` block
(CUDA torch wheel instead of the CPU index) rather than writing a second
bootstrap script.

---

## 9. Budget

All CPU figures are anchored to the one hard datum in the run records:
`clinical_embeddings` = **51 min/cell mean** at 256 tokens, `tfidf_logreg` = 6.5s.
Anything BERT-class is therefore an hours-scale block, and the budget below says
so rather than optimistically rounding it down.

| Block | Hardware | Estimate |
|---|---|---|
| §5.4 audit + harness item 0 | laptop | ~1–2 hours of work, negligible compute |
| Step 1 backfill: `ft_transformer` + `tabnet` + `clinical_embeddings` on the 2 uncovered cells | existing CPU box | ~3–5 CPU-hours: 1–3 for the two deep tabular methods (their measured range elsewhere in the grid) plus ~1.7 for the two `clinical_embeddings` cells at 51 min each |
| Step 0 re-baseline after the `TEXT_COLS` fix: `tfidf_logreg` + `clinical_embeddings` on 6 cells | existing CPU box | ~5 CPU-hours, almost entirely `clinical_embeddings`. Unavoidable: §7 item 0 changes the input text, so the old baselines do not carry over. |
| T1 (8 methods × 6 cells) | existing CPU box | ~2–4 CPU-hours, dominated by the tuning grid |
| §5.1 field ablation (6 subsets × 6 cells = 36 T1-priced runs) | existing CPU box | ~1–2 CPU-hours |
| §5.2 prefix arm (5 lengths × 6 cells = 30 T1-priced runs) | existing CPU box | ~1 CPU-hour |
| §5.2 BERT arm (4 settings × 6 cells = 24 BERT runs) | existing CPU box | **~20–30 CPU-hours.** At 51 min/cell that is ~5 CPU-hours per setting before the chunked variant's ~3x token multiplier. This is the single largest block in the plan, which is why step 3 makes it conditional on the prefix arm. |
| T2, cheap encoders (MiniLM, BGE-small, E5, ⊕TF-IDF: 4 methods × 6 cells) | existing CPU box | ~2–4 CPU-hours; a 6-layer MiniLM is roughly an order of magnitude under BERT-base |
| T2, BERT-class encoders (PubMedBERT, chunked Bio_ClinicalBERT: 2 × 6 cells) | existing CPU box | **~20 CPU-hours** (~5 for PubMedBERT at 256-equivalent, ~15 for chunked full-document at ~3x tokens). Overlaps §5.2's chunked cell — share the cache, run once. |
| T3 (6 methods × 6 cells × 3 seeds = 108 fine-tunes) | 1 × A10G-class | ~6–12 GPU-hours ⇒ roughly \$8–20 at current on-demand rates; the long-context method is ~4x the 512-token ones and is the reason for the upper bound |

No API line: the LLM tier is quarantined (§4 T4), and its budget lives in
`LLM_CONTAMINATION_PLAN.md`.

Rows sum to **~55–75 CPU-hours** and 6–12 GPU-hours, with no API spend. The
first draft of this plan said 15–20 CPU-hours by under-costing the BERT blocks
roughly 6x, the same arithmetic error that let `clinical_embeddings` eat the last
extended pass.

Where the cost actually sits: the three BERT-class blocks (§5.2's sweep, T2's
PubMedBERT and chunked Bio_ClinicalBERT, and the two re-baseline rows) are
**~45–55 of those hours.** `budget_secs` (§6.3) caps any single cell, and §5.2's
sweep is gated on its cheap prefix precursor — but note honestly that **T2's
BERT-class encoders are not gated by anything in §8 step 4 as written.** Either
add a gate there (promote them only if a cheap encoder in the same step shows
non-trivial lift over the T1 control) or accept ~20 unconditional CPU-hours.
Recommended: add the gate. With it, realistic spend if the cheap arms come back
flat is **under 20 CPU-hours**, most of that the unavoidable re-baseline.

---

## 10. Risks and how each is handled

| Risk | Handling |
|---|---|
| **The near-dup audit invalidates the premise.** | It runs first, at step 0, precisely so this is discovered before spending anything. Finding it would be a *result*, not a loss. |
| **Nothing beats the TF-IDF control.** | A legitimate and publishable outcome, and the ablations make it a substantive one: "the signal is disease-name membership, and bag-of-words is the right tool for membership" is a finding. `bert_vs_tfidf.md` already points this way. |
| **Silent cache collisions make an ablation look flat.** | The §5.2 warning and harness item 2: cache keys include `max_length` and pooling, and the truncation ablation includes an assertion that two settings produce *different* embedding arrays before any scores are reported. |
| **Cost estimates are anchored to an unrecorded instance type.** | Step 2 re-measures `tfidf_logreg` and one `clinical_embeddings` cell on the actual box and replaces the §4/§9 estimates before any expensive tier is authorized. |
| **T3 fine-tunes are unstable on 1.3k-row cells.** | 3 seeds mandatory, LoRA variant included, `setfit` included specifically for that regime, and mean ± std reported so instability is visible rather than averaged away. |
| **Cost blowout (the Tier B/C failure mode).** | `budget_secs` per cell (§6.3), the promotion rule (§3), separate `results_dir`, GPU instance terminated at step end, and the two BERT-class blocks gated on cheap precursors. |
| **An LLM method creeps back onto the leaderboard.** | `llm.py` stays a stub with a pointer to the quarantine doc, `text_benchmark.yaml` carries no `llm_*` entries, and `test_text_smoke.py` asserts the enabled list contains none (§7 items 7–10). |
| **Text-only scope leaks.** | The `"text"` feature view makes tabular columns unreachable, and a smoke test asserts it. |
| **Multiclass path breaks late.** | `failure_reason`/Phase3 is in the six from the start, and the smoke test covers both task types for every method. |

---

## 11. What gets written up

- `text-methods-benchmark.md` — the ranking, with paired deltas against the
  control, cost columns, the regime breakdown (Q1, Q3), and the selection-bias
  caveat from the §3 promotion rule.
- `text-signal-attribution.md` — §5.1 and §5.2 (Q2). This is the piece most
  likely to be the interesting one, and it is cheap enough to exist even if the
  rest is descoped.
- `near-duplicate-audit.md` — step 0, however it comes out.
- `text-cols-bug.md` — the `TEXT_COLS` name mismatch (§0.3): what was dropped,
  how far the six cells' baselines move once it is fixed, and what it does to the
  reading of the existing leaderboard. Short, and arguably the most important
  thing to come out of this plan.
- Updates to: `report.md` §5 (next steps), `CLAUDE.md` (the new feature view, the
  new config, the paired-bootstrap convention, the CPU-only amendment), and
  `README.md` (the feature-view list). **No** correction note on
  `text_vs_tabular_signal.md` — an earlier draft of this plan wrongly accused it
  of citing a nonexistent `brief_summary` field; that writeup was right and the
  featurizer was wrong.
- A short erratum in `report.md` §3.1 or the newsletter draft recording that the
  17-of-20 headline is coverage-dependent (6 of 9 on fully-covered cells).
- Separately, on its own track: `llm-contamination.md`, per
  `LLM_CONTAMINATION_PLAN.md`. It is not part of this benchmark's writeup and
  must not be merged into `results_text/leaderboard.md`.

---

## Appendix A — measured facts this plan relies on

Recorded so a future reader can tell which numbers were verified and which were
estimated.

**Verified by recomputation from `results/extracted/trialbench/results/runs/*.json`
(279 records: 269 `ok`, 10 `skipped` — all `tabpfn`, missing license token) and
from `data/*/Phase*/train_x.csv`.** Each number below was computed twice,
independently, by two passes over the raw files.

- Winner counts by PR-AUC across the 20 cells: `tfidf_logreg` 17, `lightgbm` 1,
  `xgboost` 1, `ft_transformer` 1. Identical whether ranked on
  `bootstrap['prauc']['mean']` or `point['prauc']`.
- **Coverage caveat on that count:** only 9 of 20 cells ran all three deep
  methods. `tfidf_logreg` wins 6 of those 9 and 11 of the 11 partially-covered
  cells. `ft_transformer` is the best tabular method in 6 of the 10 cells where
  it ran.
- Per-cell `n_train` / `n_test` / PR-AUC / best-tabular-method / deltas: every
  value in the §3 table, from the same run records.
- `tfidf_logreg` fit time across 20 cells: mean 6.52s, median 5.08s, max 16.01s.
- `clinical_embeddings` fit time across its 9 cells: mean 3,071s (51.2 min),
  max 7,781s (2h09.7m) — matching `report.md` §3.1's "~51 minutes, peaking at
  2h10m". Note `deploy/README.md` estimates 4–8 min/cell for the same method;
  that discrepancy is unresolved and the instance type is unrecorded.
- `clinical_embeddings` 0 wins in 9 comparable cells, mean delta −0.1169
  (`bert_vs_tfidf.md`, confirmed independently from the run records).
- Tabular feature count by task: `outcome` 38–39, dropout 39–40, mortality and
  serious-adverse 42–43, `failure_reason` 56–63.
- Text columns `concat_text` currently *matches*: 7 on mortality / dropout /
  serious-adverse, 5 on outcome and failure_reason. Columns it should match but
  does not, because `TEXT_COLS` uses un-suffixed names (§0.3):
  `brief_summary/textblock`, present and 100% non-null in **all 20** folders,
  median 43–62 words; `detailed_description/textblock`, present in all four
  `failure_reason` folders. `eligibility/study_pop/textblock` is genuinely absent
  everywhere. Headers are identical across phases within a task.
- Document length with those fields restored, median training words:
  mortality Phase1 531 → 616, mortality Phase2 454 → 531, outcome Phase4
  232 → 316, failure_reason Phase3 254 → 463.
- Concatenated document length, train split, words: mortality Phase1 median 531
  (p90 1,245, max 2,672); mortality Phase2 median 454 (p90 1,060, max 2,726);
  mortality Phase3/Phase4 medians 380 / 255; outcome Phase4 median 232 (p90 553);
  failure_reason Phase3 median 254 (p90 688).
- `concat_text` field order varies across processes (three fresh interpreters,
  three distinct orders), because `TEXT_COLS` is a `set`.
- Code facts: `base.py`'s `fit`/`predict_proba` signatures; `run_cell`'s
  `feature_view == "tabular"` else-raw branch; `ClinicalEmbeddings.MAX_LENGTH = 256`
  and its `(MODEL_NAME, sorted index)` cache key; no paired-bootstrap function
  anywhere in `src/`; run records contain no predictions, coefficients, or chosen
  hyperparameters; `except (NotImplementedError, ImportError)` → `status: skipped`;
  `results_dir` / `max_test_rows` / `bootstrap.n_resamples` / `seeds` all exist
  under those names in `configs/benchmark.yaml`; `llm_fewshot`,
  `fingerprint_fusion`, and `hint_reference` all raise `NotImplementedError`.

**Estimated, not measured:** every per-cell cost in §4 and §9, the
words-to-tokens ratio (~1.3–1.5), the ~3x chunking multiplier, GPU-hour and
dollar figures. Step 2 replaces the CPU estimates with measurements; no
expensive tier is authorized on an estimate alone.

**Data supporting the quarantine decision (§4 T4):** `outcome` NCT IDs span
NCT00000309 to NCT06161792 on Phase2 and NCT00000172 to NCT05993143 on Phase3,
(train splits; test splits are narrower)
i.e. roughly 1999 to 2024 registration dates, all well before any current
model's training cutoff. Full contamination analysis in
`LLM_CONTAMINATION_PLAN.md`.
