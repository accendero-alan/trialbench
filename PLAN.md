# Benchmarking SOTA Predictive Methods on TrialBench (Classification Tasks)

A plan for building a reproducible, CPU-only benchmark of state-of-the-art predictive
methods on the classification tasks of **TrialBench**
([ML2Health/ML2ClinicalTrials](https://github.com/ML2Health/ML2ClinicalTrials),
arXiv [2407.00631](https://arxiv.org/abs/2407.00631)), executed with Claude Code.

---

## 1. Objective and scope

TrialBench ships **one** deep baseline — a HINT-style multimodal fusion network
(`MultiModel`) that combines molecule (MPNN/ADMET), disease (GRAM over ICD codes),
protocol/eligibility text, tabular (DANet), free-text, and MeSH encoders, evaluated with a
20-sample bootstrap. This project builds a harness that runs **a variety of additional
methods on the same data, splits, and metrics**, and ranks them against that baseline.

**In scope — the five classification tasks** (each across trial phases I–IV):

| `base_name` (repo) | Task | Type | Label |
|---|---|---|---|
| `outcome` | Trial approval / outcome prediction | Binary | `trial_approval_prediction` |
| `mortality_rate_yn` | All-cause mortality event | Binary | `Y/N` |
| `serious_adverse_rate_yn` | Serious adverse event | Binary | `Y/N` |
| `patient_dropout_rate_yn` | Patient dropout event | Binary | `Y/N` |
| `failure_reason` | Trial failure-reason identification | Multiclass | `trial_failure_reason_prediction` |

That is 4 binary + 1 multiclass task × 4 phases = **up to 20 task×phase settings**
(plus an optional pooled `All` phase). Regression tasks (duration, rates, dose) and the
generation task (eligibility-criteria design) are **out of scope** by request, but the
harness is designed so they can be added later.

**Out of scope (for now):** GPU-only training, fine-tuning large language models, and the
regression/generation tasks.

---

## 2. Understanding the data

Each task×phase is delivered as four CSVs — `train_x.csv`, `train_y.csv`, `test_x.csv`,
`test_y.csv` — indexed by NCT ID. The loader carves a validation split (20%) out of train
with a fixed seed, matching the repo. There are ~58 feature columns spanning several
modalities:

- **Tabular / numeric:** `enrollment`, `number_of_arms`, arm/intervention counts,
  `masking_num`; ages (`eligibility/minimum_age`, `maximum_age`, normalized to months).
- **Categorical:** `phase`, `study_type`, `eligibility/gender`,
  `sponsors/lead_sponsor/agency_class`, `study_design_info/*`, oversight flags, etc.
  (repo uses leave-one-out target encoding).
- **Multi-hot:** `MaskingType-*`, `ipd_info_type-*` (already 0/1).
- **Free text:** `brief_title`, `brief_summary`, `detailed_description`,
  `eligibility/criteria/textblock`, `intervention/description`, `keyword`, `condition`.
- **Molecule:** `smiless` (SMILES list of the intervention drugs).
- **Disease:** `icdcode` (list-of-lists of ICD-10 codes).
- **Ontology embeddings:** `mesh_term` columns → MeSH graph embeddings (`mesh_embeddings.txt.gz`).

**Labels** live in `*_y.csv`: a `Y/N` column (binary) plus the underlying continuous rate;
`outcome` and `failure_reason` use their own named target columns.

**Class imbalance and small folds** are expected (e.g., Phase-I mortality is heavily
negative). This drives two decisions below: **PR-AUC is the headline metric**, and we keep
the repo's **bootstrap** for confidence intervals.

### Data acquisition
Three routes, in order of preference for a CPU/offline setup:

1. **Direct Zenodo download** (default, `src/data/download.py`): fetches just the 5
   classification-task zips from record `15455785` (stdlib only — `urllib` + `zipfile`,
   no extra dependencies). Preferred over the official package below for this benchmark.
2. **`pip install trialbench`** then `trialbench.function.download_all_data('data/')`
   (`--via-package` flag) — the official route, but `trialbench.function` hard-imports
   `torch` even just to call the download function, and `download_all_data` pulls all 8
   TrialBench tasks, including ~250MB of out-of-scope regression/generation data
   (duration, dose, eligibility-criteria-design) this project doesn't use.
3. **Toy samples** already in the repo under `Trialbench/data/**` — enough to develop and
   smoke-test the harness before downloading the full data.

`mesh_embeddings.txt.gz` must be placed in the trialbench package's
`data/mesh-embeddings/` directory (see repo README).

---

## 3. Methods to benchmark

We propose a broad menu across families, then **converge on what is executable on CPU**.
Each method is tiered by feasibility so execution can proceed core-first.

### Tier A — Core (CPU-native, fast, implement first)
These are the strongest standard baselines for structured/tabular data and run comfortably
on CPU. They operate on the **flattened feature matrix** (numeric + encoded categoricals +
multi-hot + optional text/molecule features from §4).

- **Linear:** Logistic Regression (L1 and L2), with class weighting.
- **Bagging:** Random Forest, Extra Trees.
- **Gradient boosting (the SOTA tabular workhorses):** XGBoost, LightGBM, CatBoost
  (CatBoost handles raw categoricals natively), plus scikit-learn `HistGradientBoosting`.
- **Simple references:** k-NN, linear SVM, and a majority/prior baseline for calibration.
- **Text-only baseline:** TF-IDF over concatenated trial text → Logistic Regression /
  linear SVM. Cheap and often surprisingly strong on eligibility text.
- **AutoML ensemble (optional, CPU):** AutoGluon-Tabular as a strong "best-effort" upper
  bound over the tabular view.

### Tier B — Deep tabular (CPU-feasible but slower; implemented)
- **TabPFN:** tabular foundation model; near-instant "training," excellent on small
  samples. Implementation enforces the pretrained model's hard limits (≤10k rows, ≤500
  features, ≤10 classes) via stratified subsampling / `SelectKBest` fit on train only.
  Also needs a one-time free license token (`TABPFN_TOKEN`, see `deploy/README.md`) —
  without one the cell is recorded as "skipped," not an error.
- **TabNet** (`pytorch-tabnet`): small network, early-stopped on validation.
- **FT-Transformer:** hand-rolled directly in plain PyTorch (`src/methods/deep_tabular.py`)
  rather than via `rtdl` (unmaintained) or `pytorch_tabular` (pins `pandas<3.0`, which
  conflicts with this repo's pandas) — a per-column feature tokenizer + CLS token + small
  Transformer encoder + linear head, kept small and early-stopped for CPU speed.
- **DANet** — the tabular encoder the repo already uses — as a like-for-like reference
  (not yet implemented; would need vendoring, like `hint_reference` below).

### Tier C — Text / clinical NLP (precompute once, then cheap)
- **Frozen clinical embeddings:** BioBERT / ClinicalBERT / PubMedBERT / BioClinicalBERT.
  Encode eligibility + protocol text **once** to fixed vectors (CPU inference is slow but a
  one-time cost; cache to disk), then train a light classifier on top.
- **Sentence-transformers** (e.g., a MiniLM or PubMedBERT-based model) as a lighter
  embedding alternative.
- Fine-tuning these transformers end-to-end is **Tier D** (GPU-preferred; flagged optional).

### Tier D — Multimodal / domain-specific and LLM (heaviest; optional on CPU)
- **Fingerprint fusion (CPU):** RDKit Morgan/ECFP fingerprints for `smiless` +
  ICD/MeSH embeddings + tabular → concatenate → gradient boosting. A pragmatic,
  CPU-friendly stand-in for the full multimodal network.
- **HINT-style `MultiModel` (repo baseline):** wrap the repo's own model as a registered
  method for a true apples-to-apples reference. Trainable on CPU per the repo defaults, but
  slow — run on a subset first.
- **LLM-based (API, not CPU-bound):** serialize each trial's structured fields to text and
  do zero-/few-shot classification with a frontier LLM; optionally LLM-generated features.
  Cost- and rate-limited rather than compute-limited; keep to a capped sample.

**CPU reality check:** Tier A is the executable core and will produce a complete
leaderboard on its own. Tiers B–D are added opportunistically; anything that can't finish
in a reasonable wall-clock budget is run on a capped subset or marked "not run (compute)".

---

## 4. Featurization strategy

Two feature *views*, built by a shared featurizer so every method sees identical splits:

1. **Tabular view** (Tier A/B): reproduce the repo's preprocessing so results are
   comparable — normalize ages via `refine_year`, drop columns >50% missing, impute,
   **leave-one-out encode** categoricals (fit on train only), keep multi-hot as-is. Add
   optional blocks: (a) TF-IDF of concatenated text, truncated by SVD; (b) Morgan
   fingerprints from `smiless`; (c) mean MeSH embedding. All transforms **fit on train,
   applied to valid/test** to prevent leakage.
2. **Raw multimodal view** (Tier C/D): pass through the original text / SMILES / ICD /
   MeSH fields for methods that do their own encoding.

Featurization is deterministic (seeded), and expensive artifacts (text embeddings,
fingerprints) are **cached** keyed by task×phase so re-runs are cheap.

---

## 5. Evaluation protocol

**Mirror TrialBench exactly, then extend.** Reuse the repo's fixed train/valid/test splits
and its metric definitions:

- **Binary:** ROC-AUC (**AUROC**), **PR-AUC** (average precision — headline for imbalance),
  F1, precision, recall, accuracy, specificity.
- **Multiclass (`failure_reason`):** macro AUROC (one-vs-rest), macro F1, PR-AUC, accuracy.
- **Confidence intervals:** the repo's **bootstrap** over the test set (default 20 resamples;
  we raise to ~1000 for stable CIs) → report **mean ± 95% CI** per metric.

**Rules that keep it honest:**
- Model selection and all thresholding use **validation only**; test is touched once.
- Fixed seeds (`123`, `42`, `55688` as in the repo) plus a configurable seed list for
  multi-seed runs; report mean ± std across seeds for stochastic methods.
- Encoders/scalers/embeddings **fit on train only**.
- Every run emits a JSON record (method, task, phase, seed, metrics, timings, env hash).

**Outputs:**
- Per-run JSON in `results/runs/`.
- A **leaderboard** (`results/leaderboard.{csv,md}`): methods × tasks with headline metric,
  ranked, and a per-phase breakdown; the HINT baseline highlighted for reference.
- Optional plots (PR curves, per-phase bars) as a follow-on.

---

## 6. Harness architecture

A thin, config-driven pipeline (details in `README.md` / `CLAUDE.md`):

```
config (YAML) → data.loader → data.features → methods.registry[method].fit/predict_proba
              → eval.metrics (+bootstrap) → results/runs/*.json → eval.leaderboard
```

- **`methods/base.py`** — a minimal `BaseMethod` interface: `fit(Xtr, ytr, Xva, yva)` and
  `predict_proba(X)`. Every method (sklearn, GBM, deep, LLM) implements the same contract,
  so adding a method is a ~30-line file plus a registry entry.
- **`methods/registry.py`** — decorator-based registry; the CLI selects methods by name
  from config. Heavy imports are lazy so missing optional deps never break the core run.
- **`data/loader.py`** — reads task×phase CSVs (or the `trialbench` package), aligns x/y on
  NCT ID, makes the valid split.
- **`data/features.py`** — the two feature views of §4, with caching.
- **`eval/metrics.py`** — classification metrics + bootstrap, matching the repo's formulas.
- **`eval/leaderboard.py`** — aggregates `results/runs/*.json` into the leaderboard.
- **`run_benchmark.py`** — CLI: `--tasks --phases --methods --seeds`, iterates the grid,
  is **resumable** (skips completed runs), and logs timing so slow methods are visible.

---

## 7. Claude Code execution workflow

The scaffold is designed for incremental, verifiable execution:

1. **Setup** — create the venv, `pip install -r requirements.txt`; confirm the smoke test
   (synthetic data) passes end-to-end. No download needed yet.
2. **Data** — run `src/data/download.py` (trialbench pkg → Zenodo fallback); validate shapes
   against the toy samples.
3. **Core run** — execute Tier A across all task×phase; produce the first full leaderboard.
   *This is the minimum shippable result.*
4. **Extend** — add Tier B (start with TabPFN), then Tier C (cache embeddings once), then
   Tier D (HINT wrapper + capped LLM), each behind its own config flag and re-run only the
   new cells.
5. **Analyze** — regenerate the leaderboard, add CIs/plots, write up findings.

`CLAUDE.md` documents conventions (how to add a method, where results go, the
"fit-on-train-only" rule) so an agent can pick up any step without re-deriving context.

---

## 8. Risks and mitigations

- **CPU wall-clock.** Deep/transformer/LLM methods are slow. *Mitigation:* tiering,
  subset/capped runs, one-time embedding caches, resumable runner, small nets/few epochs.
- **Class imbalance → misleading accuracy.** *Mitigation:* PR-AUC as headline, class
  weighting / `scale_pos_weight`, bootstrap CIs, report full metric set.
- **Leakage via encoders.** *Mitigation:* every transform fits on train only; enforced in
  the featurizer and reviewed in code.
- **Comparability with the paper.** *Mitigation:* reuse the repo's splits, seeds, encoders,
  and metric formulas; register the HINT model as a method to reproduce its numbers.
- **Optional-dependency fragility.** *Mitigation:* lazy imports + graceful "skipped
  (missing dep)" so the core leaderboard always completes.
- **Multiclass label sparsity (`failure_reason`).** *Mitigation:* macro-averaged metrics,
  report per-class support, consider merging rare classes as a sensitivity check.

---

## 9. Milestones and deliverables

| # | Milestone | Output |
|---|---|---|
| 0 | Scaffold + smoke test (this repo) | Runnable harness, synthetic smoke test green |
| 1 | Data wired in | Full task×phase CSVs loading & validated |
| 2 | Tier A leaderboard | Complete classical/GBM/TF-IDF results + CIs |
| 3 | Tier B added | Deep-tabular (TabPFN first) results |
| 4 | Tier C added | Cached clinical-embedding results |
| 5 | Tier D + write-up | HINT reference + LLM (capped); final report/plots |

**Definition of done for the core benchmark:** a leaderboard covering all five
classification tasks × four phases for every Tier A method, with bootstrap CIs, reproducible
from `python -m src.run_benchmark` on CPU.

---

## 10. Repository map

```
trialbench-classification-benchmark/
├── PLAN.md                     # this document
├── README.md                   # quickstart
├── CLAUDE.md                   # conventions for agentic execution
├── requirements.txt            # core CPU deps
├── requirements-extended.txt   # optional deep/NLP/LLM deps
├── configs/benchmark.yaml      # tasks, phases, methods, seeds
├── src/
│   ├── data/{loader,features,download}.py
│   ├── methods/{base,registry,classical,gbm,deep_tabular,text_nlp,multimodal,llm}.py
│   ├── eval/{metrics,leaderboard}.py
│   └── run_benchmark.py
├── tests/test_smoke.py         # synthetic end-to-end test (no download)
└── results/                    # run JSON + leaderboard (gitignored)
```
