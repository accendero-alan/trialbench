# TrialBench classification benchmark — problem statement & findings

_Status report. Covers the state of the current results (from
`results/results.tar.gz`), the harness code, and an investigation into the
TabNet vs. FT-Transformer performance gap._
_Date: 2026-07-24_

---

## 1. What this project is

A CPU-only harness that benchmarks a menu of predictive methods on the five
**classification** tasks of TrialBench (trial approval/outcome, mortality,
serious adverse event, patient dropout — all binary — plus multiclass failure
reason), each across trial phases I–IV. Methods are tiered: Tier A =
classical/linear/GBM/TF-IDF (CPU-native), Tier B = deep tabular (TabPFN, TabNet,
FT-Transformer), Tier C = frozen clinical-BERT embeddings, Tier D = multimodal/
LLM (stubs). PR-AUC is the headline metric because the tasks are imbalanced; every
cell reports bootstrap confidence intervals. Results are one JSON per
task×phase×method×seed cell, aggregated into `leaderboard.{md,csv}`.

The harness itself is well-engineered: leakage-safe featurization (every encoder
fit on train only), consistent class-imbalance handling, resumable runner with
atomic writes, and graceful skip-on-missing-dependency. The problems below are
about the **completeness and interpretation of the current results**, not the
core design.

---

## 2. The problem

**The current results are not yet a valid cross-method benchmark, and the
leaderboard's headline conclusion is an artifact of incomplete data.**

Two things combine to produce a misleading result:

1. The deep-learning methods only ran on part of the grid — the run was cut off
   partway through.
2. The leaderboard averages each method over only the cells it happens to have
   completed, so a method that skipped the hard tasks looks artificially strong.

Concretely, the leaderboard ranks **ft_transformer #1 overall**, but that ranking
is computed over a favorable subset of cells and does not survive a like-for-like
comparison. The defensible result today is that **tfidf_logreg is the strongest
method**, and it is the only strong method with complete coverage.

---

## 3. Findings

### 3.1 Results are incomplete for the deep methods

The full grid is 5 tasks × 4 phases × 16 methods × 1 seed = 320 cells. On disk:
**279 run records — 269 `ok`, 10 `skipped`, 0 `error`, 0 `no_data`.**

Coverage by method (out of 20 task×phase cells):

| Methods | Coverage |
|---|---|
| All 12 Tier A (majority, logreg_l1/l2, random_forest, extra_trees, hist_gbm, knn, svm_linear, xgboost, lightgbm, catboost, tfidf_logreg) | **20/20 — complete** |
| ft_transformer, tabnet | 10/20 |
| clinical_embeddings | 9/20 |
| tabpfn | 0 usable (all 10 present are `skipped`) |

The deep methods (ft_transformer, tabnet, clinical_embeddings) ran on **outcome**
and **mortality_rate_yn** (all four phases) and on **serious_adverse_rate_yn**
(Phase 1–2 only). They are **entirely absent** from **patient_dropout_rate_yn**
and **failure_reason** — no record of any kind.

That absence is diagnostic. The runner writes a JSON record for *every* cell it
reaches (`ok`, `skipped`, `no_data`, or `error`). Cells with no file at all were
never reached, so the Tier B/C pass was **interrupted partway through
`serious_adverse_rate_yn`** and never got to the last two tasks. The most likely
cause is wall-clock cost: `clinical_embeddings` averaged **~51 minutes per cell,
peaking at 2h10m** (Bio_ClinicalBERT inference on CPU), versus ~1–2 s for the
Tier A models. This snapshot is therefore either a mid-run capture or a run that
was killed and not resumed.

`tabpfn` is a clean, expected skip: no `TABPFN_TOKEN` license configured, so every
cell it reached recorded `status: "skipped"` with a helpful reason.

**The `.benchmark_complete` marker is stale.** `deploy/run_forever.sh` only touches
it on a clean full-grid exit, but its own comments note the config grows over time
(Tier A → +B → +C) and that nothing gates on the marker. It was almost certainly
written when the earlier Tier-A-only grid finished; it does **not** mean the current
16-method grid completed. Measure completeness by counting run files, not by the
marker.

### 3.2 The leaderboard's "Overall" ranking is not apples-to-apples

`eval/leaderboard.py` builds per-task and overall means with `pivot.mean(axis=1)`
and `groupby(...).mean()`. **pandas skips NaN by default**, so each method is
averaged over only the cells it actually ran. Combined with §3.1 this invalidates
the cross-method ranking:

- ft_transformer is ranked #1 overall (0.7297) — but only over its 10 completed
  cells, which **exclude `failure_reason` entirely** (the hardest task, mean
  PR-AUC ≈ 0.30 for everyone) and all of `patient_dropout`.
- In the per-task table, ft_transformer's `serious_adverse` "mean" (0.8947) is the
  average of just Phase 1–2; the Phase 3–4 `nan`s are silently dropped.

When **every method is scored on the same 10 cells ft_transformer actually ran**,
the order flips:

| Fair (same 10 cells) | PR-AUC | | As-published "Overall" | PR-AUC | cells |
|---|---|---|---|---|---|
| **tfidf_logreg** | **0.760** | | ft_transformer | 0.730 | 10 |
| ft_transformer | 0.730 | | tfidf_logreg | 0.711 | 20 |
| xgboost | 0.687 | | xgboost | 0.646 | 20 |
| lightgbm | 0.675 | | logreg_l2 | 0.640 | 20 |

So the current leaderboard inverts the top two. The honest headline is
**tfidf_logreg**, complete across all 20 cells.

**Fix:** rank only over cells shared by the compared methods, and flag
partial-coverage methods explicitly rather than mixing them into one ranking.

### 3.3 Threshold-dependent metrics are degenerate for ~19% of binary cells

`eval/metrics.py` computes F1 / precision / recall / specificity / accuracy at a
**fixed 0.5 threshold**. On these imbalanced tasks the models often never emit a
score ≥ 0.5, so they predict all-negative: **43 of 221 binary cells (19%) report
F1 = precision = recall = 0, specificity = 1.0** (e.g. `mortality/Phase1/catboost`:
AUROC 0.732, PR-AUC 0.455, but F1/recall = 0).

This contradicts the stated protocol (PLAN §5 and CLAUDE.md golden rule #2: "all
thresholding uses validation only") — no validation-based threshold selection is
implemented. The **headline ranking is unaffected** (PR-AUC and AUROC are
threshold-free), but the F1/precision/recall columns are currently not meaningful
and should not be reported as-is. Fix by tuning the threshold on validation, or by
dropping the threshold-dependent columns.

### 3.4 What is solid (keep it)

- **Leakage safety is real.** `TabularFeaturizer` fits medians, target-mean
  encodings and one-hot levels on train only; the TF-IDF vectorizer and the
  clinical-embedding scaler/classifier are all train-only. No test leakage found.
- **Imbalance handled consistently** across Tier A (`class_weight="balanced"` /
  `scale_pos_weight` / `auto_class_weights`), PR-AUC headline, 1000-resample
  bootstrap CIs.
- **Production-grade orchestration:** resumable (skip-if-exists), atomic
  `os.replace` writes, per-cell try/except that records skips/errors instead of
  crashing, a supervisor that refuses to mark "complete" when data is missing, and
  a systemd unit for reboot survival.

### 3.5 Smaller issues

- **Binary label encoding is fragile:** `loader.py` uses
  `pd.to_numeric(y).fillna(0)`; literal "Y"/"N" strings would silently all become
  0. The data here is numeric (results have signal), but this should validate
  rather than coerce.
- **Validation split uses the run seed, not a fixed 42** — differs from the
  CLAUDE.md description; no impact on the single-seed run.
- **Bootstrap stores a metric subset** (auroc/prauc/f1/accuracy) while `point`
  stores all seven — minor JSON inconsistency.
- **`catboost_info/` is committed** to the repo root (CatBoost's default log dir);
  add to `.gitignore`.
- **High-cardinality categoricals are silently dropped** in the multiclass
  one-hot path (cap = 40 levels) without logging.

---

## 4. Investigation: why TabNet ≪ FT-Transformer

Both are Tier B deep-tabular methods on the same feature matrix, yet across the
10 cells they share, **ft_transformer averages 0.730 PR-AUC vs. tabnet 0.512** —
and tabnet sits only ~0.08 above the majority-class prior (within 0.05 of it in
4 of 10 cells). TabNet is barely learning; it collapses toward the majority class.

Reading `src/methods/deep_tabular.py`, the gap is an **implementation asymmetry**,
not an architectural verdict:

1. **Class weighting (missing in TabNet).** ft_transformer builds inverse-frequency
   class weights and passes them to its loss. TabNet gets none — pytorch-tabnet's
   `fit()` takes a `weights=1` flag for auto-balancing, but it isn't set. So TabNet
   minimizes plain cross-entropy on skewed data and leans on the majority class.
2. **Feature scaling (missing for both, but TabNet suffers more).** The featurizer
   emits target-encoded categoricals (~0–1) alongside raw numerics like
   `enrollment` (hundreds–thousands). ft_transformer's per-column learned
   tokenizer + LayerNorm + AdamW absorbs the scale mismatch; TabNet's
   attention + batch-norm architecture is notoriously sensitive to unnormalized,
   mixed-scale inputs and destabilizes.
3. **Early-stopping target.** ft_transformer early-stops on the class-weighted
   validation *loss*; TabNet uses pytorch-tabnet's default (accuracy-ish) metric,
   which rewards majority-class collapse on imbalanced data.

Proposed fix for TabNet: add `weights=1`, standardize features (fit on train), and
set `eval_metric=["logloss"]`.

### 4.1 Mechanism check (sandbox limitation + directional result)

The literal test ("fixed TabNet on a real subset") could not be run in this
sandbox: the real TrialBench data is unreachable (Zenodo blocked by the proxy) and
torch/pytorch-tabnet would not install (PyTorch's index is blocked; the PyPI CUDA
build stalls). Instead the mechanism was tested with a neural tabular classifier
that *is* available (sklearn's MLP) on **synthetic data matched to the
mortality/Phase1 profile** (~1,700 rows, 43 features, 27% positive prevalence,
and the same mixed feature scales). The MLP shares TabNet's two failure modes (no
built-in class weighting; sensitivity to unscaled inputs), so it isolates whether
the fixes help.

Result (5 seeds each, identical splits):

| Variant | Test PR-AUC | AUROC | frac predicted-positive |
|---|---|---|---|
| Majority / prior | 0.270 | — | — |
| Repo-like (unscaled, imbalanced) | 0.510 ± 0.07 | 0.660 | 0.13 (collapsing) |
| **Fixed (scaled + rebalanced)** | **0.837 ± 0.02** | **0.923** | 0.28 (≈ true rate) |

Ablations: **scaling alone → 0.811**, rebalancing alone → 0.596. So feature
scaling is the dominant lever for this classifier, with rebalancing adding on top —
a **+0.33 PR-AUC** total improvement, larger than the 0.22 real-data gap between
tabnet and ft_transformer. This is consistent with the gap being an implementation
artifact, not an architectural limit.

**Caveats:** synthetic data and an MLP stand-in, not real TabNet on TrialBench —
treat as directional evidence for the mechanism, not a measurement. (I had earlier
called class weighting "the big one"; the ablation says scaling matters more for a
generic MLP. Real TabNet is even more scale-sensitive *and* ships no class
weighting, so both apply.)

**Artifacts produced:**
- `experiments/tabnet_fix_compare.py` — runs original vs. fixed TabNet on any real
  task/phase subset using the repo's own loader + featurizer; touches nothing in
  `results/`. Run on the EC2 box:
  `python -m experiments.tabnet_fix_compare --task mortality_rate_yn --phase Phase1 --seeds 42 7 123`
- `mlp_mechanism_check.py` (in the session outputs) — the synthetic check above.

---

## 5. Recommended next steps (priority order)

1. **Confirm the EC2 job's state** (`systemctl status trialbench-benchmark`,
   `logs/supervisor.log`). If it died, resume it; if running, deep cells will keep
   filling in.
2. **Decide the CPU budget for `clinical_embeddings`** (~1–2 h/cell is why the grid
   can't finish): cap rows, shorten `MAX_LENGTH`, precompute embeddings up front,
   or mark it "partial (compute)" and exclude from the headline.
3. **Fix `leaderboard.py`** to rank only on shared cells and label partial-coverage
   methods — this alone changes the headline.
4. **Add validation-based thresholding** (or drop threshold-dependent columns).
5. **Add a completeness check** to the leaderboard/README (cells present vs.
   expected per method), so "done" is measured from run files, not the marker.
6. **Validate the TabNet fix on real data** via `experiments/tabnet_fix_compare.py`
   on EC2; if it holds, apply the three changes to `src/methods/deep_tabular.py`
   and re-run just that method.

**Bottom line for reporting today:** quote only the Tier A results and lead with
**tfidf_logreg**; hold the deep-method comparison until the grid is complete and
the aggregation is fixed.
