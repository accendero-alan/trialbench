# Review — TrialBench classification benchmark

_Review of the project scaffold and the results in `results/results.tar.gz`._
_Date: 2026-07-24_

## Verdict

The harness itself is well-built: leakage-safe featurization, sensible imbalance
handling, a clean method contract, and genuinely robust EC2 orchestration
(resumable, atomic writes, graceful skips). **But the current results are not yet
a valid cross-method benchmark.** The Tier A (classical/GBM/text) grid is
complete and trustworthy; the deep-learning methods only ran on part of the grid,
and the leaderboard's aggregation silently rewards that partial coverage — so its
headline conclusion (ft_transformer is the best method) is an artifact, not a
finding.

Two things to fix before quoting any numbers: **finish (or re-run) the deep
methods across all tasks**, and **fix the leaderboard to compare methods only on
shared cells**. Details below.

---

## 1. Results completeness — the deep methods are missing ~half the grid

The grid is 5 tasks × 4 phases × 16 methods × 1 seed = 320 cells. On disk there
are **279 run records: 269 `ok`, 10 `skipped`, 0 `error`, 0 `no_data`.**

Coverage per method (out of 20 task×phase cells):

| Methods | Coverage |
|---|---|
| All 12 Tier A (majority, logreg_l1/l2, RF, extra_trees, hist_gbm, knn, svm_linear, xgboost, lightgbm, catboost, tfidf_logreg) | **20/20 — complete** |
| ft_transformer, tabnet | 10/20 |
| clinical_embeddings | 9/20 |
| tabpfn | 0 usable (all `skipped`) |

The deep methods (ft_transformer, tabnet, clinical_embeddings) ran on **outcome**
and **mortality_rate_yn** (all 4 phases), and on **serious_adverse_rate_yn** only
Phase 1–2. They are **entirely absent** from **patient_dropout_rate_yn** and
**failure_reason** — no record of any kind, not even a `skipped`/`error` stub.

That absence is diagnostic. The runner writes a JSON record for *every* cell it
reaches (`ok`, `skipped`, `no_data`, or `error` — see `run_benchmark.py` lines
135–160). Cells with no file at all were never reached, which means the Tier B/C
pass was **cut off partway through `serious_adverse_rate_yn`** — it never got to
the last two tasks. The most likely cause is wall-clock: `clinical_embeddings`
averages **51 minutes per cell and peaked at 2h10m** (Bio_ClinicalBERT inference
on CPU), versus ~1–2 s for the Tier A models. This snapshot is therefore either
(a) a mid-run capture while the deep pass is still grinding, or (b) a run that was
killed and not resumed. **Worth confirming whether the EC2 job is still running.**

`tabpfn` is a clean, expected skip: no `TABPFN_TOKEN` license set, so every cell
recorded `status: "skipped"` with a helpful reason — good graceful-degradation
behavior, exactly as designed.

### The `.benchmark_complete` marker is stale — don't trust it

`results/.benchmark_complete` exists, which normally signals a clean full-grid
finish. Here it is misleading: `deploy/run_forever.sh` only touches the marker on
a clean `exit 0`, and its own comments explain that the config grows over time
(Tier A → +Tier B → +Tier C) and that **nothing gates on the marker**. So it was
almost certainly written when the earlier, smaller (Tier A) grid completed, and
it does **not** mean the current 16-method grid finished. Treat completeness as
"count the run files," not "the marker is present."

---

## 2. Leaderboard bug — the "Overall" ranking is not apples-to-apples

`eval/leaderboard.py` builds per-task means and the Overall ranking with
`pivot.mean(axis=1)` / `groupby(...).mean()`. **pandas skips NaN by default**, so
each method is averaged over *only the cells it actually ran*. Combined with the
incomplete coverage above, this produces an invalid comparison:

- ft_transformer is ranked **#1 overall (0.7297)** — but that average is over only
  its 10 completed cells, which happen to **exclude `failure_reason` entirely**
  (the hardest task, mean PR-AUC ≈ 0.30 for everyone).
- In the per-task table, ft_transformer's `serious_adverse` mean (0.8947) is the
  average of just Phase 1–2; the `nan`s for Phase 3–4 are silently dropped, so it
  is compared against other methods' 4-phase means.

When every method is scored on the **same 10 cells ft_transformer actually ran**,
the order flips:

| Fair (same 10 cells) | PR-AUC | | As-published "Overall" | PR-AUC | cells |
|---|---|---|---|---|---|
| **tfidf_logreg** | **0.7601** | | ft_transformer | 0.7297 | 10 |
| ft_transformer | 0.7297 | | tfidf_logreg | 0.7112 | 20 |
| xgboost | 0.6870 | | xgboost | 0.6465 | 20 |
| lightgbm | 0.6747 | | logreg_l2 | 0.6403 | 20 |

So the defensible headline from the data as it stands is: **TF-IDF + Logistic
Regression is the strongest method**, and it is complete across all 20 cells.
ft_transformer looks competitive only where it has run. The current leaderboard
inverts that.

**Fix:** rank only over the set of task×phase cells that *all* compared methods
have completed (or clearly separate "full-grid" from "partial" methods and never
mix them in one ranking). Optionally treat a `nan` phase as excluding the method
from that task's ranking rather than shrinking its denominator.

---

## 3. Threshold-dependent metrics are degenerate for 19% of binary cells

`eval/metrics.py` computes F1 / precision / recall / specificity / accuracy at a
**fixed 0.5 threshold**. On these imbalanced tasks the models often never emit a
score ≥ 0.5, so they predict all-negative: **43 of 221 binary cells (19%) report
F1 = precision = recall = 0, specificity = 1.0.** Example — `mortality_rate_yn /
Phase1 / catboost`: AUROC 0.732, PR-AUC 0.455 (both fine), but F1/recall = 0.

This contradicts the stated protocol (PLAN §5 and CLAUDE.md golden rule #2: "all
thresholding uses validation only"); no validation-based threshold selection is
implemented. The **headline ranking is unaffected** because PR-AUC and AUROC are
threshold-free — but the F1/precision/recall columns are currently not meaningful
and shouldn't be reported as-is. Fix by tuning the threshold on validation (e.g.
max-F1 or a fixed target recall) before scoring test, or drop those columns and
report only threshold-free metrics.

---

## 4. What's solid (keep it)

- **Leakage safety is real.** `TabularFeaturizer` fits medians, target-mean
  encodings, and one-hot levels on train only, then applies to valid/test. The
  TF-IDF vectorizer (`text_nlp.py`) is fit on train only. `clinical_embeddings`
  fits its scaler+LogReg on train embeddings only. No test leakage found.
- **Imbalance handled consistently:** `class_weight="balanced"` /
  `scale_pos_weight` / `auto_class_weights="Balanced"` across the linear, tree,
  and GBM methods; PR-AUC as the headline; 1000-resample bootstrap CIs.
- **Orchestration is genuinely production-grade.** Resumable (skip-if-exists),
  atomic `os.replace` writes, per-cell try/except that records skips/errors
  instead of crashing, a supervisor that distinguishes real progress from a
  missing-data "fake completion," and a systemd unit for reboot survival.
- **Embedding cache** keyed by (model, row NCT ids) avoids re-embedding — the
  right call given the CPU cost.

---

## 5. Smaller issues and notes

- **Binary label encoding is fragile.** `loader.py` does
  `pd.to_numeric(y, errors="coerce").fillna(0)`. If a `Y/N` column ever contains
  the literal strings "Y"/"N" (rather than 1/0), every label silently becomes 0
  and the task looks trivially solvable. The results have real signal, so the
  data is numeric here — but this should assert/validate rather than coerce
  silently.
- **Validation split uses the run seed, not a fixed 42.** `run_cell` passes the
  run `seed` into `load_task_phase`, so a multi-seed run re-draws the train/valid
  split each seed. Fine for variance estimation, but it differs from the "fixed
  42 valid split" described in CLAUDE.md. No impact on the single-seed (42) run
  here.
- **Bootstrap stores a metric subset.** For binary, `precision`/`recall`/
  `specificity` are computed for the point estimate but not carried in the
  bootstrap dict (only auroc/prauc/f1/accuracy). Minor, but the JSON is
  inconsistent between `point` and `bootstrap`.
- **High-cardinality categoricals are silently dropped** in the multiclass
  one-hot path (`features.py`, cap = 40 levels). Reasonable, but undocumented in
  the output — worth logging what got dropped.
- **`catboost_info/` is committed** to the repo root (CatBoost's default
  training-log dir, written to cwd). Add to `.gitignore`; it's build noise.
- **`mortality_rate_yn` and `serious_adverse_rate_yn` share identical
  train/test sizes** across all phases. Expected (same trial set, two safety
  endpoints) and their labels/scores differ, so not an aliasing bug — just worth
  a one-line confirmation.

---

## 6. Recommended next steps (in priority order)

1. **Confirm the EC2 job's state.** Check `systemctl status
   trialbench-benchmark` / `logs/supervisor.log`. If it died, resume it; if it's
   running, the deep cells will keep filling in.
2. **Decide the CPU budget for `clinical_embeddings`.** At ~1–2h/cell it is the
   reason the grid can't finish. Options: cap rows, shorten `MAX_LENGTH`,
   precompute embeddings for all tasks up front, or mark it "partial (compute)"
   and exclude it from the headline ranking.
3. **Fix `leaderboard.py`** to rank only on cells shared by the compared methods,
   and label partial-coverage methods explicitly. This alone changes the
   headline.
4. **Add validation-based thresholding** (or drop threshold-dependent columns) so
   F1/precision/recall are meaningful.
5. **Add a completeness check** to the leaderboard/README: cells present vs.
   expected per method, so "done" is measured from run files, not the marker.
6. Optionally, add the `TABPFN_TOKEN` to unlock tabpfn, and consider a multi-seed
   run for the fast Tier A methods to report variance.

**Bottom line for reporting today:** quote only the Tier A results and lead with
**tfidf_logreg**; hold the deep-method comparison until the grid is complete and
the aggregation is fixed.
