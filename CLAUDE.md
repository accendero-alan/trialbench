# CLAUDE.md — working conventions for this benchmark

This repo benchmarks predictive methods on TrialBench's **classification** tasks,
on **CPU by default**. A GPU is available as of 2026-08-20 for the
disease-representation campaign (SapBERT encoding, T25 fine-tuning, local
LLM inference in Wave 2 — see `../disease-representation-test-plan.md`).
**Tier A stays CPU-only** regardless. Read `PLAN.md` first; it is the source of
truth for scope, method tiers, and the evaluation protocol.

## Golden rules

1. **No leakage.** Every encoder / scaler / vectorizer / embedding is fit on the
   **train** split only, then applied to valid/test. The featurizer already
   enforces this — keep it that way when adding methods.
2. **Touch test once.** Do all model selection and thresholding on validation.
   Metrics are computed on test with the bootstrap in `src/eval/metrics.py`.
3. **PR-AUC is the headline metric** (tasks are imbalanced). Always report the
   full metric set, but rank by PR-AUC.
4. **Reuse TrialBench's splits, seeds, and metric formulas** so results are
   comparable to the paper. Seeds already mirror the repo (42 for the valid
   split; configurable seed list for multi-seed runs).
5. **Never let one method break the run.** Import heavy/optional deps lazily
   inside `fit`; the runner records missing deps / stubs as "skipped".

## The method contract (`src/methods/base.py`)

- `feature_view = "tabular"` → you receive dense numpy arrays from
  `TabularFeaturizer`. `feature_view = "raw"` → you receive the original pandas
  DataFrame (use `src.data.features.concat_text` for text).
- `fit(X_train, y_train, X_valid, y_valid)` then `predict_proba(X)`.
- `predict_proba` returns `(n,)` = P(y=1) for binary, `(n, num_classes)` for
  multiclass. `task_type`, `num_classes`, and `seed` are on `self`.

To add a method: implement the class, `@register("name")`, add `name` to
`configs/benchmark.yaml`. See stubs in `deep_tabular.py`, `multimodal.py`,
`llm.py`, `text_nlp.py` for worked-out plans.

## Recommended execution order

1. `pip install -r requirements.txt` then `python tests/test_smoke.py` (must pass).
2. `python -m src.data.download` (or `--from-clone`), then sanity-check shapes.
3. Tier A full run: `python -m src.run_benchmark` → first complete leaderboard.
4. Tier B/C/D: implement one stub, enable it in the config, re-run (the runner
   skips already-completed cells; use `--force` to recompute).

## Gotchas

- `failure_reason` is multiclass; test rows with a class unseen in train are
  dropped by the loader. Use macro-averaged metrics.
- CPU wall-clock: start slow methods with `--max-test-rows` / `--max-train-rows`
  and a single `--phases Phase1` before scaling.
- GPU methods (SapBERT, fine-tuned encoders, local LLMs) are scoped to the
  disease-representation campaign only — don't assume GPU availability
  elsewhere in this repo, and keep GPU-only code paths lazily imported per
  golden rule 5.
- `pandas.to_markdown` needs `tabulate` (in requirements.txt).
- Label columns differ by task (`Y/N` vs named targets); the loader resolves
  them via a candidate list — extend `TASKS` in `loader.py` if a column is named
  differently in your data drop.
