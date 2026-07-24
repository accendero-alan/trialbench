# TrialBench classification benchmark

A reproducible, CPU-only harness for benchmarking a variety of predictive
methods on the **classification tasks** of
[TrialBench](https://github.com/ML2Health/ML2ClinicalTrials) — trial approval,
mortality, serious adverse event, patient dropout (binary), and failure-reason
identification (multiclass) — across trial phases I–IV.

See **[PLAN.md](PLAN.md)** for the full plan (method menu, evaluation protocol,
milestones) and **[CLAUDE.md](CLAUDE.md)** for conventions.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Verify the harness end-to-end on synthetic data (no download needed):
python tests/test_smoke.py

# 2) Get the data (direct Zenodo download of the 5 classification tasks; no torch needed):
python -m src.data.download
#    ...or the official trialbench pip package (needs torch; downloads all 8 tasks):
python -m src.data.download --via-package
#    ...or use the toy samples from a local clone to develop against:
python -m src.data.download --from-clone /path/to/ML2ClinicalTrials

# 3) Run the core (Tier A) benchmark on CPU:
python -m src.run_benchmark
#    quick dry run:
python -m src.run_benchmark --methods logreg_l2 xgboost --phases Phase1 --max-test-rows 500
```

Results land in `results/`: per-cell JSON in `results/runs/`, plus
`leaderboard.md` / `leaderboard.csv` ranking methods by PR-AUC.

## Running on EC2

`python -m src.run_benchmark` is resumable by design (it skips any
task/phase/method/seed cell whose result JSON already exists), so restarting
it after any interruption is always safe. See **[deploy/README.md](deploy/README.md)**
for pushing this repo to an EC2 instance, bootstrapping it, and running with
auto-restart on both process crashes and instance reboots.

## What's implemented vs. stubbed

**Ready to run (Tier A, CPU):** `majority`, `logreg_l1/l2`, `random_forest`,
`extra_trees`, `hist_gbm`, `knn`, `svm_linear`, `xgboost`, `lightgbm`,
`catboost`, `tfidf_logreg`.

**Ready to run (Tier B, needs `torch`):** `tabnet` (pytorch-tabnet) and
`ft_transformer` (a compact FT-Transformer hand-rolled in plain PyTorch — see
`src/methods/deep_tabular.py`) work out of the box once torch + the extended
requirements are installed (`deploy/bootstrap_ec2.sh --extended`, or
`pip install torch --index-url https://download.pytorch.org/whl/cpu` then
`pip install -r requirements-extended.txt`). `tabpfn` also needs a one-time
free license token (see `deploy/README.md`) — without one it's recorded as
"skipped", not an error, and every other method still runs.

**Stubs with implementation notes (enable in `configs/benchmark.yaml` once
filled in):** `clinical_embeddings` (Tier C); `fingerprint_fusion`,
`hint_reference` (Tier D multimodal); `llm_fewshot` (Tier D). Each stub's
docstring describes exactly what to build.

## Layout

```
configs/benchmark.yaml   tasks × phases × methods × seeds
src/data/loader.py       task→folder/label map, splits
src/data/features.py     leakage-safe tabular featurizer + concat_text
src/data/download.py     data acquisition
src/methods/             base + registry + method families
src/eval/metrics.py      metrics + bootstrap (mirrors TrialBench)
src/eval/leaderboard.py  aggregate runs → leaderboard
src/run_benchmark.py     resumable runner (CLI)
tests/test_smoke.py      synthetic end-to-end test
```

## Adding a method

Create a class in the right `src/methods/*.py`, subclass `BaseMethod`, set
`feature_view` (`"tabular"` or `"raw"`), implement `fit` / `predict_proba`,
decorate with `@register("your_name")`, and add `your_name` to the config.
Binary `predict_proba` returns shape `(n,)` = P(y=1); multiclass returns `(n, C)`.
