#!/usr/bin/env bash
# Launch one rung of the Wave 1 tabular+codes sweep with the flags the
# campaign requires baked in, instead of retyping the full command by hand.
#
# wave1-preflight-review.md B2: the bare `python -m src.run_benchmark`
# command runs configs/benchmark.yaml's defaults verbatim -- one seed
# (`seeds: [42]`) and sixteen methods, five of which (tabpfn, tabnet,
# ft_transformer, clinical_embeddings) are Tier B/C stubs that either OOM
# under tabular+codes' wide feature space (ft_transformer), run 3,000+s/cell
# (clinical_embeddings), or are license-gated (tabpfn). Every prior sweep
# passed the right flags by hand; this script is that command, not a new
# one, so a launch can't silently regress to the config's bare defaults.
#
# Usage:
#   ./deploy/run_codes_sweep.sh <rung> [extra run_benchmark args...]
#   ./deploy/run_codes_sweep.sh block
#   ./deploy/run_codes_sweep.sh block --force
#   ./deploy/run_codes_sweep.sh full --phases Phase1        # dry-run one phase first
#
# Runs run_benchmark directly (no crash-resilient supervision). For an
# unattended multi-hour run, wrap it the same way deploy/README.md documents
# for run_forever.sh, passing this script's own flags through:
#   RESULTS_DIR=results_codes_block ./deploy/run_forever.sh \
#     --feature-view-override tabular+codes --icd-granularity block \
#     --results-dir results_codes_block --seeds 42 7 123 2024 5 \
#     --methods majority logreg_l2 logreg_l1 random_forest extra_trees \
#               hist_gbm knn svm_linear xgboost lightgbm catboost
# i.e. run_forever.sh IS this same command with retry/backoff around it --
# this script exists so you don't have to retype the flags either way.
#
# H2 (wave1-preflight-review.md): running this rung alone, every method that
# parallelizes its own fit claims the whole machine (n_jobs=-1 by default) --
# fine solo. Running several rungs at once (see launch_wave1_parallel.sh)
# needs each process capped to its share instead, or they all fight over the
# same cores. Set THREADS to cap this process: it's passed through as
# --n-jobs (random_forest/extra_trees/knn/xgboost/lightgbm/catboost) *and*
# exported as OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS (BLAS-level
# threading inside logreg_l1/l2's solver and svm_linear's scaler/calibration,
# which don't take an n_jobs param at all -- env vars are the only lever for
# those, and have to be set before the venv's numpy/scikit-learn import, i.e.
# before `python` starts, not from inside run_benchmark.py):
#   THREADS=4 ./deploy/run_codes_sweep.sh block
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VALID_RUNGS=(char3 full chapter block ccsr ancestors stack)

rung="${1:-}"
if [ -z "$rung" ]; then
  echo "usage: $0 <rung> [extra run_benchmark args...]" >&2
  echo "  rung: one of ${VALID_RUNGS[*]}" >&2
  exit 1
fi
shift

valid=0
for r in "${VALID_RUNGS[@]}"; do
  [ "$r" = "$rung" ] && valid=1 && break
done
if [ "$valid" -ne 1 ]; then
  echo "error: unknown rung '$rung' -- must be one of ${VALID_RUNGS[*]}" >&2
  exit 1
fi

# char3 is T21's original baseline and keeps its existing directory name
# (results_codes/, not results_codes_char3/) -- see the run inventory in
# wave1-preflight-review.md.
if [ "$rung" = "char3" ]; then
  results_dir="results_codes"
else
  results_dir="results_codes_${rung}"
fi

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

n_jobs_args=()
if [ -n "${THREADS:-}" ]; then
  export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
  n_jobs_args=(--n-jobs "$THREADS")
fi

echo "launching rung=$rung -> $results_dir (THREADS=${THREADS:-unset, -1/whole machine}, extra args: $*)"
exec "$PY" -m src.run_benchmark \
  --feature-view-override tabular+codes \
  --icd-granularity "$rung" \
  --results-dir "$results_dir" \
  --seeds 42 7 123 2024 5 \
  --methods majority logreg_l2 logreg_l1 random_forest extra_trees \
            hist_gbm knn svm_linear xgboost lightgbm catboost \
  "${n_jobs_args[@]}" \
  "$@"
