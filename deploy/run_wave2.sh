#!/usr/bin/env bash
# P13.11 (wave2-start-plan.md): Wave 2's launch script, every flag baked in --
# per B2's lesson (wave1-preflight-review.md) that a pasted command line is a
# defect waiting to happen. Extends run_forever.sh (unchanged, already a
# generic passthrough wrapper) rather than replacing it, so crash-resilience
# and the resume/skip-if-exists behavior in run_benchmark.py still apply.
#
# run_benchmark.py's own grid is a strict cross-product of tasks × phases ×
# methods × seeds -- the amendment's five cells (configs/wave2_amendment.yaml)
# are NOT a rectangle over tasks × phases (e.g. mortality only has Phase3,
# serious_adverse_rate_yn has Phase2 AND Phase3), so this script invokes
# run_benchmark once per (model × arm × cell) combination rather than trying
# to express the amendment as --tasks/--phases lists. Run order: cheapest
# model first (fail fast on a harness bug before spending on the top rung).
#
# Requires (not created by this script -- see the plan's own W1/P14 gates,
# and the comment block below on the execution role):
#   - AWS credentials for an execution role with:
#       bedrock:InvokeModel, bedrock:CreateModelInvocationJob,
#       bedrock:GetModelInvocationJob, bedrock:CreatePromptRouter,
#       s3:GetObject / s3:PutObject -- only actually needed once batch
#       submission is wired in, see the LLM_SERVICE_TIER note just below
#   - configs/wave2_amendment.yaml resolved (P13.10 refuses otherwise, per model/cell)
#   - Per-cell fixed test subsets (P13.7) -- generated below if missing
#
# Results land in a local dir (RESULTS_DIR below, same as Wave 1) --
# deploy/fetch_wave2_results.sh pulls it back over SSH/rsync; push whatever
# needs long-term archival to SharePoint yourself
# (deploy/sharepoint_transfer.py) afterward. There is no S3 durability layer
# in this repo -- S3 here is only Bedrock batch inference's required I/O
# staging area (src/bedrock/batch.py), separate from where results live.
#
# KNOWN GAP, found 2026-08-27 (wave2-start-plan.md P13.8's status note):
# LLM_SERVICE_TIER=batch is this script's own default below, but
# src/methods/llm.py -> src/bedrock/client.py never actually submits a batch
# job -- every call is real-time synchronous regardless of this flag, while
# the cost meter still reports the batch discount as if it applied. Do not
# run this for a real billable pass until that's fixed or you've
# consciously decided to eat the accounting error; --llm-service-tier sync
# below is at least an HONEST label for what actually happens today.
#
# Usage:
#   ./deploy/run_wave2.sh                      # every model × arm × cell
#   MODELS="amazon.nova-lite" ./deploy/run_wave2.sh   # one model only (e.g. a smoke pass)
#   ARMS="L0 L1" ./deploy/run_wave2.sh                # a subset of arms
#   LLM_SERVICE_TIER=sync ./deploy/run_wave2.sh       # honest label until batch submission is wired in
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi

RESULTS_DIR="${RESULTS_DIR:-results_wave2}"
AMENDMENT_FILE="${AMENDMENT_FILE:-configs/wave2_amendment.yaml}"
LLM_SERVICE_TIER="${LLM_SERVICE_TIER:-batch}"

# L6 excluded: NotImplementedError until a MONDO/PrimeKG source and scrub
# list are committed (P13.6) -- not this script's decision to make.
ARMS="${ARMS:-L0 L1 L2 L3 L4 L5 L7}"

MODELS="${MODELS:-$($PY -c "
import yaml
with open('$AMENDMENT_FILE') as f:
    print(' '.join(yaml.safe_load(f)['models']))
")}"

CELLS_TSV="$($PY -c "
import yaml
with open('$AMENDMENT_FILE') as f:
    amendment = yaml.safe_load(f)
default_repeats = amendment.get('default_repeats', 1)
for c in amendment['cells']:
    print(c['task'], c['phase'], c.get('repeats', default_repeats), sep='\t')
")"

mkdir -p "$RESULTS_DIR" logs results/subsets

echo "==> Wave 2 launch: models=[$MODELS] arms=[$ARMS] service_tier=$LLM_SERVICE_TIER"
echo "==> cells (from $AMENDMENT_FILE):"
echo "$CELLS_TSV" | sed 's/^/    /'

# P13.7: generate each cell's fixed test subset once, up front, if it doesn't
# already exist -- L6 coverage ordering (P13.6's constraint) doesn't apply
# here since L6 is excluded above.
while IFS=$'\t' read -r task phase repeats; do
  subset_file="results/subsets/${task}_${phase}.txt"
  if [ ! -f "$subset_file" ]; then
    echo "==> generating fixed test subset: $subset_file"
    "$PY" -m src.data.subset --task "$task" --phase "$phase" --out "$subset_file"
  fi
done <<< "$CELLS_TSV"

for model in $MODELS; do
  for arm in $ARMS; do
    while IFS=$'\t' read -r task phase repeats; do
      subset_file="results/subsets/${task}_${phase}.txt"
      seeds=""
      for i in $(seq 1 "$repeats"); do
        # Deterministic multi-seed set matching the repo's own convention
        # (configs/benchmark.yaml's example: 42, 7, 123, ...) -- only the
        # designated repeat cell (§6.5) ever asks for more than one.
        case "$i" in
          1) seeds="$seeds 42" ;;
          2) seeds="$seeds 7" ;;
          3) seeds="$seeds 123" ;;
          *) echo "no seed defined for repeat index $i (extend this script if §6.5 grows)" >&2; exit 1 ;;
        esac
      done

      echo "==> $model / $arm / $task / $phase (repeats=$repeats, seeds=$seeds)"
      RESULTS_DIR="$RESULTS_DIR" ./deploy/run_forever.sh \
        --config configs/benchmark.yaml \
        --methods llm_probability \
        --tasks "$task" --phases "$phase" \
        --seeds $seeds \
        --llm-arm "$arm" --llm-model "$model" \
        --llm-service-tier "$LLM_SERVICE_TIER" \
        --test-subset-file "$subset_file" \
        --results-dir "$RESULTS_DIR"
    done <<< "$CELLS_TSV"
  done
done

echo
echo "==> Wave 2 grid complete. See $RESULTS_DIR/leaderboard.md and $RESULTS_DIR/runs/*.json."
