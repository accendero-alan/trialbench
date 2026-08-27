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
#       s3:GetObject / s3:PutObject on WAVE2_S3_BUCKET below -- required
#       for the LLM_SERVICE_TIER=batch default (deploy/w1_permissions_check.py
#       checks these before you spend anything)
#   - WAVE2_S3_BUCKET and WAVE2_BATCH_ROLE_ARN set, if using batch (the default)
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
# FIXED 2026-08-27 (was: KNOWN GAP -- LLM_SERVICE_TIER=batch was a label
# nothing enforced; see wave2-start-plan.md P13.8's status note for the
# history). Batch submission is now wired end to end (src/bedrock/batch.py
# -> src/methods/llm.py's _predict_batch), and the cost meter tracks the
# tier each call was *actually* billed at rather than trusting the
# requested one. **Still unverified against a live account**: the
# per-provider modelInput format (src/bedrock/batch_formats.py) is
# HIGH-confidence for Anthropic/Nova, LOW-confidence/UNVERIFIED for Llama/
# DeepSeek, and the batch-output S3 key layout is a documented-but-unverified
# guess (src/bedrock/batch.py's fetch_batch_output_records). Run a small
# smoke pass per model (MODELS=<one> ARMS=L1 below, or
# deploy/w1_bedrock_inventory.py --run-batch-probe) before trusting this for
# the real grid -- a wrong format fails per-record (a parse-failure-rate
# warning), not silently, but you want to know before 1,000 trials, not after.
#
# Usage:
#   ./deploy/run_wave2.sh                      # every model × arm × cell
#   MODELS="amazon.nova-lite" ./deploy/run_wave2.sh   # one model only (e.g. a smoke pass)
#   ARMS="L0 L1" ./deploy/run_wave2.sh                # a subset of arms
#   LLM_SERVICE_TIER=sync ./deploy/run_wave2.sh       # skip batch entirely (no S3/role needed)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi

RESULTS_DIR="${RESULTS_DIR:-results_wave2}"
AMENDMENT_FILE="${AMENDMENT_FILE:-configs/wave2_amendment.yaml}"
LLM_SERVICE_TIER="${LLM_SERVICE_TIER:-batch}"
WAVE2_S3_BUCKET="${WAVE2_S3_BUCKET:-}"
WAVE2_BATCH_ROLE_ARN="${WAVE2_BATCH_ROLE_ARN:-}"
if [ "$LLM_SERVICE_TIER" = "batch" ] && { [ -z "$WAVE2_S3_BUCKET" ] || [ -z "$WAVE2_BATCH_ROLE_ARN" ]; }; then
  echo "LLM_SERVICE_TIER=batch requires WAVE2_S3_BUCKET and WAVE2_BATCH_ROLE_ARN -- set both, or" >&2
  echo "run with LLM_SERVICE_TIER=sync if a batch bucket/role isn't set up yet." >&2
  exit 1
fi
BATCH_FLAGS=()
[ -n "$WAVE2_S3_BUCKET" ] && BATCH_FLAGS+=(--llm-s3-bucket "$WAVE2_S3_BUCKET")
[ -n "$WAVE2_BATCH_ROLE_ARN" ] && BATCH_FLAGS+=(--llm-batch-role-arn "$WAVE2_BATCH_ROLE_ARN")
[ -n "${WAVE2_BATCH_MIN_RECORDS:-}" ] && BATCH_FLAGS+=(--llm-batch-min-records "$WAVE2_BATCH_MIN_RECORDS")

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
        "${BATCH_FLAGS[@]}" \
        --test-subset-file "$subset_file" \
        --results-dir "$RESULTS_DIR"
    done <<< "$CELLS_TSV"
  done
done

echo
echo "==> Wave 2 grid complete. See $RESULTS_DIR/leaderboard.md and $RESULTS_DIR/runs/*.json."
