#!/usr/bin/env bash
# P13.11 (wave2-start-plan.md): pull Wave 2 artifacts back from S3.
#
# Unlike fetch_results.sh (SSH/rsync against an EC2 instance's local disk),
# Wave 2's instance is disposable and everything durable goes to S3 as it's
# produced -- this script pulls the bucket, not the instance. Fetches run
# records, prediction parquet, meter totals, and logs only. The response
# cache (src/bedrock/cache.py) is deliberately left in S3: large, not
# evidence, and re-fetchable on demand if a specific entry needs auditing.
#
# Usage:
#   WAVE2_S3_BUCKET=my-bucket WAVE2_S3_PREFIX=wave2 ./deploy/fetch_wave2_results.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${WAVE2_S3_BUCKET:?set WAVE2_S3_BUCKET, e.g. WAVE2_S3_BUCKET=my-bucket ./deploy/fetch_wave2_results.sh}"
WAVE2_S3_PREFIX="${WAVE2_S3_PREFIX:-wave2}"
RESULTS_DIR="${RESULTS_DIR:-results_wave2}"
S3_URI="s3://${WAVE2_S3_BUCKET}/${WAVE2_S3_PREFIX}"

mkdir -p "$RESULTS_DIR" logs

echo "==> fetching runs/, predictions/, leaderboard.md from ${S3_URI}/results/"
aws s3 sync "${S3_URI}/results/runs/" "${RESULTS_DIR}/runs/" --exclude "*.tmp"
aws s3 sync "${S3_URI}/results/predictions/" "${RESULTS_DIR}/predictions/"
aws s3 cp "${S3_URI}/results/leaderboard.md" "${RESULTS_DIR}/leaderboard.md" 2>/dev/null || \
  echo "    (no leaderboard.md yet -- run still in progress or not started)"

echo "==> fetching logs/"
aws s3 sync "${S3_URI}/logs/" "./logs/" || true

# Meter totals: pulled explicitly rather than swept up by the runs/ sync
# above, since T30/T31's cost artifacts (t30_cost_frontier.json,
# t31_router.json) live under results/ but outside runs/ -- they're written
# by the analysis scripts, not per-cell records.
echo "==> fetching cost/meter artifacts (t28a/t30/t31 json), if present"
aws s3 sync "${S3_URI}/results/" "${RESULTS_DIR}/" \
  --exclude "*" \
  --include "t28a_probe_gate.json" --include "t28_llm_disease_slot.json" \
  --include "t30_cost_frontier.json" --include "t31_router.json" \
  --include "t29_fresh_slice.json"

echo
echo "==> done. Response cache intentionally NOT fetched -- it stays in ${S3_URI}/results/cache/bedrock/."
echo "    See $RESULTS_DIR/leaderboard.md"
