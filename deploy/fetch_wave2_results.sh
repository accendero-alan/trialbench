#!/usr/bin/env bash
# P13.11 (wave2-start-plan.md): pull Wave 2 results back from the EC2
# instance over SSH -- same mechanism as deploy/fetch_results.sh (Wave 1),
# not S3. Wave 2 writes to a local results dir on the instance exactly like
# Wave 1 does; S3 in this repo is only Bedrock batch inference's required
# input/output staging area (src/bedrock/batch.py), not a results-durability
# mechanism. Long-term archival is a separate, manual step via
# deploy/sharepoint_transfer.py once you have the results locally -- this
# script's job is only instance -> your machine.
#
# Safe to run at any time, including mid-run (you get whatever
# leaderboard.md looks like as of the last periodic rebuild).
#
# Usage:
#   EC2_HOST=ec2-user@1.2.3.4 EC2_KEY=~/.ssh/my-key.pem ./deploy/fetch_wave2_results.sh
#   RESULTS_DIR=results_wave2 ./deploy/fetch_wave2_results.sh   # default; match run_wave2.sh's RESULTS_DIR
#   FETCH_CACHE=1 ./deploy/fetch_wave2_results.sh   # also pull results_wave2/cache/bedrock/ (large; usually skip)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${EC2_HOST:?set EC2_HOST, e.g. EC2_HOST=ec2-user@1.2.3.4 ./deploy/fetch_wave2_results.sh}"
REMOTE_DIR="${REMOTE_DIR:-~/trialbench-classification-benchmark}"
RESULTS_DIR="${RESULTS_DIR:-results_wave2}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[ -n "${EC2_KEY:-}" ] && SSH_OPTS+=(-i "$EC2_KEY")

mkdir -p "$RESULTS_DIR" logs

echo "==> fetching $RESULTS_DIR/ and logs/ from $EC2_HOST:$REMOTE_DIR"
if command -v rsync >/dev/null 2>&1; then
  RSYNC_EXCLUDE=()
  [ -z "${FETCH_CACHE:-}" ] && RSYNC_EXCLUDE=(--exclude "cache/bedrock/")
  rsync -avz --progress "${RSYNC_EXCLUDE[@]}" -e "ssh ${SSH_OPTS[*]}" \
    "$EC2_HOST:$REMOTE_DIR/$RESULTS_DIR/" "./$RESULTS_DIR/"
  rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
    "$EC2_HOST:$REMOTE_DIR/logs/" ./logs/ 2>/dev/null || true
else
  # scp has no exclude flag -- the response cache comes along if you're on
  # this path and FETCH_CACHE is unset; skip it by hand after if it's large.
  scp -r "${SSH_OPTS[@]}" "$EC2_HOST:$REMOTE_DIR/$RESULTS_DIR" .
  scp -r "${SSH_OPTS[@]}" "$EC2_HOST:$REMOTE_DIR/logs" . 2>/dev/null || true
fi

echo
echo "==> done. See $RESULTS_DIR/leaderboard.md"
echo "==> for longer-term archival, push what you need to SharePoint, e.g.:"
echo "        python deploy/sharepoint_transfer.py upload $RESULTS_DIR/leaderboard.md wave2/leaderboard.md"
