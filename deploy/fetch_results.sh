#!/usr/bin/env bash
# Pull results/ and logs/ back down from the EC2 instance. Safe to run at any
# time, including while the benchmark is still running (you'll get whatever
# results/leaderboard.md looks like as of the last periodic rebuild).
#
# Usage:
#   EC2_HOST=ec2-user@1.2.3.4 EC2_KEY=~/.ssh/my-key.pem ./deploy/fetch_results.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${EC2_HOST:?set EC2_HOST, e.g. EC2_HOST=ec2-user@1.2.3.4 ./deploy/fetch_results.sh}"
REMOTE_DIR="${REMOTE_DIR:-~/trialbench-classification-benchmark}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[ -n "${EC2_KEY:-}" ] && SSH_OPTS+=(-i "$EC2_KEY")

mkdir -p results logs

echo "==> fetching results/ and logs/ from $EC2_HOST:$REMOTE_DIR"
if command -v rsync >/dev/null 2>&1; then
  rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
    "$EC2_HOST:$REMOTE_DIR/results/" ./results/
  rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
    "$EC2_HOST:$REMOTE_DIR/logs/" ./logs/ 2>/dev/null || true
else
  scp -r "${SSH_OPTS[@]}" "$EC2_HOST:$REMOTE_DIR/results" .
  scp -r "${SSH_OPTS[@]}" "$EC2_HOST:$REMOTE_DIR/logs" . 2>/dev/null || true
fi

echo
echo "==> done. See results/leaderboard.md"
