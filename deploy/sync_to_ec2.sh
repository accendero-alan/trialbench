#!/usr/bin/env bash
# Push this repo up to an EC2 instance. Run from your local machine (Git Bash,
# WSL, macOS, Linux). Uses rsync when available (fast, incremental — safe to
# re-run after editing configs/benchmark.yaml while a run is in progress on
# the instance); falls back to a tar+ssh stream if rsync isn't installed.
#
# Excludes data/, results/, logs/, .venv/, and other local-only artifacts —
# those are produced/managed on the instance itself (see bootstrap_ec2.sh).
#
# Usage:
#   EC2_HOST=ec2-user@1.2.3.4 EC2_KEY=~/.ssh/my-key.pem ./deploy/sync_to_ec2.sh
#   EC2_HOST=my-instance-alias ./deploy/sync_to_ec2.sh          # using ~/.ssh/config
#   REMOTE_DIR=/home/ubuntu/trialbench ./deploy/sync_to_ec2.sh  # custom remote path
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${EC2_HOST:?set EC2_HOST, e.g. EC2_HOST=ec2-user@1.2.3.4 ./deploy/sync_to_ec2.sh}"
REMOTE_DIR="${REMOTE_DIR:-~/trialbench-classification-benchmark}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[ -n "${EC2_KEY:-}" ] && SSH_OPTS+=(-i "$EC2_KEY")

EXCLUDES=(.git .venv __pycache__ .pytest_cache data results logs catboost_info "*.pyc" ".DS_Store")

echo "==> syncing $HERE -> $EC2_HOST:$REMOTE_DIR"

if command -v rsync >/dev/null 2>&1; then
  RSYNC_EXCLUDES=()
  for e in "${EXCLUDES[@]}"; do RSYNC_EXCLUDES+=(--exclude "$e"); done
  ssh "${SSH_OPTS[@]}" "$EC2_HOST" "mkdir -p '$REMOTE_DIR'"
  rsync -avz --progress "${RSYNC_EXCLUDES[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    ./ "$EC2_HOST:$REMOTE_DIR/"
else
  echo "    rsync not found locally; falling back to tar+ssh (full copy, no incremental delta)"
  TAR_EXCLUDES=()
  for e in "${EXCLUDES[@]}"; do TAR_EXCLUDES+=(--exclude="./$e"); done
  ssh "${SSH_OPTS[@]}" "$EC2_HOST" "mkdir -p '$REMOTE_DIR'"
  tar czf - "${TAR_EXCLUDES[@]}" . | ssh "${SSH_OPTS[@]}" "$EC2_HOST" "tar xzf - -C '$REMOTE_DIR'"
fi

echo
echo "==> done. Next, on the instance:"
echo "    ssh ${EC2_KEY:+-i $EC2_KEY} $EC2_HOST"
echo "    cd $REMOTE_DIR && ./deploy/bootstrap_ec2.sh"
