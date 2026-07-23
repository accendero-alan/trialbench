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

# Directories we only ever want excluded at the REPO ROOT (they're produced on
# the instance, not part of the source tree). NOTE: these must be passed as an
# explicit top-level include-list rather than as --exclude patterns — both
# rsync and (especially) bsdtar treat a slash-less exclude pattern as "match
# this basename at ANY depth", so `--exclude data` also strips src/data/
# (the source package) since it shares a basename with the root-level data/
# dataset cache. Filtering the top-level listing in the shell sidesteps that
# entirely.
TOP_LEVEL_EXCLUDES=(.git .venv .pytest_cache .claude data results logs catboost_info)
# These, by contrast, we DO want excluded at every depth, and there's no name
# collision with anything under src/ or tests/, so plain --exclude is safe.
ANY_DEPTH_EXCLUDES=(__pycache__ "*.pyc" ".DS_Store")

shopt -s dotglob nullglob
INCLUDE_ITEMS=()
for f in *; do
  skip=0
  for e in "${TOP_LEVEL_EXCLUDES[@]}"; do
    [ "$f" = "$e" ] && skip=1 && break
  done
  [ "$skip" = 0 ] && INCLUDE_ITEMS+=("$f")
done
shopt -u dotglob nullglob

echo "==> syncing $HERE -> $EC2_HOST:$REMOTE_DIR"
echo "    (${#INCLUDE_ITEMS[@]} top-level items: ${INCLUDE_ITEMS[*]})"

ssh "${SSH_OPTS[@]}" "$EC2_HOST" "mkdir -p '$REMOTE_DIR'"

if command -v rsync >/dev/null 2>&1; then
  RSYNC_EXCLUDES=()
  for e in "${ANY_DEPTH_EXCLUDES[@]}"; do RSYNC_EXCLUDES+=(--exclude "$e"); done
  rsync -avz --progress "${RSYNC_EXCLUDES[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    "${INCLUDE_ITEMS[@]}" "$EC2_HOST:$REMOTE_DIR/"
else
  echo "    rsync not found locally; falling back to tar+ssh (full copy, no incremental delta)"
  TAR_EXCLUDES=()
  for e in "${ANY_DEPTH_EXCLUDES[@]}"; do TAR_EXCLUDES+=(--exclude="$e"); done
  tar czf - "${TAR_EXCLUDES[@]}" -- "${INCLUDE_ITEMS[@]}" | ssh "${SSH_OPTS[@]}" "$EC2_HOST" "tar xzf - -C '$REMOTE_DIR'"
fi

echo
echo "==> done. Next, on the instance:"
echo "    ssh ${EC2_KEY:+-i $EC2_KEY} $EC2_HOST"
echo "    cd $REMOTE_DIR && ./deploy/bootstrap_ec2.sh"
