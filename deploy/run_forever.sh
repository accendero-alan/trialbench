#!/usr/bin/env bash
# Crash-resilient supervisor for the benchmark run.
#
# `python -m src.run_benchmark` is already cell-resumable (it skips any
# task/phase/method/seed whose results/runs/*.json exists), so restarting it
# is always safe and cheap — a restart just re-does whatever cell was
# in-flight when the process died. This wrapper is what actually performs
# that restart when the *process itself* dies: OOM kill, a segfault in a
# native dependency (numpy/xgboost/etc.), a spot-instance interruption, or
# anything else that bypasses run_benchmark.py's own per-cell try/except.
#
# It does NOT protect against the instance rebooting — for that, install
# deploy/trialbench-benchmark.service (systemd restarts this script on boot).
#
# Usage:
#   ./deploy/run_forever.sh                      # full grid from configs/benchmark.yaml
#   ./deploy/run_forever.sh --phases Phase1       # passthrough args -> run_benchmark
#   MAX_RETRIES=50 ./deploy/run_forever.sh        # override the retry cap
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

LOCK_FILE="/tmp/trialbench-benchmark.lock"
LOG_DIR="$HERE/logs"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"
RESULTS_DIR="${RESULTS_DIR:-results}"
RUNS_DIR="$RESULTS_DIR/runs"
DONE_MARKER="$RESULTS_DIR/.benchmark_complete"

MAX_RETRIES="${MAX_RETRIES:-20}"        # consecutive attempts with NO new progress before giving up
BACKOFF_START="${BACKOFF_START:-30}"    # seconds
BACKOFF_MAX="${BACKOFF_MAX:-600}"       # seconds

mkdir -p "$LOG_DIR" "$RUNS_DIR"

# Prevent two supervisors racing on the same results dir (e.g. systemd + a manual nohup).
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another run_forever.sh is already running (lock: $LOCK_FILE)" | tee -a "$SUPERVISOR_LOG"
  exit 1
fi

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$SUPERVISOR_LOG"; }

if [ -f "$DONE_MARKER" ]; then
  log "found $DONE_MARKER — benchmark already completed. Delete it (or pass --force to a manual run) to redo."
  exit 0
fi

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

n_runs_dir_files() { ls "$RUNS_DIR" 2>/dev/null | wc -l | tr -d ' '; }

consecutive_no_progress=0
backoff=$BACKOFF_START
attempt=0

while true; do
  attempt=$((attempt + 1))
  before=$(n_runs_dir_files)
  attempt_log="$LOG_DIR/attempt_${attempt}.log"
  log "attempt #$attempt: starting run_benchmark (args: $*) -> $attempt_log"

  "$PY" -m src.run_benchmark "$@" >>"$attempt_log" 2>&1
  status=$?

  after=$(n_runs_dir_files)
  log "attempt #$attempt: exited with status $status ($before -> $after result files)"

  if [ "$status" -eq 0 ]; then
    log "run_benchmark finished cleanly. Marking complete."
    touch "$DONE_MARKER"
    exit 0
  fi

  if [ "$after" -gt "$before" ]; then
    consecutive_no_progress=0
    backoff=$BACKOFF_START
  else
    consecutive_no_progress=$((consecutive_no_progress + 1))
  fi

  if [ "$consecutive_no_progress" -ge "$MAX_RETRIES" ]; then
    log "giving up after $consecutive_no_progress consecutive attempts with no new progress."
    log "check $attempt_log for the failure and re-run manually once fixed:"
    log "  $PY -m src.run_benchmark $*"
    exit 1
  fi

  log "retrying in ${backoff}s (no-progress streak: $consecutive_no_progress/$MAX_RETRIES)"
  sleep "$backoff"
  backoff=$(( backoff * 2 < BACKOFF_MAX ? backoff * 2 : BACKOFF_MAX ))
done
