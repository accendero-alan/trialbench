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

# Deliberately NOT short-circuiting on $DONE_MARKER's mere presence: this
# project's whole workflow is "implement one more stub, enable it in
# configs/benchmark.yaml, re-run" (see CLAUDE.md), so a config that grows
# over time is the norm, not the exception. A marker that means "never
# invoke run_benchmark again" would silently ignore every method added
# after the first time the grid finished -- which is exactly what happened
# when Tier B and, separately, Tier C were added. run_benchmark.py's own
# skip-if-exists resume logic already makes a "nothing new to do" attempt
# cheap (~1-2s even at the full ~320-cell grid size), so there's no real
# cost to always giving it a chance to notice new work. We still touch
# DONE_MARKER below purely as an informational "has finished at least once"
# signal -- nothing here reads it back.
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

n_runs_dir_files() { ls "$RUNS_DIR" 2>/dev/null | wc -l | tr -d ' '; }

# run_benchmark.py catches FileNotFoundError *per cell* and records it as
# status "no_data" rather than crashing (see CLAUDE.md's "never let one
# method break the run") -- so a grid run against an empty/missing data/
# directory "completes" every cell and exits 0 just like a real run. Gate
# completion on there being zero no_data cells so a data problem can't get
# silently recorded as done forever.
n_no_data_files() { grep -l '"status": "no_data"' "$RUNS_DIR"/*.json 2>/dev/null | wc -l | tr -d ' '; }

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
    no_data=$(n_no_data_files)
    if [ "$no_data" -gt 0 ]; then
      log "run_benchmark exited cleanly but $no_data cell(s) have status \"no_data\" --"
      log "  that means the data dir (data/, or wherever --data-root points) is missing/incomplete,"
      log "  NOT that the benchmark actually ran. Refusing to mark complete."
      log "  Populate the data (python -m src.data.download, or deploy/bootstrap_ec2.sh), then clear"
      log "  the stale no_data results so they get recomputed on the next restart:"
      log "    grep -l '\"status\": \"no_data\"' $RUNS_DIR/*.json | xargs rm -f"
      exit 1
    fi
    log "run_benchmark finished cleanly with no no_data cells."
    touch "$DONE_MARKER"  # informational only -- see comment above; nothing gates on this file
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
