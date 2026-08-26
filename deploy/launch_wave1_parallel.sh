#!/usr/bin/env bash
# Launch several Wave 1 rungs concurrently, each capped to its fair share of
# cores instead of every process independently claiming the whole machine
# (wave1-preflight-review.md H2 -- measured: five uncapped concurrent fits
# is usually slower than running them one at a time, because n_jobs=-1 and
# BLAS thread counts don't coordinate across processes).
#
# Usage:
#   ./deploy/launch_wave1_parallel.sh <rung> [<rung> ...]
#   ./deploy/launch_wave1_parallel.sh block full ccsr ancestors stack
#   CORES=16 ./deploy/launch_wave1_parallel.sh block full   # override detected core count
#
# Each rung runs as `THREADS=<cores/N> ./deploy/run_codes_sweep.sh <rung>` in
# the background, logging to logs/wave1_<rung>.log. Waits for all of them and
# reports which succeeded/failed. Re-running is safe/cheap for whichever
# rungs already finished (run_benchmark.py's per-cell resume, B3-guarded).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <rung> [<rung> ...]" >&2
  echo "  e.g.: $0 block full ccsr ancestors stack" >&2
  exit 1
fi

n_rungs="$#"
cores="${CORES:-$(python3 -c 'import os; print(os.cpu_count() or 1)')}"
threads=$(( cores / n_rungs ))
if [ "$threads" -lt 1 ]; then threads=1; fi

mkdir -p logs
echo "cores=$cores, rungs=$n_rungs -> THREADS=$threads per process"

pids=()
names=()
for rung in "$@"; do
  log="logs/wave1_${rung}.log"
  echo "  launching $rung -> $log (THREADS=$threads)"
  THREADS="$threads" ./deploy/run_codes_sweep.sh "$rung" >"$log" 2>&1 &
  pids+=("$!")
  names+=("$rung")
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  done:   ${names[$i]} (pid ${pids[$i]})"
  else
    echo "  FAILED: ${names[$i]} (pid ${pids[$i]}) -- see logs/wave1_${names[$i]}.log"
    failed=1
  fi
done

exit "$failed"
