#!/usr/bin/env bash
# Bootstrap a fresh EC2 instance to run the TrialBench classification benchmark.
#
# Idempotent: safe to re-run (e.g. after a reboot) — it will skip work that's
# already done. Run this ON the instance, from the repo root, after the code
# has landed there (see sync_to_ec2.sh).
#
#   ./deploy/bootstrap_ec2.sh                 # core deps + data + smoke test
#   ./deploy/bootstrap_ec2.sh --extended       # also install requirements-extended.txt
#   ./deploy/bootstrap_ec2.sh --skip-data      # skip the TrialBench download step
#   ./deploy/bootstrap_ec2.sh --skip-smoke     # skip the self-verification smoke test
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

INSTALL_EXTENDED=0
SKIP_DATA=0
SKIP_SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --extended) INSTALL_EXTENDED=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

echo "==> repo root: $HERE"

# --- 1. System packages (OS-detecting: Amazon Linux/RHEL dnf, Debian/Ubuntu apt) ---
if [ -f /etc/os-release ]; then . /etc/os-release; fi
PKG_MGR=""
if command -v dnf >/dev/null 2>&1; then PKG_MGR=dnf
elif command -v yum >/dev/null 2>&1; then PKG_MGR=yum
elif command -v apt-get >/dev/null 2>&1; then PKG_MGR=apt
fi

echo "==> installing system packages (${PKG_MGR:-none detected})"
case "$PKG_MGR" in
  dnf|yum)
    sudo "$PKG_MGR" -y update -q
    sudo "$PKG_MGR" -y install -q python3 python3-pip python3-devel gcc gcc-c++ make git tmux htop
    ;;
  apt)
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-venv python3-pip python3-dev build-essential git tmux htop
    ;;
  *)
    echo "    no known package manager found; assuming python3/build tools are already present"
    ;;
esac

# --- 2. Virtualenv ---
if [ ! -d .venv ]; then
  echo "==> creating .venv"
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -q --upgrade pip

# --- 3. Python dependencies ---
echo "==> installing requirements.txt"
pip install -q -r requirements.txt
if [ "$INSTALL_EXTENDED" = "1" ]; then
  echo "==> installing requirements-extended.txt (uncommented lines only)"
  grep -vE '^\s*#|^\s*$' requirements-extended.txt > /tmp/req-extended-active.txt || true
  if [ -s /tmp/req-extended-active.txt ]; then
    pip install -q -r /tmp/req-extended-active.txt
  else
    echo "    (nothing uncommented in requirements-extended.txt — edit it first to select Tier B/C/D deps)"
  fi
fi

# --- 4. Data ---
mkdir -p logs results/runs
if [ "$SKIP_DATA" = "1" ]; then
  echo "==> skipping data download (--skip-data)"
elif [ -d data ] && [ -n "$(ls -A data 2>/dev/null)" ]; then
  echo "==> data/ already populated, skipping download"
else
  echo "==> downloading TrialBench data"
  python -m src.data.download || {
    echo "    automatic download failed — see the Zenodo hint above and populate data/ manually,"
    echo "    then re-run this script with --skip-data."
    exit 1
  }
fi

# --- 5. Self-verify ---
if [ "$SKIP_SMOKE" = "1" ]; then
  echo "==> skipping smoke test (--skip-smoke)"
else
  echo "==> running smoke test"
  python tests/test_smoke.py
fi

echo
echo "==> bootstrap complete. Next:"
echo "    source .venv/bin/activate"
echo "    ./deploy/run_forever.sh                     # start the benchmark with auto-restart"
echo "    # or, for reboot-resilience via systemd:"
echo "    sudo cp deploy/trialbench-benchmark.service /etc/systemd/system/"
echo "    sudo systemctl daemon-reload && sudo systemctl enable --now trialbench-benchmark"
