#!/usr/bin/env bash
# Install and enable the systemd unit so the benchmark auto-resumes across
# process crashes AND instance reboots (e.g. spot interruption + relaunch,
# maintenance reboot). Run with sudo from the repo root on the EC2 instance.
#
#   sudo ./deploy/install_service.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "run this with sudo: sudo ./deploy/install_service.sh" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
UNIT_SRC="$HERE/deploy/trialbench-benchmark.service"
UNIT_DST="/etc/systemd/system/trialbench-benchmark.service"

echo "==> repo dir:  $HERE"
echo "==> run as:    $RUN_USER"
echo "==> unit file: $UNIT_DST"

sed -e "s#__REPO_DIR__#$HERE#g" -e "s#__RUN_USER__#$RUN_USER#g" "$UNIT_SRC" > "$UNIT_DST"

chmod +x "$HERE/deploy/run_forever.sh"
mkdir -p "$HERE/logs"
chown -R "$RUN_USER" "$HERE/logs" "$HERE/results" 2>/dev/null || true

systemctl daemon-reload
systemctl enable --now trialbench-benchmark

echo
echo "==> installed. Useful commands:"
echo "    systemctl status trialbench-benchmark"
echo "    journalctl -u trialbench-benchmark -f"
echo "    tail -f $HERE/logs/supervisor.log"
echo "    tail -f $HERE/results/leaderboard.md"
