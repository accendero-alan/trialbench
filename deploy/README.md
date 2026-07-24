# Deploying to EC2

Everything here assumes you already have an EC2 instance running (Amazon Linux
2023, Ubuntu 22.04/24.04, or similar) reachable over SSH. Nothing in this repo
provisions or launches AWS infrastructure itself — these scripts push code,
set the instance up, and run the benchmark with automatic restart.

**CPU sizing:** any compute-optimized or general-purpose instance is fine
(this is a CPU-only benchmark per `CLAUDE.md`/`PLAN.md`) — `c6i.2xlarge` (8
vCPU) or bigger is a reasonable default for Tier A across all 20 task×phase
cells. More RAM helps if you enable TF-IDF/embedding-heavy methods.

## 1. Push the code up

From your local machine:

```bash
# bash (Git Bash / WSL / macOS / Linux)
EC2_HOST=ec2-user@<instance-ip> EC2_KEY=~/.ssh/your-key.pem ./deploy/sync_to_ec2.sh
```

```powershell
# PowerShell
.\deploy\sync_to_ec2.ps1 -EC2Host ec2-user@<instance-ip> -EC2Key ~\.ssh\your-key.pem
```

`EC2_HOST` can also be a `~/.ssh/config` alias if you've already set one up.
This copies the repo minus `data/`, `results/`, `logs/`, `.venv/` — those are
produced on the instance. Re-run it any time you change `configs/benchmark.yaml`
or add a method; it's safe to sync onto a running instance.

## 2. Bootstrap the instance

SSH in and run the bootstrap script once:

```bash
ssh -i ~/.ssh/your-key.pem ec2-user@<instance-ip>
cd trialbench-classification-benchmark
./deploy/bootstrap_ec2.sh
```

This detects the package manager (apt or dnf/yum), installs system deps,
creates `.venv`, installs `requirements.txt`, downloads the TrialBench data
(`src/data/download.py` — direct Zenodo download of the 5 classification
tasks by default, no `trialbench`/torch needed for this step), and runs
`tests/test_smoke.py` to self-verify before you commit to a long run. Flags:

- `--extended` — installs the CPU-only `torch` wheel plus whatever you've
  uncommented in `requirements-extended.txt` (Tier B/C/D deps). Tier B
  (`tabnet`, `ft_transformer`) is enabled by default in
  `configs/benchmark.yaml` and works right away once this flag is used.
  `tabpfn` additionally needs a one-time **free** license token: visit
  https://ux.priorlabs.ai, accept the license, and copy your API key from
  the Account page. Then, depending on how you're running it:
  - **Manual/tmux run:** `export TABPFN_TOKEN="<key>"` in the same shell
    before `./deploy/run_forever.sh`.
  - **systemd:** create `.env` in the repo root (gitignored — never
    committed) with a line `TABPFN_TOKEN=<key>`, then
    `sudo systemctl restart trialbench-benchmark`. The unit reads it via
    `EnvironmentFile` — plain shell `export`/`~/.bashrc` won't reach a
    systemd-managed process, since it doesn't inherit your SSH session's
    environment.

  Without the token, `tabpfn` cells are recorded as "skipped," not an
  error — `tabnet`/`ft_transformer` and everything else still runs.
- `--skip-data` — if you're populating `data/` manually (e.g. scp'd over
  separately, or already present from a prior run).
- `--skip-smoke` — skip the self-verification step.

It's idempotent — re-running after a reboot or a `sync_to_ec2` won't redo
finished steps.

## 3. Kick off the run (with auto-restart)

Two ways to run it, both wrapping the same crash-resilient supervisor
(`deploy/run_forever.sh`):

**Quick / manual** (no root needed) — good for a first run you want to watch:

```bash
tmux new -s benchmark
./deploy/run_forever.sh
# Ctrl-b d to detach; tmux attach -t benchmark to come back
```

**Systemd** (recommended for an unattended multi-hour/day run) — also
resumes automatically if the *instance itself* reboots, e.g. a maintenance
reboot or a spot-instance interruption + relaunch:

```bash
sudo ./deploy/install_service.sh
systemctl status trialbench-benchmark
```

Either way, `run_forever.sh` passes any extra args straight through to
`python -m src.run_benchmark`, so you can stage the run the same way the
[README](../README.md) quickstart suggests:

```bash
./deploy/run_forever.sh --phases Phase1 --max-test-rows 500   # quick check
./deploy/run_forever.sh                                        # full grid
```

## How the restart mechanism works

Two layers, matching the two ways a long CPU run actually fails:

1. **Per-cell resume** (already built into `src/run_benchmark.py`): every
   task×phase×method×seed cell writes `results/runs/<stem>.json` atomically
   (write-then-`os.replace`) only once it fully completes. On any invocation,
   cells whose JSON already exists are skipped. So simply re-running
   `python -m src.run_benchmark` after *any* interruption — a killed process,
   a reboot, a manual Ctrl-C — picks up exactly where it left off. The
   leaderboard is rebuilt from whatever's on disk every 10 completed cells
   and again whenever the process exits (even via an exception), so
   `results/leaderboard.md` is never more than a few cells stale.

2. **Process-level auto-restart** (`deploy/run_forever.sh` +
   `deploy/trialbench-benchmark.service`): protects against the whole
   `python` process dying — OOM kill, a native-library segfault, a spot
   reclaim. `run_forever.sh` loops on `python -m src.run_benchmark`, checks
   whether `results/runs/` grew between attempts, and backs off (30s → 600s
   cap) between retries. It gives up after `MAX_RETRIES` (default 20)
   *consecutive* attempts with **no new progress** — a real, reproducible
   failure (bad config, missing dependency) won't crash-loop forever; check
   `logs/attempt_N.log` for the last failure. Any attempt that *does* make
   progress resets the counter. The systemd unit adds one more layer on top:
   if the instance reboots, `trialbench-benchmark.service` starts on boot
   and `run_forever.sh` resumes from disk exactly as above.

   A `results/.benchmark_complete` marker is written once the full grid
   finishes cleanly; `run_forever.sh` exits immediately if it finds one, so
   restarting the service post-completion is a no-op until you delete the
   marker (or pass `--force` to a manual run to recompute specific cells).

## Monitoring

```bash
tail -f logs/supervisor.log        # retry/backoff decisions
tail -f logs/attempt_<N>.log       # a specific attempt's run_benchmark output
tail -f results/leaderboard.md     # current standings, updated during the run
journalctl -u trialbench-benchmark -f   # if running under systemd
```

## 4. Pull results back down

Safe to run any time, including mid-run:

```bash
EC2_HOST=ec2-user@<instance-ip> EC2_KEY=~/.ssh/your-key.pem ./deploy/fetch_results.sh
```

```powershell
.\deploy\fetch_results.ps1 -EC2Host ec2-user@<instance-ip> -EC2Key ~\.ssh\your-key.pem
```

## Notes

- These scripts assume `rsync` for incremental syncs where available (bash
  side); they fall back to a `tar`-over-`ssh` stream (used by default on the
  PowerShell side, since Windows doesn't ship rsync) when it isn't.
- Nothing here manages the EC2 instance's lifecycle (launch/stop/terminate)
  or security group / key pair setup — that's on you or your existing IaC.
- If you're running this as a spot instance for cost, the two-layer restart
  above is exactly what makes that safe: a reclaim just means the next
  launch's systemd unit resumes the grid from `results/runs/`.
