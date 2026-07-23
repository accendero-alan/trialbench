"""Aggregate results/runs/*.json into leaderboard.csv and leaderboard.md.

The leaderboard ranks methods by the headline metric (PR-AUC) averaged over the
run's phases, per task, and overall. Multi-seed cells are averaged first.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def _load(results_dir):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "runs", "*.json")):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # A stray partial file (e.g. a kill mid-write on an older run, before
            # run_benchmark.py wrote atomically) shouldn't take down the whole
            # leaderboard build.
            print(f"  warn: skipping unreadable run file {path}: {e}")
            continue
        if rec.get("status") != "ok":
            rows.append({"task": rec.get("task"), "phase": rec.get("phase"),
                         "method": rec.get("method"), "seed": rec.get("seed"),
                         "status": rec.get("status"), "headline_mean": np.nan})
            continue
        headline = rec.get("headline", "prauc")
        h = rec.get("bootstrap", {}).get(headline, {})
        row = {"task": rec["task"], "phase": rec["phase"], "method": rec["method"],
               "seed": rec["seed"], "status": "ok",
               "headline_metric": headline,
               "headline_mean": h.get("mean", np.nan),
               "headline_lo": h.get("lo", np.nan), "headline_hi": h.get("hi", np.nan),
               "fit_secs": rec.get("fit_secs")}
        for k, v in rec.get("bootstrap", {}).items():
            row[f"{k}_mean"] = v.get("mean", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build(results_dir):
    df = _load(results_dir)
    if df.empty:
        print("no runs found")
        return df
    ok = df[df["status"] == "ok"].copy()
    os.makedirs(results_dir, exist_ok=True)
    ok.to_csv(os.path.join(results_dir, "results_long.csv"), index=False)

    lines = ["# TrialBench classification leaderboard", ""]
    if ok.empty:
        lines.append("_No successful runs yet._")
        _write_md(results_dir, lines)
        return df

    # average over seeds -> (task, phase, method)
    tpm = ok.groupby(["task", "phase", "method"])["headline_mean"].mean().reset_index()

    # per-task table: methods × phases (headline = PR-AUC)
    for task in sorted(tpm["task"].unique()):
        sub = tpm[tpm["task"] == task]
        pivot = sub.pivot_table(index="method", columns="phase", values="headline_mean")
        pivot["mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("mean", ascending=False)
        lines.append(f"## {task}  (headline: PR-AUC)")
        lines.append("")
        lines.append(pivot.round(4).to_markdown())
        lines.append("")

    # overall ranking: mean headline across all task×phase
    overall = (tpm.groupby("method")["headline_mean"].mean()
               .sort_values(ascending=False).round(4))
    lines.append("## Overall (mean PR-AUC across all task×phase)")
    lines.append("")
    lines.append(overall.to_frame("mean_prauc").to_markdown())
    lines.append("")

    pivot_csv = tpm.pivot_table(index="method", columns=["task", "phase"], values="headline_mean")
    pivot_csv.to_csv(os.path.join(results_dir, "leaderboard.csv"))
    _write_md(results_dir, lines)
    return df


def _write_md(results_dir, lines):
    with open(os.path.join(results_dir, "leaderboard.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "results")
