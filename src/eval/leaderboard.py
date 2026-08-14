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


def shared_cell_set(tpm: pd.DataFrame, methods) -> list:
    """(task, phase) cells where every method in ``methods`` has a completed
    result in the long-format frame ``tpm`` (columns: task, phase, method).

    This is the fix for P4 (newsletter-part2-test-plan.md): a mean/ranking
    over cells is only valid when every compared method ran every cell in
    the set, per the plan's standing rule #3. Test scripts (T1, T8, ...)
    should call this directly with an explicit method list rather than
    reimplementing the pivot/dropna dance.
    """
    methods = list(methods)
    sub = tpm[tpm["method"].isin(methods)]
    counts = sub.groupby(["task", "phase"])["method"].nunique()
    full = counts[counts == len(set(methods))]
    return list(full.index)


def coverage(tpm: pd.DataFrame) -> pd.DataFrame:
    """method × task table of how many distinct phases each method completed,
    against how many phases exist for that task at all -- so partial
    coverage (ft_transformer, tabnet, clinical_embeddings today) is visible
    instead of silently blended into a skipna mean."""
    total_phases = tpm.groupby("task")["phase"].nunique()
    done = tpm.groupby(["method", "task"])["phase"].nunique().unstack(fill_value=0)
    out = done.astype(str) + "/" + done.columns.map(total_phases).astype(str)
    return out


def _ranked_mean(pivot: pd.DataFrame) -> pd.Series:
    """Mean over only the columns (phases) every row (method) in ``pivot``
    has a value for -- i.e. the shared cell set for exactly the methods
    present in this table. Columns not fully covered are dropped from the
    mean, not skipna-averaged over (the bug this replaces: the old
    ``pivot.mean(axis=1)`` silently skipped NaN, blending partial-coverage
    methods in on an unequal footing -- see report.md)."""
    shared_cols = pivot.columns[pivot.notna().all(axis=0)]
    if len(shared_cols) == 0:
        return pd.Series(np.nan, index=pivot.index)
    return pivot[shared_cols].mean(axis=1)


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

    cov = coverage(tpm)
    lines.append("## Coverage (phases completed / phases in task)")
    lines.append("")
    lines.append(cov.to_markdown())
    lines.append("")
    cov.to_csv(os.path.join(results_dir, "coverage.csv"))

    # per-task table: methods × phases (headline = PR-AUC)
    for task in sorted(tpm["task"].unique()):
        sub = tpm[tpm["task"] == task]
        pivot = sub.pivot_table(index="method", columns="phase", values="headline_mean")
        n_shared = int(pivot.notna().all(axis=0).sum())
        pivot["mean_shared"] = _ranked_mean(pivot)
        pivot = pivot.sort_values("mean_shared", ascending=False)
        lines.append(f"## {task}  (headline: PR-AUC; ranked by mean_shared, "
                      f"{n_shared}/{pivot.shape[1] - 1} phases shared by every method shown)")
        lines.append("")
        lines.append(pivot.round(4).to_markdown())
        lines.append("")

    # overall ranking: restricted to the (task, phase) cells every currently
    # -successful method ran -- NOT a skipna mean across mismatched coverage.
    all_methods = sorted(tpm["method"].unique())
    shared = set(shared_cell_set(tpm, all_methods))
    tpm_shared = tpm[tpm.apply(lambda r: (r["task"], r["phase"]) in shared, axis=1)]
    overall = (tpm_shared.groupby("method")["headline_mean"].mean()
               .sort_values(ascending=False).round(4))
    lines.append(f"## Overall (mean PR-AUC over the {len(shared)} cells shared "
                  f"by all {len(all_methods)} methods with a completed run)")
    lines.append("")
    if overall.empty:
        lines.append("_No cell is shared by every method with a completed run "
                      "-- see the coverage table above; compare specific "
                      "method subsets with `shared_cell_set()` instead._")
    else:
        lines.append(overall.to_frame("mean_prauc_shared").to_markdown())
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
