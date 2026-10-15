"""T28a for Jev, title recognition re-run with era matching.

t28a_jev.py's title recognition cleared its pre-registered rule (diff
+0.035, p 1.8e-8), but post hoc the effect scaled with the NCT-number gap
between the real id and its control id (near tercile +0.012, p 0.23; far
+0.060). NCT numbers run in registration order, so a model matching a
title's era to the id's era passes that test without recalling anything.
Its control ids were random sampled trials (median gap 1.2M numbers), and
its decoys were random TrialBench titles.

This run removes era as a route:
- Pool: every TrialBench trial with a title, train/valid/test across all
  binary tasks and phases, minus the 200 sampled trials (54,674 trials).
- For each sampled trial, the 5 pool trials nearest in NCT number: the
  nearest is the control id, the next 4 supply the decoy titles. Median
  gap to the 5th nearest is ~200 numbers, so all six ids are near-
  contemporaneous.
- Same as before otherwise: the same 200 trials (T28a sample, seed 42),
  5 cyclic rotations per question, score = mean p(true title); the control
  asks the identical rotated options under the control id.

Decision rule, fixed before the run (also recorded on newsletter-pipeline
TASK-006 before any call):
- Primary: one-sided paired Wilcoxon signed-rank on p_true(real) -
  p_true(control), alpha 0.01. p < 0.01 and mean diff > 0 ->
  RECOGNITION_SHOWN: Jev identifies TrialBench trials from the id alone
  beyond era; its TrialBench outcome figures are not quoted as reading
  until a T28b-style arm separates the two. Otherwise ->
  NO_RECOGNITION_DETECTED: the first run's effect is attributed to era,
  no identity-level recall at the detectable size. This is weak evidence:
  it does not exclude recognising a fully described trial (T28a's known
  gap), so outcome discrimination stays unattributed either way.
- Detectable size: the first run's diff SD (0.087) gives a minimum
  detectable mean diff of ~0.02 at 80% power, one-sided alpha 0.01, n 200.
  A null rules out an average effect of that size, not a smaller one.
- Reported, not decisive: top-1 hits real vs control (McNemar exact,
  since the arms are paired); Spearman of diff vs |NCT gap| (should be
  ~0; if still significant, era is not fully controlled and the artifact
  says so); real p_true vs chance 0.2; the first run's five strongest
  trials, individually.

Usage:
    python -m experiments.t28a_jev_era
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

from experiments import t28a_jev as base
from experiments._common import git_sha
from experiments.t28a_contamination_probes import BINARY_TASKS, PHASES, _pool_trials
from src.data.loader import load_task_phase

OUT_PATH = "results/experiments/t28a_jev_era.json"
FIRST_RUN_PATH = "results/experiments/t28a_jev.json"
ALPHA = 0.01
N_NEAR = 5  # nearest = control id, next 4 = decoys


def _num(nct_id: str) -> int:
    return int(re.sub(r"\D", "", nct_id))


def _title_pool(data_root: str, seed: int, exclude: set) -> pd.DataFrame:
    frames = []
    for task in BINARY_TASKS:
        for phase in PHASES:
            try:
                td = load_task_phase(data_root, task, phase, seed=seed)
            except FileNotFoundError:
                continue
            for X in (td.X_train, td.X_valid, td.X_test):
                if X is not None and "brief_title" in X:
                    frames.append(pd.DataFrame({"nct_id": X.index.astype(str),
                                                "brief_title": X["brief_title"].values}))
    pool = pd.concat(frames).dropna().drop_duplicates("nct_id")
    pool = pool[~pool["nct_id"].isin(exclude)].copy()
    pool["num"] = pool["nct_id"].map(_num)
    return pool.sort_values("num").reset_index(drop=True)


def _nearest(pool: pd.DataFrame, nct_id: str, true_title: str) -> pd.DataFrame:
    gap = (pool["num"] - _num(nct_id)).abs()
    near = pool.assign(gap=gap).sort_values(["gap", "num"])
    near = near[near["brief_title"].str.strip().str.lower() != str(true_title).strip().lower()]
    near = near.drop_duplicates("brief_title")
    return near.head(N_NEAR)


def _mcnemar_exact(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    return float(binomtest(a_only, n, 0.5, alternative="greater").pvalue) if n else 1.0


def run(data_root="data", n_trials=200, seed=42, workers=base.WORKERS):
    base._load_cache()
    sample = _pool_trials(data_root, n_trials, seed)
    pool = _title_pool(data_root, seed, set(sample["nct_id"]))

    jobs = []
    for _, row in sample.iterrows():
        near = _nearest(pool, row["nct_id"], row["brief_title"])
        control_id = near.iloc[0]["nct_id"]
        decoys = near.iloc[1:N_NEAR]
        jobs.append((row, [row["brief_title"]] + decoys["brief_title"].tolist(), control_id,
                     decoys["nct_id"].tolist(), int(near["gap"].max())))

    def work(job):
        row, options, control_id, _, _ = job
        return base._title_block(row["nct_id"], options), base._title_block(control_id, options)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(work, jobs))

    rows, calls = [], []
    for (row, options, control_id, decoy_ids, max_gap), (real, ctrl) in zip(jobs, results):
        calls += [r["call"] for r in real["rotations"] + ctrl["rotations"]]
        rows.append({
            "nct_id": row["nct_id"], "task": row["task"], "title_options": options,
            "decoy_nct_ids": decoy_ids, "control_nct_id": control_id,
            "control_gap": abs(_num(control_id) - _num(row["nct_id"])), "max_gap_in_set": max_gap,
            "real_p_true": real["p_true"], "real_top1_hit": real["top1_hit"],
            "real_mean_p_by_option": real["mean_p_by_option"],
            "ctrl_p_true": ctrl["p_true"], "ctrl_top1_hit": ctrl["top1_hit"],
            "ctrl_mean_p_by_option": ctrl["mean_p_by_option"],
            "real_choice_slots": [r["choice_slot"] for r in real["rotations"]],
            "ctrl_choice_slots": [r["choice_slot"] for r in ctrl["rotations"]],
            "n_parsed": real["n_parsed"] + ctrl["n_parsed"],
        })

    brng = np.random.default_rng(seed + 11)
    paired = [r for r in rows if r["real_p_true"] is not None and r["ctrl_p_true"] is not None]
    real_p = np.array([r["real_p_true"] for r in paired]); ctrl_p = np.array([r["ctrl_p_true"] for r in paired])
    diff = real_p - ctrl_p
    w_p = float(wilcoxon(diff, alternative="greater").pvalue) if np.any(diff != 0) else 1.0
    real_only = sum(r["real_top1_hit"] and not r["ctrl_top1_hit"] for r in paired)
    ctrl_only = sum(r["ctrl_top1_hit"] and not r["real_top1_hit"] for r in paired)
    gaps = np.array([r["control_gap"] for r in paired])
    rho = spearmanr(gaps, diff)
    all_slots = [s for r in rows for s in r["real_choice_slots"] + r["ctrl_choice_slots"] if s]

    first = {}
    if os.path.exists(FIRST_RUN_PATH):
        prev = json.load(open(FIRST_RUN_PATH, encoding="utf-8"))["per_trial"]
        top = sorted(prev, key=lambda p: -(p["real_p_true"] - p["ctrl_p_true"]))[:5]
        now = {r["nct_id"]: r for r in rows}
        first = {p["nct_id"]: {"first_real": p["real_p_true"], "first_ctrl": p["ctrl_p_true"],
                               "era_real": now[p["nct_id"]]["real_p_true"],
                               "era_ctrl": now[p["nct_id"]]["ctrl_p_true"]} for p in top}

    shown = bool(w_p < ALPHA and diff.mean() > 0)
    title = {
        "n_paired": len(paired), "chance": 0.2,
        "real_mean_p_true": float(real_p.mean()), "real_ci_95": base._boot_ci(real_p, brng),
        "ctrl_mean_p_true": float(ctrl_p.mean()), "ctrl_ci_95": base._boot_ci(ctrl_p, brng),
        "mean_diff": float(diff.mean()), "mean_diff_ci_95": base._boot_ci(diff, brng),
        "diff_sd": float(diff.std(ddof=1)),
        "wilcoxon_p_one_sided": w_p, "alpha": ALPHA,
        "real_vs_chance_wilcoxon_p": float(wilcoxon(real_p - 0.2, alternative="greater").pvalue),
        "real_top1_hits": int(sum(r["real_top1_hit"] for r in paired)),
        "ctrl_top1_hits": int(sum(r["ctrl_top1_hit"] for r in paired)),
        "mcnemar_real_only": int(real_only), "mcnemar_ctrl_only": int(ctrl_only),
        "mcnemar_exact_p_one_sided": _mcnemar_exact(real_only, ctrl_only),
        "control_gap_median": int(np.median(gaps)), "control_gap_max": int(gaps.max()),
        "max_gap_in_set_median": int(np.median([r["max_gap_in_set"] for r in paired])),
        "spearman_diff_vs_gap": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
        "slot_choice_share": {L: round(all_slots.count(L) / len(all_slots), 4) for L in base.LETTERS},
        "first_run_strongest_trials": first,
    }
    usage = [base._usage(c) for c in calls]
    meter = {
        "calls": len(calls), "cache_hits": sum(bool(c.get("cache_hit")) for c in calls),
        "http_status_counts": {str(k): sum(c["status"] == k for c in calls)
                               for k in sorted({c["status"] for c in calls}, key=str)},
        "input_tokens": int(sum(u[0] for u in usage)), "output_tokens": int(sum(u[1] for u in usage)),
        "dollars_realized": round(sum(u[2] for u in usage), 6),
        "rotation_parse_rate": float(np.mean([r["n_parsed"] for r in rows]) / 10),
    }
    artifact = {
        "test_id": "t28a_jev_era", "model": base.MODEL, "endpoint": base.URL,
        "n_trials": n_trials, "seed": seed, "git_sha": git_sha(), "pool_size": len(pool),
        "claim_at_stake": "Whether Jev's title recognition in t28a_jev survives when era cannot favour the "
                          "true title: decoys and control id drawn from the trials nearest in NCT number.",
        "decision_rule": __doc__.split("Decision rule, fixed before the run")[1].split("Usage:")[0].strip(),
        "title_recognition": title,
        "branch": "RECOGNITION_SHOWN" if shown else "NO_RECOGNITION_DETECTED",
        "meter": meter, "per_trial": rows,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1, default=str)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=base.WORKERS)
    a = ap.parse_args()
    art = run(a.data_root, a.n_trials, a.seed, a.workers)
    print(json.dumps({k: art[k] for k in ["pool_size", "title_recognition", "branch", "meter"]},
                     indent=1, default=str))
