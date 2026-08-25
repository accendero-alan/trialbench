"""T22 -- Bound the disease share per cell, from both directions.

disease-representation-test-plan.md (T22 to T29, continuing T21's campaign).
Tests whether Part 2's central claim -- "the most informative thing about a
trial is the disease it studies" -- holds up quantitatively, and whether the
disease *share* of predictable signal orders clinical (mortality, SAE) >
operational (dropout), as the whole downstream campaign assumes.

Four arms, all cells, 5 seeds:
  (a) disease-only, codes  -- T21's "codes" view (char3 rollup), reused from
      results_codes_only. Not necessarily complete (T21's own run may still
      be in progress) -- missing (task, phase, method, seed) points are
      dropped from that arm's mean, same as T1/T21's own handling.
  (b) disease-only, text   -- disease_text_only (P10), results_disease_text.
  (c) disease-blind, text  -- disease_blind (P10), results_disease_text.
  (d) full arms of record  -- T1's tabular arm + tfidf_logreg, both from
      results/extracted/trialbench/results (the T1 baseline).

Usage: `python -m experiments.t22_disease_share` (after arms (a)-(c) exist;
(a) partial is tolerated, (b)/(c) are a ~10-minute sweep -- see the plan's
Cost line for T22).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.data.loader import TASKS, load_task_phase
from src.eval.pooled_bootstrap import pool_predictions, pooled_paired_bootstrap
from src.eval.predictions import load_predictions

NONTEXT_METHODS = [
    "majority", "logreg_l2", "logreg_l1", "random_forest", "extra_trees",
    "hist_gbm", "knn", "svm_linear", "xgboost", "lightgbm", "catboost",
]
FULL_METHODS = NONTEXT_METHODS + ["tfidf_logreg"]
SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"

BASELINE_DIR = "results/extracted/trialbench/results"  # T1's tabular arm + tfidf_logreg (top-level, for load_predictions)
BASELINE_RUNS = BASELINE_DIR + "/runs"                  # same arm, run-JSON files (for _load_point)
ARM_A_DIR = "results_codes_only"                        # T21's "codes" (char3) view
ARM_BC_DIR = "results_disease_text"                      # disease_text_only, disease_blind
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
OUT_PATH = "results/experiments/t22_disease_share.json"

BINARY_TASKS = [t for t, (_, _, tt) in TASKS.items() if tt == "binary"]
ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]
BINARY_CELLS = [(t, p) for t, p in ALL_CELLS if t in BINARY_TASKS]
CLINICAL_TASKS = ["mortality_rate_yn", "serious_adverse_rate_yn"]
OPERATIONAL_TASK = "patient_dropout_rate_yn"


def _load_point(runs_dir, task, phase, method, seed):
    path = os.path.join(runs_dir, f"{task}__{phase}__{method}__seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    return rec["point"]["prauc"] if rec.get("status") == "ok" else None


def _mean_over_seeds(runs_dir, task, phase, method):
    vals = [_load_point(runs_dir, task, phase, method, s) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def _best_method(runs_dir, task, phase, methods):
    """(best_method_name, best_mean) by mean-over-seeds PR-AUC, an oracle
    argmax over test-set means -- descriptive, not a validation-based
    selection (standing rule 4's caveat noted, not resolved: T21's own
    gap_closure sub-analysis used the same convention)."""
    best_name, best_mean = None, float("-inf")
    for m in methods:
        mean, n = _mean_over_seeds(runs_dir, task, phase, m)
        if n and mean > best_mean:
            best_name, best_mean = m, mean
    return best_name, (best_mean if best_name else float("nan"))


def _pooled_trials(runs_dir, task, method, seeds=SEEDS, phases=("Phase1", "Phase2", "Phase3", "Phase4")):
    """Pool one method's test-split predictions across a task's phases and
    seeds (pooled_bootstrap's chosen convention -- see its docstring)."""
    rows = []
    for phase in phases:
        for seed in seeds:
            try:
                df = load_predictions(runs_dir, task, phase, method, seed, split="test")
            except FileNotFoundError:
                continue
            rows.append((df["y_true"].to_numpy(), df["y_proba"].to_numpy()))
    return pool_predictions(rows)


def _disease_share_bootstrap(task, majority_method, disease_dir, disease_method,
                              full_dir, full_method, n_resamples=1000, ci=0.95, seed=0):
    """Pooled bootstrap for disease_share = (disease_only - majority) /
    (full - majority), per task. Not a simple two-arm delta (pooled_bootstrap
    handles that case directly) -- this resamples once per draw and computes
    the compound ratio on each draw, so the reported CI reflects the ratio's
    own sampling distribution rather than three separately-bootstrapped
    endpoints combined after the fact.
    """
    maj = _pooled_trials(BASELINE_DIR, task, majority_method)
    dis = _pooled_trials(disease_dir, task, disease_method)
    full = _pooled_trials(full_dir, task, full_method)

    # Inner-join on an assumption: majority/disease/full pool the *same*
    # (phase, seed) test sets in the same phase order, so row i in each is
    # the same trial -- true here since every arm runs the identical T1 grid
    # and TrialBench's test split doesn't vary by seed within a phase. Guard
    # it rather than assume it silently.
    n = len(maj["y_true"])
    if not (len(dis["y_true"]) == len(full["y_true"]) == n) or not (
        np.array_equal(maj["y_true"], dis["y_true"]) and np.array_equal(maj["y_true"], full["y_true"])
    ):
        raise ValueError(f"{task}: majority/disease/full pooled label vectors don't align row-for-row; "
                          f"lens={len(maj['y_true'])},{len(dis['y_true'])},{len(full['y_true'])}")

    from sklearn.metrics import average_precision_score
    y = maj["y_true"]
    rng = np.random.default_rng(seed)
    shares, maj_v, dis_v, full_v = [], [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = average_precision_score(yt, maj["proba"][idx])
        d = average_precision_score(yt, dis["proba"][idx])
        f = average_precision_score(yt, full["proba"][idx])
        denom = f - m
        maj_v.append(m); dis_v.append(d); full_v.append(f)
        shares.append((d - m) / denom if denom else float("nan"))

    shares = np.asarray(shares, dtype=float)
    shares = shares[~np.isnan(shares)]
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    has = len(shares) > 0
    return {
        "mean_majority": float(np.mean(maj_v)) if maj_v else float("nan"),
        "mean_disease_only": float(np.mean(dis_v)) if dis_v else float("nan"),
        "mean_full": float(np.mean(full_v)) if full_v else float("nan"),
        "disease_share_mean": float(np.mean(shares)) if has else float("nan"),
        "disease_share_lo": float(np.percentile(shares, lo_q)) if has else float("nan"),
        "disease_share_hi": float(np.percentile(shares, hi_q)) if has else float("nan"),
        "disease_share_width": float(np.percentile(shares, hi_q) - np.percentile(shares, lo_q)) if has else float("nan"),
        "n_resamples_used": int(len(shares)), "n_rows": n,
        "disease_method": disease_method, "full_method": full_method,
    }


def main():
    with Timer() as t:
        # ---- per-cell descriptive table (all 20 cells, mean-based) --------
        per_cell = []
        for task, phase in ALL_CELLS:
            maj_mean, maj_n = _mean_over_seeds(BASELINE_RUNS, task, phase, "majority")
            a_best, a_mean = _best_method(ARM_A_DIR + "/runs", task, phase, NONTEXT_METHODS)
            b_mean, b_n = _mean_over_seeds(ARM_BC_DIR + "/runs", task, phase, "disease_text_only")
            c_mean, c_n = _mean_over_seeds(ARM_BC_DIR + "/runs", task, phase, "disease_blind")
            tfidf_mean, tfidf_n = _mean_over_seeds(BASELINE_RUNS, task, phase, "tfidf_logreg")
            full_best, full_mean = _best_method(BASELINE_RUNS, task, phase, FULL_METHODS)

            disease_only_mean = np.nanmax([a_mean, b_mean if b_n else np.nan])
            denom = full_mean - maj_mean
            disease_share = (disease_only_mean - maj_mean) / denom if denom else float("nan")
            blind_drop = tfidf_mean - c_mean if (tfidf_n and c_n) else float("nan")

            per_cell.append({
                "task": task, "phase": phase,
                "majority_mean": maj_mean,
                "arm_a_codes_best_method": a_best, "arm_a_codes_mean": a_mean,
                "arm_b_disease_text_only_mean": b_mean if b_n else None,
                "arm_c_disease_blind_mean": c_mean if c_n else None,
                "disease_only_mean_best_of_ab": float(disease_only_mean) if not np.isnan(disease_only_mean) else None,
                "full_best_method": full_best, "full_mean": full_mean,
                "tfidf_logreg_mean": tfidf_mean if tfidf_n else None,
                "disease_share": disease_share, "blind_drop": blind_drop,
                "arm_a_n_seeds": None,  # filled from _best_method's underlying call if needed
            })
            print(f"  {task}/{phase}: disease_share={disease_share:.3f} blind_drop={blind_drop if not np.isnan(blind_drop) else float('nan'):.3f}"
                  if not np.isnan(blind_drop) else f"  {task}/{phase}: disease_share={disease_share:.3f} blind_drop=nan", flush=True)

        pc_df = pd.DataFrame(per_cell)

        # ---- pooled per task (compound-ratio bootstrap) -------------------
        # "Best" method per task chosen once (not per-phase) by mean PR-AUC
        # pooled over that task's 4 phases -- avoids a different "best"
        # method per phase fragmenting the pooled trial set. Documented
        # simplification; the plan doesn't specify pooling mechanics for a
        # 3-arm ratio (see this module's _disease_share_bootstrap docstring).
        pooled_per_task = {}
        for task in BINARY_TASKS:
            task_rows = pc_df[pc_df["task"] == task]
            try:
                a_mode = task_rows["arm_a_codes_best_method"].mode()
                full_mode = task_rows["full_best_method"].mode()
                if full_mode.empty:
                    raise ValueError(f"{task}: no full-arm best method available (T1 baseline missing?)")
                full_method = full_mode.iat[0]
                use_a = (not a_mode.empty) and (
                    np.nanmean(task_rows["arm_a_codes_mean"])
                    >= np.nanmean(pd.to_numeric(task_rows["arm_b_disease_text_only_mean"], errors="coerce"))
                )
                disease_dir, disease_method = (ARM_A_DIR, a_mode.iat[0]) if use_a else (ARM_BC_DIR, "disease_text_only")
                res = _disease_share_bootstrap(
                    task, "majority", disease_dir, disease_method, BASELINE_DIR, full_method,
                )
            except (ValueError, FileNotFoundError, IndexError) as e:
                res = {"error": str(e)}
            pooled_per_task[task] = res
            print(f"  pooled {task}: {res}", flush=True)

        # blind_drop pooled per task (a simple two-arm delta -- the direct
        # pooled_paired_bootstrap case)
        blind_drop_pooled = {}
        for task in BINARY_TASKS:
            tfidf = _pooled_trials(BASELINE_DIR, task, "tfidf_logreg")
            blind = _pooled_trials(ARM_BC_DIR, task, "disease_blind")
            if len(tfidf["y_true"]) and len(tfidf["y_true"]) == len(blind["y_true"]) and \
               np.array_equal(tfidf["y_true"], blind["y_true"]):
                blind_drop_pooled[task] = pooled_paired_bootstrap(
                    tfidf["y_true"], tfidf["proba"], blind["proba"], metric="prauc",
                )
            else:
                blind_drop_pooled[task] = {"error": "row mismatch or missing predictions"}

        # ---- decision rule --------------------------------------------------
        def share_point(task):
            return pooled_per_task.get(task, {}).get("disease_share_mean", float("nan"))

        def share_width(task):
            return pooled_per_task.get(task, {}).get("disease_share_width", float("nan"))

        clinical_shares = [share_point(t) for t in CLINICAL_TASKS]
        dropout_share = share_point(OPERATIONAL_TASK)
        widths = [share_width(t) for t in CLINICAL_TASKS] + [share_width(OPERATIONAL_TASK)]
        max_width = float(np.nanmax(widths)) if any(not np.isnan(w) for w in widths) else float("nan")

        order_confirmed = (
            not np.isnan(dropout_share) and all(not np.isnan(s) for s in clinical_shares)
            and all((s - dropout_share) > max_width for s in clinical_shares)
        )
        blind_orders_same = all(
            blind_drop_pooled.get(t, {}).get("mean_delta", float("nan")) >
            blind_drop_pooled.get(OPERATIONAL_TASK, {}).get("mean_delta", float("-inf"))
            for t in CLINICAL_TASKS
        ) if "error" not in blind_drop_pooled.get(OPERATIONAL_TASK, {}) else False

        min_clinical_share = min([s for s in clinical_shares if not np.isnan(s)], default=float("nan"))

        if order_confirmed and blind_orders_same:
            verdict = (f"CONFIRMED: pooled disease_share on {CLINICAL_TASKS} ({clinical_shares}) exceeds "
                       f"{OPERATIONAL_TASK} ({dropout_share:.3f}) by more than the pooled CI width "
                       f"({max_width:.3f}), and blind_drop orders the same way.")
        elif not np.isnan(dropout_share) and not np.isnan(min_clinical_share) \
                and abs(min_clinical_share - dropout_share) <= max_width:
            verdict = ("SOFTENED to 'disease dominates everywhere': dropout's disease_share is not "
                       "distinguishable from the clinical tasks' at this pooled CI width.")
        else:
            verdict = "INCONCLUSIVE on this split (see pooled_per_task and blind_drop_pooled for the raw numbers)."

        if not np.isnan(min_clinical_share) and min_clinical_share < (1.0 / 3.0):
            verdict += (f" Also: disease-only recovers only {min_clinical_share:.1%} of full skill on the "
                        f"weakest clinical task (< 1/3) -- premise weakened per the plan's own threshold.")

    artifact = {
        "test_id": "T22",
        "claim_at_stake": "the most informative thing about a trial is the disease it studies "
                           "(disease share orders clinical > mixed > operational)",
        "inputs": {"nontext_methods": NONTEXT_METHODS, "full_methods": FULL_METHODS, "seeds": SEEDS,
                   "baseline_dir": BASELINE_DIR, "arm_a_dir": ARM_A_DIR, "arm_bc_dir": ARM_BC_DIR},
        "per_cell": per_cell,
        "pooled_per_task_disease_share": pooled_per_task,
        "pooled_per_task_blind_drop": blind_drop_pooled,
        "decision_rule": {
            "primary": "CONFIRMED if pooled disease_share on mortality+SAE exceeds dropout's by more "
                        "than the pooled CI width, and blind_drop orders the same way. Else SOFTENED "
                        "to 'disease dominates everywhere' if shares aren't distinguishable. If "
                        "disease-only recovers <1/3 of full skill on a clinical task, the premise is "
                        "flagged as weakened regardless of the ordering verdict.",
        },
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}")
    print(artifact["verdict"])
    return artifact


if __name__ == "__main__":
    main()
