"""T20 -- The city target-encoding leak.

newsletter-part2-test-plan.md, T20. `location/facility/address/city__te` is
flagged as possible encoding leakage in the blended cells -- the same
diagnostic shape that caught `brief_summary/textblock` in T8: a near-unique
categorical, smoothed target-encoded, separating classes in training and
collapsing toward the global mean at test as most test cities are unseen.

Per cell (16 binary cells): report train cardinality, rows-per-level, the
fraction of test rows whose city is unseen in train (falls back to the
global mean), and the collapse signature (std of the encoded column on train
vs. on test -- if test collapses toward the global mean, its std should be
much smaller). Then re-fit the 12 Tier A methods with the raw column dropped
before featurization, 5 seeds, paired against the current (post-T8-repair)
on-disk baseline.

Usage: `python -m experiments.t20_city_encoding`
"""
from __future__ import annotations

import json

import numpy as np

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import TabularFeaturizer
from src.data.loader import load_task_phase
from src.eval import metrics as M
from src.methods.registry import get as get_method

SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"
CITY_COL = "location/facility/address/city"
METHODS = [
    "majority", "logreg_l2", "logreg_l1", "random_forest", "extra_trees",
    "hist_gbm", "knn", "svm_linear", "xgboost", "lightgbm", "catboost",
]
BINARY_TASKS = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn", "patient_dropout_rate_yn"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
RUNS_DIR = "results/extracted/trialbench/results/runs"
OUT_PATH = "results/experiments/t20_city_encoding.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
CELLS = [(task, phase) for task in BINARY_TASKS for phase in PHASES]


def _load_baseline(task, phase, method, seed):
    try:
        with open(f"{RUNS_DIR}/{task}__{phase}__{method}__seed{seed}.json") as f:
            rec = json.load(f)
    except FileNotFoundError:
        return None
    return rec["point"]["prauc"] if rec.get("status") == "ok" else None


def _diagnose(td):
    fz = TabularFeaturizer(task_type=td.task_type).fit(td.X_train, td.y_train)
    if CITY_COL not in fz.categorical_:
        return None
    train_vals = td.X_train[CITY_COL].astype("object").where(td.X_train[CITY_COL].notna(), "__nan__").astype(str)
    test_vals = td.X_test[CITY_COL].astype("object").where(td.X_test[CITY_COL].notna(), "__nan__").astype(str)
    train_levels = set(train_vals.unique())
    enc = fz.cat_encoders_[CITY_COL]
    train_encoded = train_vals.map(enc).fillna(fz.global_mean_).to_numpy(dtype=float)
    test_encoded = test_vals.map(enc).fillna(fz.global_mean_).to_numpy(dtype=float)
    frac_test_unseen = float((~test_vals.isin(train_levels)).mean())
    return {
        "train_cardinality": len(train_levels),
        "n_train_rows": len(train_vals),
        "mean_rows_per_level": len(train_vals) / max(len(train_levels), 1),
        "frac_test_rows_city_unseen_in_train": frac_test_unseen,
        "std_encoded_train": float(np.std(train_encoded)),
        "std_encoded_test": float(np.std(test_encoded)),
        "global_mean": fz.global_mean_,
    }


def _drop_city(X):
    return X.drop(columns=[CITY_COL], errors="ignore")


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}

    per_cell = []
    with Timer() as t:
        for task, phase in CELLS:
            td0 = load_task_phase(DATA_ROOT, task, phase, seed=42)
            diag = _diagnose(td0)

            per_method_gain = {}
            for method in METHODS:
                paired_gains = []
                for seed in SEEDS:
                    baseline = _load_baseline(task, phase, method, seed)
                    if baseline is None:
                        continue
                    td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
                    fz = TabularFeaturizer(task_type=td.task_type)
                    Ztr = fz.fit_transform(_drop_city(td.X_train), td.y_train)
                    Zva, Zte = fz.transform(_drop_city(td.X_valid)), fz.transform(_drop_city(td.X_test))
                    Cls = get_method(method)
                    m = Cls(task_type=td.task_type, num_classes=td.num_classes, seed=seed)
                    m.fit(Ztr, td.y_train, Zva, td.y_valid)
                    proba = m.predict_proba(Zte)
                    dropped_prauc = M.compute(td.y_test, proba, td.task_type, td.num_classes)["prauc"]
                    paired_gains.append(dropped_prauc - baseline)
                if paired_gains:
                    per_method_gain[method] = float(np.mean(paired_gains))

            delta = delta_cells.get((task, phase), float("nan"))
            mean_gain = float(np.mean(list(per_method_gain.values()))) if per_method_gain else float("nan")
            exceeds = bool(abs(mean_gain) > delta) if not np.isnan(delta) and per_method_gain else None
            per_cell.append({
                "task": task, "phase": phase, "delta_cell": delta,
                "diagnostic": diag, "per_method_gain_from_dropping": per_method_gain,
                "mean_gain_from_dropping": mean_gain, "gain_exceeds_delta_cell": exceeds,
            })
            print(f"  {task}/{phase}: cardinality={diag['train_cardinality'] if diag else 'N/A'} "
                  f"frac_unseen={diag['frac_test_rows_city_unseen_in_train'] if diag else float('nan'):.3f} "
                  f"mean_gain={mean_gain:+.4f} delta={delta:.4f} exceeds={exceeds}", flush=True)

    n_above = sum(1 for c in per_cell if c["gain_exceeds_delta_cell"] is True)
    if n_above >= 5:
        verdict = (f"Every tabular baseline in Part 1 and Part 2 must be restated without the "
                    f"city column and blend lifts recomputed: mean gain from dropping exceeds "
                    f"delta_cell on {n_above}/16 cells.")
    else:
        verdict = (f"The concern is closed in one sentence: dropping the city column changes "
                    f"PR-AUC by more than delta_cell on only {n_above}/16 cells.")

    artifact = {
        "test_id": "T20",
        "claim_at_stake": "the tabular baselines in the blended cells are inflated by a city-encoding leak",
        "inputs": {"seeds": SEEDS, "methods": METHODS, "city_col": CITY_COL},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "n_cells_gain_above_delta_cell": n_above,
        "decision_rule": "If dropping the city column changes tabular PR-AUC by more than "
                          "delta_cell in >=5/16 cells, every tabular baseline and blend lift is "
                          "restated without it. Otherwise, closed in one sentence.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}: {n_above}/16 cells with gain > delta_cell")
    print(verdict)
    return artifact


if __name__ == "__main__":
    main()
