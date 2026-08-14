"""Precondition P2 (newsletter-part2-test-plan.md): persist validation and
test predictions per fit, keyed by NCT id, instead of only the aggregate
metrics ``run_benchmark.py`` writes to ``results/runs/``.

Without this, any test that needs raw scores (blend weight fitting, paired
bootstraps on blend lift, novel-scaffold significance) has to refit every
method from scratch every time it's asked a new question about the same fit.

Layout: ``<results_dir>/predictions/<task>/<phase>/<method>_<seed>.parquet``,
one row per (nct_id, split) with columns ``nct_id, split, y_true, y_proba``.
Binary tasks only for now (P2 is written to unblock T13/T16-T19, all binary);
``y_proba`` there is the scalar P(y=1). Multiclass support (a per-class column
set) can be added when a multiclass test actually needs this.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def pred_path(results_dir: str, task: str, phase: str, method: str, seed: int) -> str:
    return os.path.join(results_dir, "predictions", task, phase, f"{method}_{seed}.parquet")


def has_predictions(results_dir: str, task: str, phase: str, method: str, seed: int) -> bool:
    return os.path.exists(pred_path(results_dir, task, phase, method, seed))


def save_predictions(results_dir: str, task: str, phase: str, method: str, seed: int,
                      valid_ids, valid_proba, valid_y, test_ids, test_proba, test_y) -> str:
    """Write valid+test predictions for one (task, phase, method, seed) fit.

    Atomic write (tmp + os.replace) so a kill mid-write never leaves a parquet
    file that looks done on resume, matching run_benchmark.py's own JSON
    write discipline.
    """
    out_path = pred_path(results_dir, task, phase, method, seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.concat([
        pd.DataFrame({
            "nct_id": np.asarray(valid_ids).astype(str),
            "split": "valid",
            "y_true": np.asarray(valid_y, dtype=float),
            "y_proba": np.asarray(valid_proba, dtype=float),
        }),
        pd.DataFrame({
            "nct_id": np.asarray(test_ids).astype(str),
            "split": "test",
            "y_true": np.asarray(test_y, dtype=float),
            "y_proba": np.asarray(test_proba, dtype=float),
        }),
    ], ignore_index=True)

    tmp_path = out_path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    return out_path


def load_predictions(results_dir: str, task: str, phase: str, method: str, seed: int,
                      split: str | None = None) -> pd.DataFrame:
    path = pred_path(results_dir, task, phase, method, seed)
    df = pd.read_parquet(path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df
