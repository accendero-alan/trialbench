"""Load a TrialBench classification task at a given phase.

Data layout (per task folder, per phase)::

    <data_root>/<folder>/<Phase>/train_x.csv
    <data_root>/<folder>/<Phase>/train_y.csv
    <data_root>/<folder>/<Phase>/test_x.csv
    <data_root>/<folder>/<Phase>/test_y.csv

*_x.csv is indexed by NCT id with ~58 multimodal feature columns.
*_y.csv is indexed by NCT id and carries the label column(s). For the binary
event tasks the label is ``Y/N``; ``outcome`` and ``failure_reason`` use their
own named target columns. We resolve the label by trying a per-task candidate
list, falling back to the last column.

A validation split is carved from train (20%, seed 42) to mirror the repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# task name -> (folder, label-column candidates, task_type)
TASKS = {
    "outcome": (
        "trial-approval-forecasting",
        ["trial_approval_prediction", "Y/N", "outcome", "label"],
        "binary",
    ),
    "mortality_rate_yn": ("mortality-event-prediction", ["Y/N"], "binary"),
    "serious_adverse_rate_yn": ("serious-adverse-event-forecasting", ["Y/N"], "binary"),
    "patient_dropout_rate_yn": ("patient-dropout-event-forecasting", ["Y/N"], "binary"),
    "failure_reason": (
        "trial-failure-reason-identification",
        ["trial_failure_reason_prediction", "label", "Y/N"],
        "multiclass",
    ),
}


@dataclass
class TaskData:
    task: str
    phase: str
    task_type: str          # "binary" | "multiclass"
    num_classes: int
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_valid: pd.DataFrame
    y_valid: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    class_names: list        # original label values, index-aligned to encoded ints


def _read_xy(folder_path: str, split: str):
    x = pd.read_csv(os.path.join(folder_path, f"{split}_x.csv"), index_col=0, low_memory=False)
    y = pd.read_csv(os.path.join(folder_path, f"{split}_y.csv"), index_col=0, low_memory=False)
    # align on index
    common = x.index.intersection(y.index)
    return x.loc[common], y.loc[common]


def _pick_label(y_df: pd.DataFrame, candidates) -> pd.Series:
    for c in candidates:
        if c in y_df.columns:
            return y_df[c]
    return y_df.iloc[:, -1]


def load_task_phase(data_root: str, task: str, phase: str,
                    valid_size: float = 0.2, seed: int = 42,
                    max_train_rows=None, max_test_rows=None) -> TaskData:
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}'. Known: {sorted(TASKS)}")
    folder, label_candidates, task_type = TASKS[task]
    folder_path = os.path.join(data_root, folder, phase)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"missing data dir: {folder_path}\n"
            f"Run `python -m src.data.download` or point data_root at your TrialBench data."
        )

    Xtr_full, ytr_df = _read_xy(folder_path, "train")
    Xte, yte_df = _read_xy(folder_path, "test")

    ytr_raw = _pick_label(ytr_df, label_candidates)
    yte_raw = _pick_label(yte_df, label_candidates)

    # Encode labels to contiguous ints; class_names[i] is the original value for int i.
    if task_type == "binary":
        ytr_enc = pd.to_numeric(ytr_raw, errors="coerce").fillna(0).astype(int).clip(0, 1).values
        yte_enc = pd.to_numeric(yte_raw, errors="coerce").fillna(0).astype(int).clip(0, 1).values
        class_names = [0, 1]
        num_classes = 2
    else:
        cats = pd.Categorical(ytr_raw.astype(str))
        class_names = list(cats.categories)
        mapping = {c: i for i, c in enumerate(class_names)}
        ytr_enc = cats.codes.astype(int)
        yte_enc = np.array([mapping.get(str(v), -1) for v in yte_raw], dtype=int)
        # drop test rows whose class is unseen in train
        keep = yte_enc >= 0
        Xte, yte_enc = Xte.loc[keep], yte_enc[keep]
        num_classes = len(class_names)

    if max_train_rows:
        Xtr_full = Xtr_full.iloc[:max_train_rows]
        ytr_enc = ytr_enc[:max_train_rows]
    if max_test_rows:
        Xte = Xte.iloc[:max_test_rows]
        yte_enc = yte_enc[:max_test_rows]

    strat = ytr_enc if len(np.unique(ytr_enc)) > 1 else None
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr_full, ytr_enc, test_size=valid_size, random_state=seed, shuffle=True, stratify=strat
    )

    return TaskData(
        task=task, phase=phase, task_type=task_type, num_classes=num_classes,
        X_train=Xtr, y_train=np.asarray(ytr), X_valid=Xva, y_valid=np.asarray(yva),
        X_test=Xte, y_test=np.asarray(yte_enc), class_names=class_names,
    )
