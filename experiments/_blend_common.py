"""Shared reconstruction helpers for T17/T18/T19, all of which read the
predictions T16 persisted (results/predictions/...) rather than refitting.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from src.eval.predictions import load_predictions

BINARY_TASKS = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn", "patient_dropout_rate_yn"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
SEEDS = [42, 7, 123, 2024, 5]
TABULAR_METHODS = ["logreg_l1", "logreg_l2", "svm_linear", "knn", "random_forest",
                   "extra_trees", "hist_gbm", "xgboost", "lightgbm", "catboost"]
TEXT_METHOD = "tfidf_logreg"
RESULTS_DIR = "results"
CELLS = [(task, phase) for task in BINARY_TASKS for phase in PHASES]


def _rank_pct(s):
    r = rankdata(np.asarray(s, dtype=float), method="average")
    return (r - 1.0) / max(len(r) - 1, 1)


def load_cell_seed(task, phase, seed):
    """Reconstruct (y_test, tab_te_rank, txt_te_rank, tab_te_scores, txt_te_scores)
    for one (task, phase, seed) from T16's persisted predictions."""
    te_ranks, te_scores = [], []
    y_test = None
    for method in TABULAR_METHODS:
        df = load_predictions(RESULTS_DIR, task, phase, method, seed, split="test")
        if y_test is None:
            y_test = df["y_true"].to_numpy()
        te_scores.append(df["y_proba"].to_numpy())
        te_ranks.append(_rank_pct(df["y_proba"].to_numpy()))
    tab_te_rank = np.mean(te_ranks, axis=0)

    txt_df = load_predictions(RESULTS_DIR, task, phase, TEXT_METHOD, seed, split="test")
    txt_te_scores = txt_df["y_proba"].to_numpy()
    txt_te_rank = _rank_pct(txt_te_scores)

    return y_test, tab_te_rank, txt_te_rank
