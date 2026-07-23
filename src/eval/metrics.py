"""Classification metrics + bootstrap, matching TrialBench's definitions.

Binary (num_classes == 2, headline = PR-AUC):
    AUROC, PR-AUC (average precision), F1, precision, recall, accuracy, specificity
Multiclass (failure_reason):
    macro AUROC (one-vs-rest), macro F1, PR-AUC (macro OvR), accuracy

Metrics that are undefined on a given (bootstrap) sample — e.g. AUROC when only
one class is present — return NaN and are ignored in the aggregate.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BINARY_METRICS = ["auroc", "prauc", "f1", "precision", "recall", "accuracy", "specificity"]
MULTICLASS_METRICS = ["auroc", "prauc", "f1", "accuracy"]
HEADLINE = "prauc"


def _safe(fn):
    try:
        return float(fn())
    except Exception:
        return float("nan")


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auroc": _safe(lambda: roc_auc_score(y_true, y_score)),
        "prauc": _safe(lambda: average_precision_score(y_true, y_score)),
        "f1": _safe(lambda: f1_score(y_true, y_pred, zero_division=0)),
        "precision": _safe(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": _safe(lambda: accuracy_score(y_true, y_pred)),
        "specificity": specificity,
    }


def multiclass_metrics(y_true, y_proba, num_classes: int) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = np.argmax(y_proba, axis=1)
    labels = list(range(num_classes))

    def _auroc():
        return roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr", labels=labels)

    def _prauc():
        aps = []
        for c in labels:
            yc = (y_true == c).astype(int)
            if yc.sum() == 0:
                continue
            aps.append(average_precision_score(yc, y_proba[:, c]))
        return float(np.mean(aps)) if aps else float("nan")

    return {
        "auroc": _safe(_auroc),
        "prauc": _safe(_prauc),
        "f1": _safe(lambda: f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "accuracy": _safe(lambda: accuracy_score(y_true, y_pred)),
    }


def compute(y_true, y_pred_proba, task_type: str, num_classes: int) -> dict:
    if task_type == "binary":
        return binary_metrics(y_true, y_pred_proba)
    return multiclass_metrics(y_true, y_pred_proba, num_classes)


def bootstrap(y_true, y_pred_proba, task_type: str, num_classes: int,
              n_resamples: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """Return {metric: {mean, lo, hi, std}} over bootstrap resamples of the test set."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    keys = BINARY_METRICS if task_type == "binary" else MULTICLASS_METRICS
    samples = {k: [] for k in keys}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = compute(y_true[idx], y_pred_proba[idx], task_type, num_classes)
        for k in keys:
            samples[k].append(m[k])

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[k] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "std": float("nan")}
        else:
            out[k] = {
                "mean": float(np.mean(arr)),
                "lo": float(np.percentile(arr, lo_q)),
                "hi": float(np.percentile(arr, hi_q)),
                "std": float(np.std(arr)),
            }
    return out
