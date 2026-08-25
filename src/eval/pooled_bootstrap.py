"""Pooled paired bootstrap over trials, for the disease-representation
campaign's standing rule 7 (disease-representation-test-plan.md): per-cell
deltas mostly won't clear ``delta_cell`` (T1's noise floor), so the primary
inference for every arm-vs-arm contrast pools trials across a task's cells
and resamples *trials*, not cells -- preserving the correlation between two
arms' predictions on the same trial, which is what makes a paired delta's CI
tighter than bootstrapping each arm independently would give.

Pooling choice, spelled out because the plan doesn't fix it: TrialBench's
test split is seed-independent (only train/valid vary by seed -- confirmed
in T21's covered-subset sub-analysis), so pooling *only* across a task's 4
phases would already give every trial its full weight once per seed's fit.
This module pools across phases AND seeds -- each (phase, seed) contributes
that phase's test trials once, using that seed's fit -- so the resulting CI
also captures cross-seed model variance, not just cross-phase trial
variance. Callers assemble the pooled arrays; this module only does the
resampling.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_METRIC_FNS = {
    "prauc": average_precision_score,
    "auroc": roc_auc_score,
}


def pool_predictions(rows: list) -> dict:
    """``rows``: a list of ``(y_true_array, proba_array)`` pairs, e.g. one
    per (phase, seed) cell-fit -> concatenated ``{"y_true", "proba"}``."""
    if not rows:
        return {"y_true": np.array([]), "proba": np.array([])}
    return {
        "y_true": np.concatenate([r[0] for r in rows]),
        "proba": np.concatenate([r[1] for r in rows]),
    }


def pooled_paired_bootstrap(y_true, proba_a, proba_b, metric: str = "prauc",
                             n_resamples: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """``y_true``/``proba_a``/``proba_b``: already-pooled, row-aligned arrays
    (the same trial/seed-fit at the same row index in all three -- an inner
    join on ``nct_id`` per (phase, seed) before pooling, if the two arms'
    predictions weren't fit on identical row sets). Returns
    ``{mean_a, mean_b, mean_delta, lo, hi, std, n_resamples_used, n_rows}``,
    delta = a - b, resampled by drawing rows with replacement so both arms
    see the identical resampled trial set on every draw. Resamples with only
    one class present are skipped (the metric is undefined), so
    ``n_resamples_used`` can be less than ``n_resamples`` on small/imbalanced
    pools -- report both, not just the CI.
    """
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)
    if not (len(y_true) == len(proba_a) == len(proba_b)):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, proba_a={len(proba_a)}, proba_b={len(proba_b)}")
    fn = _METRIC_FNS[metric]
    n = len(y_true)
    rng = np.random.default_rng(seed)

    deltas, a_vals, b_vals = [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        a = fn(yt, proba_a[idx])
        b = fn(yt, proba_b[idx])
        a_vals.append(a)
        b_vals.append(b)
        deltas.append(a - b)

    deltas_arr = np.asarray(deltas, dtype=float)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    has = len(deltas_arr) > 0
    return {
        "mean_a": float(np.mean(a_vals)) if has else float("nan"),
        "mean_b": float(np.mean(b_vals)) if has else float("nan"),
        "mean_delta": float(np.mean(deltas_arr)) if has else float("nan"),
        "lo": float(np.percentile(deltas_arr, lo_q)) if has else float("nan"),
        "hi": float(np.percentile(deltas_arr, hi_q)) if has else float("nan"),
        "std": float(np.std(deltas_arr)) if has else float("nan"),
        "n_resamples_used": int(len(deltas_arr)),
        "n_resamples_requested": n_resamples,
        "n_rows": n,
    }
