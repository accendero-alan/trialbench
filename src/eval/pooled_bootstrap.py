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

Resampling unit (wave1-preflight-review.md A1): because the test split is
seed-independent, pooling across (phase, seed) puts every trial into the
pooled array once per seed -- five (near-)identical copies of the same
trial, not five independent trials. Resampling *rows* i.i.d. (the original
implementation) treats those five correlated copies as independent draws,
which understates the true variance: measured on T22's SAE ``blind_drop``,
row resampling gave a CI 2.17x narrower than resampling by trial, enough to
flip a CI that should contain zero into one that excludes it. The fix is a
*cluster* bootstrap: resample distinct ``nct_id``s with replacement and carry
every row for a drawn id (i.e. every seed's copy of that trial) as one unit.
``cluster_bootstrap_indices`` is the one place that logic lives; every pooled
test in this campaign (this module's ``pooled_paired_bootstrap`` and T22's
custom compound-ratio resampling) draws its resample indices from it.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

_METRIC_FNS = {
    "prauc": average_precision_score,
    "auroc": roc_auc_score,
    # Thresholds the verbalized 0-1 probability at 0.5 -- balanced_accuracy_score
    # needs discrete predictions, not a continuous score, unlike prauc/auroc above.
    "balanced_accuracy": lambda y_true, proba: balanced_accuracy_score(y_true, (np.asarray(proba) >= 0.5).astype(int)),
}


def pool_predictions(rows: list) -> dict:
    """``rows``: a list of ``(nct_id_array, y_true_array, proba_array)``
    triples, e.g. one per (phase, seed) cell-fit -> concatenated
    ``{"nct_id", "y_true", "proba"}``. ``nct_id`` is required (not optional)
    so every caller pools it alongside the arrays it resamples -- see A1
    above; ``load_predictions`` already returns it, so this is a column
    rename away, not a new join."""
    if not rows:
        return {"nct_id": np.array([]), "y_true": np.array([]), "proba": np.array([])}
    return {
        "nct_id": np.concatenate([r[0] for r in rows]),
        "y_true": np.concatenate([r[1] for r in rows]),
        "proba": np.concatenate([r[2] for r in rows]),
    }


def cluster_bootstrap_indices(nct_id, n_resamples: int, seed: int = 0) -> Iterator[np.ndarray]:
    """Yield ``n_resamples`` row-index arrays, each a cluster-bootstrap draw
    over ``nct_id``: resample the distinct ids with replacement, and for
    every drawn id include *all* of its rows (every seed's copy of that
    trial) rather than drawing individual rows. A drawn index array's length
    varies resample to resample (it's the sum of the drawn ids' row counts,
    not always ``len(nct_id)``) -- that's expected; every metric function
    used on it only cares about alignment between the arrays it's applied to,
    not a fixed length.
    """
    nct_id = np.asarray(nct_id)
    order = np.argsort(nct_id, kind="stable")
    sorted_ids = nct_id[order]
    unique_ids, start_idx, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    groups = [order[s:s + c] for s, c in zip(start_idx, counts)]
    n_clusters = len(unique_ids)
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        yield np.concatenate([groups[d] for d in drawn])


def pooled_paired_bootstrap(nct_id, y_true, proba_a, proba_b, metric: str = "prauc",
                             n_resamples: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """``nct_id``/``y_true``/``proba_a``/``proba_b``: already-pooled,
    row-aligned arrays (the same trial/seed-fit at the same row index in all
    four -- an inner join on ``nct_id`` per (phase, seed) before pooling, if
    the two arms' predictions weren't fit on identical row sets). Returns
    ``{mean_a, mean_b, mean_delta, lo, hi, std, n_resamples_used, n_rows,
    n_clusters}``, delta = a - b, resampled by cluster-bootstrapping
    ``nct_id`` (see module docstring / ``cluster_bootstrap_indices``) so both
    arms see the identical resampled trial set -- including every seed's
    copy of each resampled trial -- on every draw. Resamples with only one
    class present are skipped (the metric is undefined), so
    ``n_resamples_used`` can be less than ``n_resamples`` on small/imbalanced
    pools -- report both, not just the CI.
    """
    nct_id = np.asarray(nct_id)
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)
    if not (len(nct_id) == len(y_true) == len(proba_a) == len(proba_b)):
        raise ValueError(f"length mismatch: nct_id={len(nct_id)}, y_true={len(y_true)}, "
                          f"proba_a={len(proba_a)}, proba_b={len(proba_b)}")
    fn = _METRIC_FNS[metric]
    n_clusters = len(np.unique(nct_id))

    deltas, a_vals, b_vals = [], [], []
    for idx in cluster_bootstrap_indices(nct_id, n_resamples, seed=seed):
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
        "n_rows": int(len(y_true)),
        "n_clusters": int(n_clusters),
    }


def two_sample_cluster_bootstrap(nct_id_x, y_true_x, proba_x, nct_id_y, y_true_y, proba_y,
                                 metric: str = "balanced_accuracy", n_resamples: int = 1000,
                                 ci: float = 0.95, seed: int = 0) -> dict:
    """Independent-groups counterpart to ``pooled_paired_bootstrap``, for two
    arms drawn from DISJOINT trial pools with no ``nct_id`` to pair on --
    e.g. ``docs/t28b_opus_recall_spec.md``'s Arm A vs Arm B, partitioned by a
    results-posting-date cutoff so by construction no trial can appear in
    both. Resamples each arm's ``nct_id``s independently
    (``cluster_bootstrap_indices``, one generator per arm; ``y``'s seed is
    offset by 1 from ``x``'s so the two draw streams aren't accidentally
    identical when the two arms happen to share a size) and compares draw
    ``i`` of X against draw ``i`` of Y.

    This is NOT the same statistical object ``pooled_paired_bootstrap``
    returns: there, both arms see the identical resampled trial set on every
    draw, so the delta's variance only reflects the *contrast* between two
    methods on the same rows. Here the two draws are independent, so
    ``mean_delta``'s CI reflects both arms' own sampling uncertainty added
    together -- correctly wider for the same n, because there is no
    shared-row correlation left to exploit when the rows themselves differ.

    Returns ``{mean_x, mean_y, mean_delta, lo, hi, std, n_resamples_used,
    n_resamples_requested, n_rows_x, n_rows_y, n_clusters_x, n_clusters_y}``,
    delta = x - y. A resample pair where either arm's drawn ``y_true`` is
    single-class is skipped (the metric is undefined on that side), so
    ``n_resamples_used`` can be less than requested on small/imbalanced
    arms -- report both, not just the CI, same discipline as the paired
    function above.
    """
    nct_id_x, nct_id_y = np.asarray(nct_id_x), np.asarray(nct_id_y)
    y_true_x, y_true_y = np.asarray(y_true_x), np.asarray(y_true_y)
    proba_x, proba_y = np.asarray(proba_x, dtype=float), np.asarray(proba_y, dtype=float)
    for name, a, b in (("x", nct_id_x, y_true_x), ("x", nct_id_x, proba_x),
                       ("y", nct_id_y, y_true_y), ("y", nct_id_y, proba_y)):
        if len(a) != len(b):
            raise ValueError(f"length mismatch within arm {name!r}: {len(a)} vs {len(b)}")
    fn = _METRIC_FNS[metric]

    x_gen = cluster_bootstrap_indices(nct_id_x, n_resamples, seed=seed)
    y_gen = cluster_bootstrap_indices(nct_id_y, n_resamples, seed=seed + 1)
    x_vals, y_vals, deltas = [], [], []
    for idx_x, idx_y in zip(x_gen, y_gen):
        ytx, yty = y_true_x[idx_x], y_true_y[idx_y]
        if len(np.unique(ytx)) < 2 or len(np.unique(yty)) < 2:
            continue
        vx, vy = fn(ytx, proba_x[idx_x]), fn(yty, proba_y[idx_y])
        x_vals.append(vx)
        y_vals.append(vy)
        deltas.append(vx - vy)

    deltas_arr = np.asarray(deltas, dtype=float)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    has = len(deltas_arr) > 0
    return {
        "mean_x": float(np.mean(x_vals)) if has else float("nan"),
        "mean_y": float(np.mean(y_vals)) if has else float("nan"),
        "mean_delta": float(np.mean(deltas_arr)) if has else float("nan"),
        "lo": float(np.percentile(deltas_arr, lo_q)) if has else float("nan"),
        "hi": float(np.percentile(deltas_arr, hi_q)) if has else float("nan"),
        "std": float(np.std(deltas_arr)) if has else float("nan"),
        "n_resamples_used": int(len(deltas_arr)),
        "n_resamples_requested": n_resamples,
        "n_rows_x": int(len(y_true_x)), "n_rows_y": int(len(y_true_y)),
        "n_clusters_x": int(len(np.unique(nct_id_x))), "n_clusters_y": int(len(np.unique(nct_id_y))),
    }
