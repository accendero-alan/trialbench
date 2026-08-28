"""src/eval/pooled_bootstrap.py's two_sample_cluster_bootstrap() -- the
independent-groups counterpart pooled_paired_bootstrap has no equivalent
for, needed by docs/t28b_opus_recall_spec.md's Arm A vs Arm B contrast
(disjoint nct_id sets, no row to pair on).

Run:  python tests/test_pooled_bootstrap.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.pooled_bootstrap import two_sample_cluster_bootstrap  # noqa: E402


def _synthetic_arm(n, discriminates, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    if discriminates:
        proba = np.clip(y * 0.6 + rng.normal(0, 0.15, n) + 0.2, 0, 1)
    else:
        proba = rng.uniform(0, 1, n)
    nct_id = np.array([f"NCT{seed}{i:04d}" for i in range(n)])
    return nct_id, y, proba


def test_detects_real_gap_between_independent_arms():
    idx_x, yx, px = _synthetic_arm(200, discriminates=True, seed=1)
    idx_y, yy, py = _synthetic_arm(200, discriminates=False, seed=2)
    r = two_sample_cluster_bootstrap(idx_x, yx, px, idx_y, yy, py,
                                     metric="balanced_accuracy", n_resamples=500, seed=1)
    assert r["mean_x"] > 0.6, r
    assert abs(r["mean_y"] - 0.5) < 0.1, r
    assert r["lo"] > 0, "CI should exclude 0 given a clear real gap"
    assert r["n_rows_x"] == 200 and r["n_rows_y"] == 200
    print("real-gap detection OK:", {k: r[k] for k in ("mean_x", "mean_y", "lo", "hi")})


def test_no_gap_when_arms_are_equally_uninformative():
    """A → B where nothing actually changed (same generating process, both
    null) must not manufacture a spurious CI excluding 0 -- the T28b
    decision table's "no drop detected" reading depends on this."""
    idx_x, yx, px = _synthetic_arm(200, discriminates=False, seed=3)
    idx_y, yy, py = _synthetic_arm(200, discriminates=False, seed=4)
    r = two_sample_cluster_bootstrap(idx_x, yx, px, idx_y, yy, py,
                                     metric="balanced_accuracy", n_resamples=500, seed=1)
    assert r["lo"] <= 0 <= r["hi"], f"expected CI to contain 0 for two null arms, got {r}"
    print("no-spurious-gap OK:", {k: r[k] for k in ("mean_x", "mean_y", "lo", "hi")})


def test_single_class_draws_are_skipped_not_fabricated():
    """A tiny, heavily imbalanced arm can draw an all-one-class resample --
    that draw must be skipped, not silently scored with an undefined
    metric."""
    idx_x = np.array([f"X{i}" for i in range(5)])
    yx = np.array([1, 1, 1, 1, 0])  # one single negative -- easy to draw an all-1 resample
    px = np.array([0.9, 0.8, 0.7, 0.6, 0.4])
    idx_y = np.array([f"Y{i}" for i in range(20)])
    yy = np.array([0, 1] * 10)
    py = np.random.default_rng(0).uniform(0, 1, 20)
    r = two_sample_cluster_bootstrap(idx_x, yx, px, idx_y, yy, py, n_resamples=200, seed=5)
    assert r["n_resamples_used"] <= r["n_resamples_requested"]
    assert r["n_resamples_used"] > 0, "expected at least some usable draws"
    print("single-class-skip OK:", r["n_resamples_used"], "/", r["n_resamples_requested"])


if __name__ == "__main__":
    test_detects_real_gap_between_independent_arms()
    test_no_gap_when_arms_are_equally_uninformative()
    test_single_class_draws_are_skipped_not_fabricated()
    print("pooled_bootstrap tests passed")
