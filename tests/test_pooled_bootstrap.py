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

from src.eval.pooled_bootstrap import (  # noqa: E402
    cluster_bootstrap_indices,
    diff_in_diff_bootstrap,
    one_sample_cluster_bootstrap,
    pooled_paired_bootstrap,
    two_sample_cluster_bootstrap,
)


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


def _synthetic_arm_pair(n, method_good, ref_good, seed):
    """One arm, two scorers (method + ref) on the SAME rows -- the shape
    diff_in_diff_bootstrap actually needs (unlike two_sample_cluster_bootstrap's
    two independent arms)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    def score(good):
        return np.clip(y * 0.5 + rng.normal(0, 0.15, n) + (0.25 if good else 0.0), 0, 1)
    nct_id = np.array([f"N{seed}_{i:04d}" for i in range(n)])
    return nct_id, y, score(method_good), score(ref_good)


def test_diff_in_diff_detects_when_method_drops_more_than_reference():
    """docs/t28b_reanalysis_plan.md R2: testing each arm's drop for
    significance separately cannot tell you whether the two drops differ
    from each other (Gelman-Stern). diff_in_diff_bootstrap must actually
    answer that question directly."""
    nct_a, y_a, method_a, ref_a = _synthetic_arm_pair(250, method_good=True, ref_good=True, seed=1)
    nct_b, y_b, method_b, ref_b = _synthetic_arm_pair(250, method_good=False, ref_good=True, seed=2)
    r = diff_in_diff_bootstrap(nct_a, y_a, method_a, ref_a, nct_b, y_b, method_b, ref_b,
                               metric="balanced_accuracy", n_resamples=500, seed=1)
    assert r["lo"] > 0, r  # method's drop significantly exceeds the reference's
    assert r["mean_delta_method"] > r["mean_delta_ref"]
    print("diff-in-diff real-difference OK:", r["mean_diff"], "CI=[", r["lo"], r["hi"], "]")


def test_diff_in_diff_null_when_both_drop_comparably():
    """The actual motivating failure mode: both a real method and a
    reference dropping by a similar amount must NOT read as
    'method-specific recall' just because the method's own drop happened
    to individually clear a significance threshold."""
    nct_a, y_a, method_a, ref_a = _synthetic_arm_pair(250, method_good=True, ref_good=True, seed=3)
    nct_b, y_b, method_b, ref_b = _synthetic_arm_pair(250, method_good=False, ref_good=False, seed=4)
    r = diff_in_diff_bootstrap(nct_a, y_a, method_a, ref_a, nct_b, y_b, method_b, ref_b,
                               metric="balanced_accuracy", n_resamples=500, seed=1)
    assert r["lo"] <= 0 <= r["hi"], r
    print("diff-in-diff null-when-comparable OK:", r["mean_diff"], "CI=[", r["lo"], r["hi"], "]")


def test_diff_in_diff_reports_observed_not_assumed_correlation():
    nct_a, y_a, method_a, ref_a = _synthetic_arm_pair(250, method_good=True, ref_good=True, seed=5)
    nct_b, y_b, method_b, ref_b = _synthetic_arm_pair(250, method_good=True, ref_good=True, seed=6)
    r = diff_in_diff_bootstrap(nct_a, y_a, method_a, ref_a, nct_b, y_b, method_b, ref_b,
                               metric="balanced_accuracy", n_resamples=300, seed=1)
    assert r["rho"] is not None and -1.0 <= r["rho"] <= 1.0
    print("diff-in-diff rho OK:", r["rho"])


def test_one_sample_cluster_bootstrap_ci_contains_point_estimate():
    nct_id, y, proba = _synthetic_arm(250, discriminates=True, seed=7)
    r = one_sample_cluster_bootstrap(nct_id, y, proba, metric="balanced_accuracy", n_resamples=500, seed=1)
    assert r["lo"] <= r["point"] <= r["hi"], r
    assert r["n_rows"] == 250 and r["n_clusters"] == 250
    print("one-sample CI OK:", r["point"], "CI=[", r["lo"], r["hi"], "]")


def test_cluster_bootstrap_indices_empty_input_yields_empty_draws_not_a_crash():
    """Found live on T28b-L0's disease-swap arm: a fully-empty arm (every
    candidate had no eligible cross-chapter donor -- real on AACT's
    icdcode-sparse Arm B) used to crash inside np.concatenate([]) (\"need
    at least one array to concatenate\") instead of yielding the empty
    resample that draw legitimately represents."""
    draws = list(cluster_bootstrap_indices(np.array([]), n_resamples=10, seed=0))
    assert len(draws) == 10
    assert all(len(d) == 0 for d in draws)
    print("empty-input cluster_bootstrap_indices OK:", len(draws), "draws, all empty")


def test_one_sample_and_paired_bootstrap_degrade_to_nan_on_empty_arm():
    """The two real callers of cluster_bootstrap_indices must not crash
    when handed a genuinely empty arm -- they already have a `has = len(
    ...) > 0` fallback to NaN for \"zero usable resamples\"; this just
    confirms the empty-input case reaches that fallback instead of
    raising, now that cluster_bootstrap_indices itself doesn't."""
    empty = np.array([])
    r1 = one_sample_cluster_bootstrap(empty, empty, empty, metric="auroc", n_resamples=10, seed=0)
    assert r1["n_resamples_used"] == 0 and np.isnan(r1["lo"]) and np.isnan(r1["hi"])
    r2 = pooled_paired_bootstrap(empty, empty, empty, empty, metric="auroc", n_resamples=10, seed=0)
    assert r2["n_resamples_used"] == 0 and np.isnan(r2["lo"]) and np.isnan(r2["hi"])
    print("empty-arm bootstrap degradation OK: both report n_resamples_used=0, CI=[nan, nan]")


if __name__ == "__main__":
    test_detects_real_gap_between_independent_arms()
    test_no_gap_when_arms_are_equally_uninformative()
    test_single_class_draws_are_skipped_not_fabricated()
    test_diff_in_diff_detects_when_method_drops_more_than_reference()
    test_diff_in_diff_null_when_both_drop_comparably()
    test_diff_in_diff_reports_observed_not_assumed_correlation()
    test_one_sample_cluster_bootstrap_ci_contains_point_estimate()
    test_cluster_bootstrap_indices_empty_input_yields_empty_draws_not_a_crash()
    test_one_sample_and_paired_bootstrap_degrade_to_nan_on_empty_arm()
    print("pooled_bootstrap tests passed")
