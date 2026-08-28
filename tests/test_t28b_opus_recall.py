"""T28b acceptance test (docs/t28b_opus_recall_spec.md), run against a fake
injected Bedrock client. Needs the real TrialBench data and AACT snapshot
present locally (no synthetic fixture -- T28b's sampling reads real trial
content, unlike T28a's fixture-friendly design), so this skips cleanly if
either is absent rather than failing the suite.

Run:  python tests/test_t28b_opus_recall.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.t28b_opus_recall import (  # noqa: E402
    DECISION_METRIC,
    DISEASE_SWAP_MOVE_THRESHOLD,
    decide,
    preflight_text_identity,
    run,
)
from src.data.aact import SNAPSHOT_DIR  # noqa: E402

_SNAPSHOT_PRESENT = os.path.exists(os.path.join(SNAPSHOT_DIR, "studies.txt"))
_TRIALBENCH_PRESENT = os.path.exists(os.path.join("data", "mortality-event-prediction"))


class _FakeConverseClient:
    """Always answers the same verbalized probability -- enough to exercise
    every code path (parsing, scoring, bootstrap, disease-swap join) without
    needing a real model or any particular prediction quality."""

    def __init__(self, probability=50):
        self.calls = 0
        self.probability = probability

    def converse(self, modelId, messages, inferenceConfig):
        self.calls += 1
        return {
            "output": {"message": {"content": [{"text": f'{{"probability": {self.probability}}}'}]}},
            "usage": {"inputTokens": 40, "outputTokens": 5},
        }


def test_text_identity_preflight_confirms_shared_content():
    """The spec's own stop condition: if TrialBench and AACT's summary
    text differ systematically, the A->B contrast reads a reconstruction
    difference, not recall. Confirmed live (session notes): a naive
    byte-for-byte compare shows 0/50 matching purely on formatting noise
    (whitespace, AACT's "~" paragraph breaks, escaped brackets, bullet
    markers) -- content-normalized, it should be the large majority."""
    r = preflight_text_identity(n_check=50)
    assert r["n_checked"] > 0
    assert r["n_identical"] / r["n_checked"] > 0.85, (
        f"only {r['n_identical']}/{r['n_checked']} matched after normalization -- "
        f"investigate: {r['example_mismatches']}"
    )
    print("text-identity preflight OK:", r["n_identical"], "/", r["n_checked"])


def test_full_pipeline_tiny_sample():
    with tempfile.TemporaryDirectory() as results_dir:
        fake_client = _FakeConverseClient()
        out_path = os.path.join(results_dir, "t28b_test.json")
        artifact = run(
            data_root="data", results_dir=results_dir, n_arm_a=8, n_arm_b=8, n_arm_c=4, n_swap=4,
            seed=42, boto_client=fake_client, out_path=out_path, n_resamples=100,
        )
        for key in ("test_id", "inputs", "preflight", "n_sampled", "primary_a_vs_b", "diff_in_diff",
                   "per_endpoint_arm_a", "per_trial", "disease_swap", "branch", "branch_reason",
                   "meter", "git_sha", "wall_clock_secs"):
            assert key in artifact, f"missing top-level key {key!r}"
        assert artifact["test_id"] == "T28b"
        assert artifact["n_sampled"]["arm_a"] == 8
        assert artifact["n_sampled"]["arm_c"] == 4  # n_arm_c cap respected, not the spec default ~416
        assert artifact["branch"] in ("OUTCOME_RECALL_DEMONSTRATED", "INCONCLUSIVE",
                                      "NO_RECALL_FINDING_CALIBRATION_ARTIFACT")
        # A uniform 50% answer regardless of trial content has zero
        # discrimination on every arm -- flat AUROC everywhere must read
        # as R1's calibration-artifact/no-signal branch, never a false
        # OUTCOME_RECALL_DEMONSTRATED.
        assert artifact["branch"] == "NO_RECALL_FINDING_CALIBRATION_ARTIFACT", artifact["branch_reason"]
        assert artifact["meter"]["calls"] == fake_client.calls
        assert artifact["disease_swap"] is not None
        assert len(artifact["per_trial"]["opus_arm_a"]) == 8
        assert set(REPORTED_METRICS_FOR_TEST) <= set(artifact["primary_a_vs_b"]["opus"])
        print("full pipeline (tiny sample) OK:", artifact["branch"], artifact["n_sampled"])


REPORTED_METRICS_FOR_TEST = ("auroc", "prauc", "balanced_accuracy")


def test_decide_recall_demonstrated_needs_both_r1_and_r2():
    """R1: Opus's own DECISION_METRIC must drop significantly. R2: that
    drop must significantly exceed the reference's -- not just two
    independently-significant deltas (the Gelman-Stern fallacy the
    reanalysis fixed)."""
    primary_opus = {DECISION_METRIC: {"mean_delta": 0.3, "lo": 0.1, "hi": 0.5}}
    primary_ref = {DECISION_METRIC: {"mean_delta": 0.0, "lo": -0.05, "hi": 0.05}}
    diff_in_diff = {"mean_diff": 0.25, "lo": 0.08, "hi": 0.42, "rho": 0.1}
    branch, reason = decide(primary_opus, primary_ref, diff_in_diff)
    assert branch == "OUTCOME_RECALL_DEMONSTRATED", (branch, reason)
    print("recall-demonstrated OK:", branch)


def test_decide_calibration_artifact_when_opus_auroc_flat():
    """R1 fails (Opus's AUROC doesn't drop) -> stop, regardless of what R2
    would have said. This is the case the original balanced-accuracy-only
    decide() could not detect: a thresholding artifact masquerading as a
    real drop."""
    primary_opus = {DECISION_METRIC: {"mean_delta": 0.0, "lo": -0.05, "hi": 0.05}}
    primary_ref = {DECISION_METRIC: {"mean_delta": 0.0, "lo": -0.05, "hi": 0.05}}
    diff_in_diff = {"mean_diff": 0.0, "lo": -0.05, "hi": 0.05, "rho": 0.0}
    branch, reason = decide(primary_opus, primary_ref, diff_in_diff)
    assert branch == "NO_RECALL_FINDING_CALIBRATION_ARTIFACT", (branch, reason)
    print("calibration-artifact OK:", branch)


def test_decide_inconclusive_when_r1_passes_but_r2_does_not():
    """The motivating failure mode this reanalysis fixed (shape matches
    T28b's real balanced-accuracy result, illustratively, not asserting
    these are verified AUROC numbers): Opus's own drop clears its CI, the
    reference's drop is smaller and does not clear its own CI, but the
    DIFFERENCE between the two (not each one's own significance) doesn't
    clear zero -- must read as INCONCLUSIVE, not
    OUTCOME_RECALL_DEMONSTRATED, which is exactly what the original
    two-independent-tests decide() got wrong on this shape."""
    primary_opus = {DECISION_METRIC: {"mean_delta": 0.055, "lo": 0.015, "hi": 0.095}}
    primary_ref = {DECISION_METRIC: {"mean_delta": 0.021, "lo": -0.028, "hi": 0.069}}
    diff_in_diff = {"mean_diff": 0.034, "lo": -0.029, "hi": 0.097, "rho": 0.0}
    branch, reason = decide(primary_opus, primary_ref, diff_in_diff)
    assert branch == "INCONCLUSIVE", (branch, reason)
    print("inconclusive OK:", branch)


if __name__ == "__main__":
    if not (_SNAPSHOT_PRESENT and _TRIALBENCH_PRESENT):
        print(f"SKIPPED: needs both the AACT snapshot ({SNAPSHOT_DIR}/studies.txt) and TrialBench "
             "data (data/mortality-event-prediction/) present locally.")
    else:
        test_text_identity_preflight_confirms_shared_content()
        test_full_pipeline_tiny_sample()
    test_decide_recall_demonstrated_needs_both_r1_and_r2()
    test_decide_calibration_artifact_when_opus_auroc_flat()
    test_decide_inconclusive_when_r1_passes_but_r2_does_not()
    print("t28b tests passed")
