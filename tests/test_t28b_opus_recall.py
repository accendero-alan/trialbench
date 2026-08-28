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
        for key in ("test_id", "inputs", "preflight", "n_sampled", "primary_a_vs_b",
                   "disease_swap", "branch", "branch_reason", "meter", "git_sha", "wall_clock_secs"):
            assert key in artifact, f"missing top-level key {key!r}"
        assert artifact["test_id"] == "T28b"
        assert artifact["n_sampled"]["arm_a"] == 8
        assert artifact["n_sampled"]["arm_c"] == 4  # n_arm_c cap respected, not the spec default ~416
        assert artifact["branch"] in ("OUTCOME_RECALL_DEMONSTRATED", "DISTRIBUTION_SHIFT_NOT_RECALL",
                                      "NO_OUTCOME_RECALL_DETECTED", "OPUS_MORE_ROBUST_THAN_REFERENCE")
        # A uniform 50% answer regardless of trial content has no
        # discrimination on either arm -- both must read as "no drop"
        # (the null-input case), never a false OUTCOME_RECALL_DEMONSTRATED.
        assert artifact["branch"] == "NO_OUTCOME_RECALL_DETECTED", artifact["branch_reason"]
        assert artifact["meter"]["calls"] == fake_client.calls
        assert artifact["disease_swap"] is not None
        print("full pipeline (tiny sample) OK:", artifact["branch"], artifact["n_sampled"])


def test_decide_recall_demonstrated_only_when_opus_drops_and_reference_does_not():
    primary_drops = {"mean_delta": 0.3, "lo": 0.1, "hi": 0.5}
    reference_flat = {"mean_delta": 0.0, "lo": -0.05, "hi": 0.05}
    branch, reason = decide(primary_drops, reference_flat)
    assert branch == "OUTCOME_RECALL_DEMONSTRATED", (branch, reason)
    print("recall-demonstrated OK:", branch)


def test_decide_no_recall_when_neither_drops():
    flat = {"mean_delta": 0.0, "lo": -0.05, "hi": 0.05}
    branch, reason = decide(dict(flat), dict(flat))
    assert branch == "NO_OUTCOME_RECALL_DETECTED", (branch, reason)
    print("no-recall OK:", branch)


def test_decide_distribution_shift_when_both_drop():
    both_drop = {"mean_delta": 0.2, "lo": 0.05, "hi": 0.4}
    branch, reason = decide(dict(both_drop), dict(both_drop))
    assert branch == "DISTRIBUTION_SHIFT_NOT_RECALL", (branch, reason)
    print("distribution-shift OK:", branch)


def test_decide_opus_more_robust_when_only_reference_drops():
    primary_flat = {"mean_delta": 0.0, "lo": -0.05, "hi": 0.05}
    reference_drops = {"mean_delta": 0.2, "lo": 0.05, "hi": 0.4}
    branch, reason = decide(primary_flat, reference_drops)
    assert branch == "OPUS_MORE_ROBUST_THAN_REFERENCE", (branch, reason)
    print("opus-more-robust OK:", branch)


if __name__ == "__main__":
    if not (_SNAPSHOT_PRESENT and _TRIALBENCH_PRESENT):
        print(f"SKIPPED: needs both the AACT snapshot ({SNAPSHOT_DIR}/studies.txt) and TrialBench "
             "data (data/mortality-event-prediction/) present locally.")
    else:
        test_text_identity_preflight_confirms_shared_content()
        test_full_pipeline_tiny_sample()
    test_decide_recall_demonstrated_only_when_opus_drops_and_reference_does_not()
    test_decide_no_recall_when_neither_drops()
    test_decide_distribution_shift_when_both_drop()
    test_decide_opus_more_robust_when_only_reference_drops()
    print("t28b tests passed")
