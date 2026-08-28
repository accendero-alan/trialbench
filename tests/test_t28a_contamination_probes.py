"""T28a acceptance test (wave2-start-plan.md), run against a fake injected
Bedrock client and a fake AACT loader -- no live AWS, and no need for the
real 2.5GB AACT snapshot to exercise the registration-date join path.

Run:  python tests/test_t28a_contamination_probes.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.t28a_contamination_probes import (  # noqa: E402
    OUTCOME_RECALL_SHRINK_THRESHOLD,
    _clopper_pearson_ci,
    _decide_branch,
    per_task_outcome_discrimination,
    run,
)
from tests.test_smoke import _write_task  # noqa: E402

# amazon.nova-lite: a real priced entry in configs/bedrock_prices.yaml with a
# real (non-null) cutoff, so this exercises the AUROC-vs-cutoff code path
# rather than the "no cutoff" skip path.
MODEL = "amazon.nova-lite"


class _FakeConverseClient:
    """Returns content-aware canned text so every instrument's parser gets
    something plausible to score, without needing a real model."""

    def __init__(self):
        self.calls = 0

    def converse(self, modelId, messages, inferenceConfig):
        self.calls += 1
        prompt = messages[0]["content"][0]["text"]
        if "Yes or No" in prompt:
            text = "Yes"
        elif "single number" in prompt:
            text = "500"
        elif "next word" in prompt:
            text = "example"
        elif "missing column" in prompt:
            text = "enrollment"
        elif "official title" in prompt:
            text = "A Study of an Example Intervention"
        else:
            text = "the trial continued as expected under standard procedures for this arm"
        return {
            "output": {"message": {"content": [{"text": text}]}},
            "usage": {"inputTokens": 40, "outputTokens": 5},
        }


def _fake_aact_loader(all_old_dates: bool):
    """Ten synthetic NCT ids (matching tests.test_smoke._write_task's
    "NCT{i:08d}" convention) with registration dates either all before every
    model's cutoff (`all_old_dates=True`, forces a single-class label -> AUROC
    "not computable") or split across a real cutoff (mixed classes)."""
    ids = [f"NCT{i:08d}" for i in range(10)]
    dates = (["2015-01-01"] * 10) if all_old_dates else (["2015-01-01"] * 5 + ["2026-06-01"] * 5)

    def _load_table(name, snapshot_dir=None):
        assert name == "studies"
        return pd.DataFrame({"nct_id": ids, "study_first_posted_date": dates})
    return _load_table


def test_artifact_shape_and_auroc_not_computable_with_single_class():
    """--detectors=True (opt in): exercises the detector arm itself. The
    default (off) path is covered separately by
    test_detector_arm_off_by_default_records_a_reason."""
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, "mortality-event-prediction", "Phase1", n_train=20, n_test=10)
        _write_task(data_root, "serious-adverse-event-forecasting", "Phase2", n_train=20, n_test=10)

        fake_client = _FakeConverseClient()
        out_path = os.path.join(results_dir, "t28a_probe_gate.json")
        artifact = run(
            data_root=data_root, results_dir=results_dir, n_trials=6, models=[MODEL], seed=42,
            boto_client=fake_client, aact_loader=_fake_aact_loader(all_old_dates=True),
            out_path=out_path, run_detectors=True,
        )

        assert artifact["test_id"] == "T28a"
        for key in ("inputs", "n_trials_sampled", "per_model", "decision_rule", "verdict",
                   "git_sha", "wall_clock_secs"):
            assert key in artifact, f"missing top-level key {key!r}"

        assert artifact["n_trials_sampled"] == 6
        assert MODEL in artifact["per_model"]
        m = artifact["per_model"][MODEL]

        # Every sampled trial appears in per_trial.
        assert len(m["per_trial"]) == 6

        # All-old-dates -> single class on the pre/post-cutoff label ->
        # every detector AUROC is None ("not computable"), not a fabricated number.
        # (ngram_coverage/guided_prompting_delta are also None for a second,
        # fixture-specific reason: tests.test_smoke._make_x's synthetic rows
        # have no brief_summary/textblock column, so those two instruments'
        # prefix/suffix split has nothing to work with -- tabular_memorization
        # doesn't depend on that column, which is why it's still exercised.)
        for name, auroc in m["detector_aurocs"].items():
            assert auroc is None, f"{name} AUROC should be None (single class), got {auroc}"
        assert m["blind_baseline_auroc"] is None

        # Real dollars from configs/bedrock_prices.yaml, not zero/hardcoded.
        assert m["meter"]["dollars_realized"] > 0, m["meter"]
        assert m["meter"]["calls"] == fake_client.calls

        assert m["branch"] in ("SHRINK_TO_UNRECOGNIZED_STRATUM", "STRATIFY", "PROCEED_AS_DESIGNED")
        assert m["detector_arm_status"] == "computed"
        print("shape + not-computable OK:", m["branch"], m["meter"]["dollars_realized"])


def test_detector_arm_off_by_default_records_a_reason():
    """F4 (2026-08-28 decision): the detector arm is off by default for the
    gating run. Its aggregate fields must be None with an explicit reason
    recorded, not silently absent or a bare None a reader could mistake for
    "ran, found nothing"."""
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, "mortality-event-prediction", "Phase1", n_train=20, n_test=10)

        fake_client = _FakeConverseClient()
        out_path = os.path.join(results_dir, "t28a_probe_gate.json")
        artifact = run(
            data_root=data_root, results_dir=results_dir, n_trials=6, models=[MODEL], seed=42,
            boto_client=fake_client, aact_loader=_fake_aact_loader(all_old_dates=False),
            out_path=out_path,
        )
        m = artifact["per_model"][MODEL]
        assert m["detector_aurocs"] is None
        assert m["blind_baseline_auroc"] is None
        assert m["recognition_uninformative"] is None
        assert m["cross_instrument_agreement"] is None
        assert m["detector_arm_status"].startswith("disabled (--detectors not passed):")
        assert "docs/t28a_fixes_before_full_run.md" in m["detector_arm_status"]
        # The recall probes still ran -- dropping the detector arm must not
        # silently drop the two probes the gating decision actually needs.
        assert m["title_recall_rate"] is not None
        assert artifact["inputs"]["detector_arm_decision"]["this_run_used_detectors"] is False
        print("detector-arm-off-by-default OK:", m["detector_arm_status"][:60] + "...")


def test_auroc_computable_with_mixed_classes():
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, "mortality-event-prediction", "Phase1", n_train=20, n_test=10)

        fake_client = _FakeConverseClient()
        out_path = os.path.join(results_dir, "t28a_probe_gate.json")
        artifact = run(
            data_root=data_root, results_dir=results_dir, n_trials=8, models=[MODEL], seed=42,
            boto_client=fake_client, aact_loader=_fake_aact_loader(all_old_dates=False),
            out_path=out_path, run_detectors=True,
        )
        m = artifact["per_model"][MODEL]
        assert m["cutoff_note"] is None, m["cutoff_note"]  # nova-lite has a real cutoff
        computed = [v for v in m["detector_aurocs"].values() if v is not None]
        assert computed, f"expected at least one computable AUROC with mixed classes: {m['detector_aurocs']}"
        for auroc in computed:
            assert 0.0 <= auroc <= 1.0
        print("mixed-class AUROC OK:", m["detector_aurocs"])


def test_decide_branch_null_input_is_negative():
    """A4's house rule: feed the verdict function a null / no-effect input
    and the verdict must come back negative -- here, PROCEED_AS_DESIGNED
    with no signal claimed on either probe. Guards against a repeat of
    F1's bug (a threshold that fires on a model that merely answers)."""
    branch, reason = _decide_branch(
        title_hits=0, title_n=200, title_recall_rate=0.0,
        outcome_recall_rate=OUTCOME_RECALL_SHRINK_THRESHOLD + 0.315,  # base rate itself -- must NOT trigger SHRINK alone
        outcome_significant_tasks={},
    )
    assert branch == "PROCEED_AS_DESIGNED", (branch, reason)
    assert "no signal" in reason
    print("null-input branch OK:", branch)


def test_decide_branch_title_recall_significant_shrinks_or_stratifies():
    branch_high, _ = _decide_branch(
        title_hits=5, title_n=200, title_recall_rate=0.025,
        outcome_recall_rate=OUTCOME_RECALL_SHRINK_THRESHOLD + 0.01, outcome_significant_tasks={},
    )
    assert branch_high == "SHRINK_TO_UNRECOGNIZED_STRATUM", branch_high

    branch_low, _ = _decide_branch(
        title_hits=5, title_n=200, title_recall_rate=0.025,
        outcome_recall_rate=OUTCOME_RECALL_SHRINK_THRESHOLD - 0.01, outcome_significant_tasks={},
    )
    assert branch_low == "STRATIFY", branch_low
    print("title-significant branch OK:", branch_high, branch_low)


def test_decide_branch_outcome_significant_task_is_proceed_not_shrink():
    """F1's core fix: a real predictive signal (no title recall) must never
    read as contamination."""
    branch, reason = _decide_branch(
        title_hits=0, title_n=200, title_recall_rate=0.0,
        outcome_recall_rate=0.62,
        outcome_significant_tasks={
            "mortality_rate_yn": {"n": 34, "balanced_accuracy": 0.792,
                                  "fisher_exact_p": 0.002, "majority_class_rate": 0.6},
        },
    )
    assert branch == "PROCEED_AS_DESIGNED", branch
    assert "mortality_rate_yn" in reason and "0.792" in reason
    print("outcome-significant branch OK:", branch)


def test_clopper_pearson_ci_single_hit_is_thin_but_positive():
    lower, upper = _clopper_pearson_ci(1, 200)
    assert lower is not None and 0.0 < lower < 0.01
    assert upper is not None and upper < 0.05

    lower0, upper0 = _clopper_pearson_ci(0, 200)
    assert lower0 == 0.0
    assert upper0 is not None and upper0 < 0.02  # rule-of-three: ~3/200
    print("Clopper-Pearson CI OK:", (lower, upper), (lower0, upper0))


def test_per_task_outcome_discrimination_base_rate_invariant():
    """Regression guard for the Simpson's-paradox bug F2 fixes: a model
    that always answers the majority class must NOT show as discriminating
    once scored per task, even though pooled raw accuracy would look high."""
    per_trial = []
    # task A: 90% positive, model always says positive -> "accuracy" 0.9 but
    # balanced accuracy 0.5 (no discrimination).
    for i in range(20):
        label = 1 if i < 18 else 0
        per_trial.append({"task": "task_a", "outcome_true_label": label, "outcome_parsed_answer": 1})
    stats = per_task_outcome_discrimination(per_trial)
    assert stats["task_a"]["n"] == 20
    assert stats["task_a"]["balanced_accuracy"] == 0.5
    print("per-task balanced accuracy OK:", stats)


if __name__ == "__main__":
    test_artifact_shape_and_auroc_not_computable_with_single_class()
    test_detector_arm_off_by_default_records_a_reason()
    test_auroc_computable_with_mixed_classes()
    test_decide_branch_null_input_is_negative()
    test_decide_branch_title_recall_significant_shrinks_or_stratifies()
    test_decide_branch_outcome_significant_task_is_proceed_not_shrink()
    test_clopper_pearson_ci_single_hit_is_thin_but_positive()
    test_per_task_outcome_discrimination_base_rate_invariant()
    print("t28a contamination probe tests passed")
