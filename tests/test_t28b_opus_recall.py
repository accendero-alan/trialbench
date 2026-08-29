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
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.t28b_opus_recall as t28b_mod  # noqa: E402
from experiments.t28b_opus_recall import (  # noqa: E402
    DECISION_METRIC,
    DISEASE_SWAP_MOVE_THRESHOLD,
    assert_no_disease_leak,
    decide,
    elicit_row,
    preflight_text_identity,
    render_arm_with_disease_override,
    run,
    run_l0_null_check,
    swap_disease,
)
from src.bedrock.client import BedrockClient  # noqa: E402
from src.bedrock.meter import Meter  # noqa: E402
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
            data_root="data", results_dir=results_dir, n_arm_a=8, n_arm_b=8, n_arm_c=4,
            seed=42, boto_client=fake_client, out_path=out_path, n_resamples=100,
        )
        for key in ("test_id", "inputs", "preflight", "n_sampled", "primary_a_vs_b", "diff_in_diff",
                   "per_endpoint_arm_a", "per_trial", "disease_swap", "disease_swap_note", "branch",
                   "branch_reason", "meter", "git_sha", "wall_clock_secs"):
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
        # The swap arm was withdrawn (docs/t28b_l0_implementation_plan.md --
        # hardcoded-L7 rendering bug); it must stay None with an explanatory
        # note, not silently vanish or reappear as a dict.
        assert artifact["disease_swap"] is None
        assert "docs/t28b_l0_implementation_plan.md" in artifact["disease_swap_note"]
        assert len(artifact["per_trial"]["opus_arm_a"]) == 8
        assert set(REPORTED_METRICS_FOR_TEST) <= set(artifact["primary_a_vs_b"]["opus"])
        print("full pipeline (tiny sample) OK:", artifact["branch"], artifact["n_sampled"])


def test_l0_null_check_tiny_sample():
    """The corrected disease-sensitivity probe end to end: three arms
    (L1/L0/swap) on tiny Arm A/B slices, with a fake client that answers
    identically regardless of prompt content. No memorisation signature
    and no real sensitivity should be detectable under a flat response,
    but the pipeline must run start to finish and produce a self-describing
    artifact -- this is a wiring test, not a statistical-power test."""
    with tempfile.TemporaryDirectory() as results_dir:
        fake_client = _FakeConverseClient()
        out_path = os.path.join(results_dir, "t28b_l0_test.json")
        artifact = run_l0_null_check(
            data_root="data", results_dir=results_dir, n_arm_a=8, n_arm_b=8,
            n_l0_a=6, n_l0_b=6, seed=42, boto_client=fake_client, out_path=out_path,
            n_resamples=100,
        )
        for key in ("test_id", "inputs", "preflight", "curve_arm_a", "curve_arm_b", "reading",
                   "reading_reason", "per_trial", "meter", "git_sha", "wall_clock_secs"):
            assert key in artifact, f"missing top-level key {key!r}"
        assert artifact["test_id"] == "T28b-L0"
        assert artifact["reading"] in ("MEMORISATION_SIGNATURE", "QUARANTINE_LIFTED_DISEASE_UNINFORMATIVE",
                                      "NOT_SEPARABLE_ON_ARM_A_ALONE", "NORMAL_BEHAVIOUR")
        for curve_key in ("opus_l1_vs_l0", "opus_l1_vs_swap", "reference_l1_vs_l0", "reference_l1_vs_swap"):
            assert curve_key in artifact["curve_arm_a"]
            assert curve_key in artifact["curve_arm_b"]
        assert artifact["meter"]["calls"] == fake_client.calls
        print("l0 null check (tiny sample) OK:", artifact["reading"])


def _fixture_row(nct_id="NCT00000001", condition="Psoriasis"):
    """A single synthetic trial row shaped like TrialBench's schema
    (mirrors tests/test_serialize.py's synthetic sample), with the disease
    name deliberately present in both the brief summary and the eligibility
    criteria -- the two places the withdrawn swap arm leaked it from."""
    return pd.Series({
        "nct_id": nct_id,
        "phase": "Phase 2", "enrollment": 120, "number_of_arms": 2,
        "Active Comparator Arm Number": 1, "Experimental Arm Number": 1,
        "study_design_info/allocation": "Randomized",
        "study_design_info/intervention_model": "Parallel Assignment",
        "study_design_info/intervention_model_description": f"Patients with {condition} receive study drug.",
        "study_design_info/masking": "None (Open Label)",
        "study_design_info/primary_purpose": "Treatment",
        "eligibility/criteria/textblock": f"Inclusion: diagnosed with {condition}.",
        "eligibility/gender": "All", "eligibility/healthy_volunteers": "No",
        "eligibility/minimum_age": "18 Years", "eligibility/maximum_age": "75 Years",
        "sponsors/lead_sponsor/agency_class": "Industry",
        "oversight_info/has_dmc": "Yes",
        "oversight_info/is_fda_regulated_device": "No",
        "oversight_info/is_fda_regulated_drug": "Yes",
        "condition": f"['{condition}']",
        "condition_browse/mesh_term": "[]",
        "icdcode": "['L40.9']",
        "brief_summary/textblock": f"A study of {condition} in adult participants.",
    })


def test_swap_rendering_via_override_has_no_disease_leak():
    """The assertion whose absence cost the original swap arm
    (docs/t28b_l0_implementation_plan.md): a rendering that swaps only
    `condition` while leaving the shared body (brief summary, eligibility
    criteria) untouched still names the real disease. The corrected
    `render_arm_with_disease_override` must actually avoid this, and
    `assert_no_disease_leak` must actually catch it when it doesn't."""
    row = _fixture_row(condition="Psoriasis")
    slot_row = row.copy()
    slot_row["condition"] = "['Asthma']"

    # Corrected rendering: the override path swaps the disease slot only,
    # and the shared body's scrubber masks the ORIGINAL disease's terms
    # (row supplies the scrub source, slot_row only the filler).
    fixed = render_arm_with_disease_override(row, "L1", str(row["nct_id"]), slot_row)
    assert "psoriasis" not in fixed.text.lower(), (
        "corrected swap rendering still leaked the original disease:\n" + fixed.text
    )
    assert_no_disease_leak(row, slot_row, fixed.text)  # must not raise

    # The withdrawn bug, reproduced directly: rendering the already-swapped
    # row on its own (no override) leaves the untouched brief summary /
    # eligibility criteria naming the real disease verbatim.
    from src.data.serialize import render_arm
    buggy_swapped_row = row.copy()
    buggy_swapped_row["condition"] = "['Asthma']"
    buggy = render_arm(buggy_swapped_row, "L7", str(row["nct_id"]))
    assert "psoriasis" in buggy.text.lower(), "fixture no longer reproduces the withdrawn leak"
    try:
        assert_no_disease_leak(row, slot_row, buggy.text)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "assert_no_disease_leak must catch the withdrawn L7 rendering bug"
    print("swap-rendering disease-leak check OK")


def test_swap_disease_skips_when_own_chapter_is_unresolvable():
    """Found live on real Arm B data: NCT00782431 (condition "Influenza")
    has icdcode=None, so _row_chapters(row) came back empty and the
    original `[c for c in donor_pool if c not in own_chapters]` excluded
    NOTHING -- an unverifiable chapter silently read as "no conflicts"
    rather than "can't tell," so the donor could be drawn from the
    trial's own real disease area, including a same-NAMED donor from a
    different Arm A trial ("Influenza" swapped for "Influenza"). That
    crashed assert_no_disease_leak downstream; the real fix is here, one
    layer up, where the false eligibility was manufactured."""
    row = _fixture_row(condition="Influenza")
    row["icdcode"] = None
    # Adversarial on purpose: even the chapter that WOULD be correct for
    # this trial contains a donor literally named "Influenza" (as it did
    # live, from a different NCT) -- if own_chapters were ever incorrectly
    # resolved as non-empty here, this donor pool would still be able to
    # produce a same-name "swap" by chance, which the second guard below
    # covers separately.
    donor_pool = {"J00-J99": ["Influenza", "Asthma"], "C00-D49": ["Breast Cancer"]}
    for seed in range(20):
        result = swap_disease(row, donor_pool, np.random.default_rng(seed))
        assert result is None, (
            f"seed={seed}: a trial with an unresolvable icdcode must be skipped as a swap "
            f"candidate, not swapped -- got {result['condition'] if result is not None else result}"
        )
    print("swap_disease unresolvable-chapter guard OK: skipped on all 20 seeds")


def test_swap_disease_never_picks_a_same_named_donor():
    """Defense in depth for a second, independent way the same failure
    mode can happen: even with a resolvable, genuinely-different-chapter
    donor pool, real-world ICD/condition data can be noisy enough that the
    trial's own disease name also appears as a donor entry in an eligible
    chapter (e.g. a miscategorized trial). swap_disease must filter those
    out before drawing, not rely on chapter exclusion alone."""
    row = _fixture_row(condition="Influenza")
    row["icdcode"] = "['J09']"  # resolves to J00-J99, a real, non-empty chapter
    # The only "eligible" (different-chapter) donor pool also happens to
    # contain "Influenza" itself, alongside a genuinely different disease.
    donor_pool = {"C00-D49": ["Influenza", "Breast Cancer"]}
    seen_names = set()
    for seed in range(30):
        result = swap_disease(row, donor_pool, np.random.default_rng(seed))
        if result is not None:
            from src.data.features import _recursive_parse_terms
            names = _recursive_parse_terms(result["condition"])
            seen_names.update(names)
            assert "influenza" not in [n.lower() for n in names], (
                f"seed={seed}: swap_disease picked a same-named donor -- not a real swap"
            )
    assert seen_names, "expected at least one successful swap to Breast Cancer across 30 seeds"
    print("swap_disease same-name-donor guard OK: never picked 'Influenza', saw", seen_names)


def test_swap_disease_condition_is_clean_python_literal():
    """numpy 2.x regression this repo hit live: rng.choice on a Python
    list of str returns a numpy str_ scalar, and repr() on THAT scalar is
    "np.str_('Asthma')", not "'Asthma'" -- repr([donor_name]) then writes
    condition as the garbled literal text "[np.str_('Asthma')]", which
    ast.literal_eval can't parse. _disease_filler's parser silently falls
    back to treating that whole string as the disease name, so the model
    would have been shown "[np.str_('Asthma')]" verbatim as the swapped
    disease -- not a crash, and not caught by assert_no_disease_leak
    (which only checks the ORIGINAL disease's terms, not donor validity)."""
    row = _fixture_row(condition="Influenza")
    row["icdcode"] = "['J09']"
    donor_pool = {"C00-D49": ["Breast Cancer", "Asthma"]}
    result = swap_disease(row, donor_pool, np.random.default_rng(0))
    assert result is not None
    assert "np.str_" not in result["condition"], (
        f"condition is a garbled numpy repr, not a clean Python list literal: {result['condition']!r}"
    )
    from src.data.features import _recursive_parse_terms
    terms = _recursive_parse_terms(result["condition"])
    assert terms == [str(t) for t in terms] and all(isinstance(t, str) for t in terms), terms
    assert terms[0] in ("Breast Cancer", "Asthma"), terms
    print("swap_disease clean-literal check OK:", result["condition"])


def test_elicit_row_renders_requested_arm():
    """Regression guard for the exact bug P1 fixed: elicit_row used to
    hardcode render_arm(row, "L7", ...) regardless of the `arm` argument.
    Patches the module's render_arm to record what arm it was actually
    called with, for both the default and an explicit non-default arm."""
    row = _fixture_row()
    client = BedrockClient(region="us-west-2", boto_client=_FakeConverseClient())
    meter = Meter()
    seen_arms = []
    original_render_arm = t28b_mod.render_arm

    def recording_render_arm(row, arm, nct_id):
        seen_arms.append(arm)
        return original_render_arm(row, arm, nct_id)

    with tempfile.TemporaryDirectory() as results_dir:
        with mock.patch.object(t28b_mod, "render_arm", side_effect=recording_render_arm):
            elicit_row(client, results_dir, "fake-model-id", "mortality_rate_yn", row, meter)
            elicit_row(client, results_dir, "fake-model-id", "mortality_rate_yn", row, meter, arm="L0")

    assert seen_arms == ["L7", "L0"], (
        f"elicit_row must render the arm it is passed, not a hardcoded one -- saw {seen_arms}"
    )
    print("elicit_row arm-parameterisation check OK:", seen_arms)


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
        test_l0_null_check_tiny_sample()
    test_swap_rendering_via_override_has_no_disease_leak()
    test_swap_disease_skips_when_own_chapter_is_unresolvable()
    test_swap_disease_never_picks_a_same_named_donor()
    test_swap_disease_condition_is_clean_python_literal()
    test_elicit_row_renders_requested_arm()
    test_decide_recall_demonstrated_needs_both_r1_and_r2()
    test_decide_calibration_artifact_when_opus_auroc_flat()
    test_decide_inconclusive_when_r1_passes_but_r2_does_not()
    print("t28b tests passed")
