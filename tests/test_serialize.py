"""P13.2 acceptance test (wave2-start-plan.md): standing rule 9 as an
assertion. Renders sampled trials across all seven implemented arms
(L0-L5, L7 -- L6 isn't built yet, see serialize.py's ARM_DESCRIPTIONS) and
checks the byte-identical-residual invariant, plus a deliberate mutation
that must fail it.

Prefers real TrialBench data (any task/phase folder under data_root) so the
free-text edge cases real trials produce (BMI formulas with literal braces,
condition names that collide with unrelated field text, disease names
embedded in eligibility criteria / intervention-model descriptions) are
actually exercised; falls back to a small synthetic frame shaped like
test_smoke.py's when no data is present, so this still runs somewhere
without a data download.

Run:  python -m pytest -q tests/test_serialize.py   (or)
      python tests/test_serialize.py
"""
from __future__ import annotations

import glob
import os
import sys
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data.serialize as serialize_mod  # noqa: E402
from src.data.serialize import ARMS, Rendered, assert_body_identical, render_arm  # noqa: E402

CHECK_ARMS = tuple(a for a in ARMS if a != "L6")  # L6 not yet implemented


def _find_real_sample(n=200, seed=42):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = glob.glob(os.path.join(here, "data", "*", "Phase*", "train_x.csv"))
    if not hits:
        return None
    df = pd.read_csv(sorted(hits)[0], index_col=0, low_memory=False)
    return df.sample(min(n, len(df)), random_state=seed)


def _sample(n=200, seed=42):
    real = _find_real_sample(n=n, seed=seed)
    return real if real is not None else _synthetic_sample(n=min(n, 20), seed=seed)


def _synthetic_sample(n=20, seed=42):
    conditions = ["Type 2 Diabetes Mellitus", "Asthma", "Breast Cancer", "Healthy"]
    codes = ["['E11.9']", "['J45.909']", "['C50.911']", "[]"]
    rows = []
    for i in range(n):
        j = i % len(conditions)
        rows.append({
            "phase": "Phase 2", "enrollment": 100 + i, "number_of_arms": 2,
            "Active Comparator Arm Number": 1, "Experimental Arm Number": 1,
            "study_design_info/allocation": "Randomized",
            "study_design_info/intervention_model": "Parallel Assignment",
            "study_design_info/intervention_model_description": (
                f"Patients with {conditions[j]} receive study drug." if j != 3 else ""
            ),
            "study_design_info/masking": "None (Open Label)",
            "study_design_info/primary_purpose": "Treatment",
            "eligibility/criteria/textblock": f"Inclusion: diagnosed with {conditions[j]}.",
            "eligibility/gender": "All", "eligibility/healthy_volunteers": (
                "Accepts Healthy Volunteers" if conditions[j] == "Healthy" else "No"
            ),
            "eligibility/minimum_age": "18 Years", "eligibility/maximum_age": "75 Years",
            "sponsors/lead_sponsor/agency_class": "Industry",
            "oversight_info/has_dmc": "Yes",
            "oversight_info/is_fda_regulated_device": "No",
            "oversight_info/is_fda_regulated_drug": "Yes",
            "condition": f"['{conditions[j]}']",
            "condition_browse/mesh_term": "[]",
            "icdcode": codes[j],
            "brief_summary/textblock": f"A study of {conditions[j]}.",
        })
    return pd.DataFrame(rows, index=[f"NCT{i:08d}" for i in range(n)])


def test_all_arms_render_no_empty_slot():
    sample = _sample()
    for nct_id, row in sample.iterrows():
        for arm in CHECK_ARMS:
            rendered = render_arm(row, arm, str(nct_id))
            assert rendered.disease.strip(), f"{nct_id}/{arm}: empty disease slot"
            assert rendered.text.strip()


def test_body_identical_across_arms():
    sample = _sample()
    assert_body_identical(sample, arms=CHECK_ARMS)  # raises on failure


def test_mutation_is_caught():
    """Patches the real render_arm (as assert_body_identical actually calls
    it -- an unqualified module-global lookup, so patching the module
    attribute affects calls made from inside assert_body_identical too) so
    L1's rendering picks up an extra arm-specific line outside the slot.
    The residual-equality check must catch it."""
    sample = _sample(n=5)
    original_render_arm = serialize_mod.render_arm

    def buggy_render_arm(row, arm, nct_id):
        rendered = original_render_arm(row, arm, nct_id)
        if arm == "L1":
            rendered = Rendered(
                nct_id=rendered.nct_id, arm=arm,
                text=rendered.text + "\nEXTRA ARM-SPECIFIC LINE",
                disease=rendered.disease,
            )
        return rendered

    raised = False
    with mock.patch.object(serialize_mod, "render_arm", side_effect=buggy_render_arm):
        try:
            serialize_mod.assert_body_identical(sample, arms=CHECK_ARMS)
        except AssertionError:
            raised = True
    assert raised, "a deliberate body mutation must be detected as a residual mismatch"


if __name__ == "__main__":
    test_all_arms_render_no_empty_slot()
    test_body_identical_across_arms()
    test_mutation_is_caught()
    print("serialize test passed")
