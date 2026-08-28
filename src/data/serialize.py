"""P13.2 (wave2-start-plan.md): one trial serialization template, one
``{DISEASE}`` slot, shared by all of P13.6's arms (L0-L7). The template's
non-slot fields are exactly the plan's enumerated list -- ``phase``,
``enrollment``, ``number_of_arms``, the intervention and arm counts,
``study_design_info/*``, ``eligibility/*``,
``sponsors/lead_sponsor/agency_class``, ``oversight_info/*`` -- and nothing
else. Two columns are deliberately excluded even though they're on the row:
``brief_summary/textblock`` names the disease in plain text and belongs to
L7's slot content alone, and ``brief_title`` commonly does too, so both stay
out of the shared body. ``condition``, ``condition_browse/mesh_term`` and
``icdcode`` are disease fields -- slot content for the arms that use them,
never body content.

Standing rule 9 (the experiment's validity) is implemented as an assertion,
not a convention: :func:`assert_body_identical` renders every arm for a
sample of trials, strips each arm's own disease substring out of its
rendering, and requires the eight residuals to be byte-identical. Run it
over the whole sample before any model call is made.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .ccsr import icd10_ccsr, icd10_ccsr_name
from .features import _disease_mask_terms, _recursive_parse_terms
from .icd10_hierarchy import icd10_block, icd10_chapter, icd10_full_descriptor

ARMS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")

ARM_DESCRIPTIONS = {
    "L0": "masked (disease slot withheld)",
    "L1": "raw condition name(s)",
    "L2": "bare ICD-10-CM code(s)",
    "L3": "ICD-10-CM code(s) + official descriptor(s)",
    "L4": "MeSH condition term(s)",
    "L5": "CCSR category name(s)",
    "L6": "condition name(s) + enriched external context (NOT YET IMPLEMENTED -- "
          "see the L6 note in wave2-start-plan.md P13.6: needs a pinned MONDO/PrimeKG "
          "source and a committed scrub list before it can render)",
    "L7": "condition name(s) + the trial's own brief summary",
}

L0_MASK_TEXT = "[condition withheld]"

# Terms that denote the *absence* of a disease (a healthy-volunteer study)
# rather than naming one. "Healthy" is a real ``condition`` value in this
# corpus and also appears legitimately in the unrelated
# ``eligibility/healthy_volunteers`` field's own text ("Accepts Healthy
# Volunteers") -- a genuine word collision, not a serialization leak, and the
# mirror image of wave1-preflight-review.md's L7 finding (short condition
# strings like "ALL" scrubbing unrelated words). There is nothing to blind
# when the "condition" is the absence of one, so these are excluded from
# assert_body_identical's leak check rather than producing a permanent false
# positive on real data.
GENERIC_NON_DISEASE_TERMS = frozenset({
    "healthy", "healthy volunteer", "healthy volunteers",
    "normal", "normal volunteers", "n/a", "na", "none", "unknown",
})

# The plan's exact non-slot field list. Order is fixed so the rendered body
# is deterministic across processes/runs (a prompt_sha256 that moved because
# dict iteration order changed would be its own silent bug).
_ARM_COUNT_COLS = (
    "Active Comparator Arm Number", "Experimental Arm Number",
    "No Intervention Arm Number", "Other Arm Number",
    "Placebo Comparator Arm Number", "Sham Comparator Arm Number",
)
_INTERVENTION_COUNT_COLS = (
    "Behavioral intervention Number", "Biological intervention Number",
    "Combination Product intervention Number", "Device intervention Number",
    "Diagnostic Test intervention Number", "Dietary Supplement intervention Number",
    "Drug intervention Number", "Genetic intervention Number",
    "Other intervention Number", "Procedure intervention Number",
    "Radiation intervention Number",
)
_STUDY_DESIGN_COLS = (
    "study_design_info/allocation", "study_design_info/intervention_model",
    "study_design_info/intervention_model_description", "study_design_info/masking",
    "study_design_info/masking_description", "study_design_info/masking_num",
    "study_design_info/primary_purpose",
)
_ELIGIBILITY_COLS = (
    "eligibility/criteria/textblock", "eligibility/gender",
    "eligibility/healthy_volunteers", "eligibility/maximum_age",
    "eligibility/minimum_age",
)
_OVERSIGHT_COLS = (
    "oversight_info/has_dmc", "oversight_info/is_fda_regulated_device",
    "oversight_info/is_fda_regulated_drug",
)
_SPONSOR_COLS = ("sponsors/lead_sponsor/agency_class",)

# Every column the shared body may read, in render order. DISEASE_COLS
# (below) must never appear here -- assert_body_identical is what catches a
# future edit that adds one by accident.
BODY_COLS = (
    ("phase", "Phase"),
    ("enrollment", "Enrollment"),
    ("number_of_arms", "Number of arms"),
) + tuple((c, c) for c in _ARM_COUNT_COLS + _INTERVENTION_COUNT_COLS) + (
    ("study_design_info/allocation", "Allocation"),
    ("study_design_info/intervention_model", "Intervention model"),
    ("study_design_info/intervention_model_description", "Intervention model description"),
    ("study_design_info/masking", "Masking"),
    ("study_design_info/masking_description", "Masking description"),
    ("study_design_info/masking_num", "Masking (numeric)"),
    ("study_design_info/primary_purpose", "Primary purpose"),
    ("eligibility/criteria/textblock", "Eligibility criteria"),
    ("eligibility/gender", "Eligibility gender"),
    ("eligibility/healthy_volunteers", "Healthy volunteers accepted"),
    ("eligibility/minimum_age", "Minimum age"),
    ("eligibility/maximum_age", "Maximum age"),
    ("sponsors/lead_sponsor/agency_class", "Lead sponsor class"),
    ("oversight_info/has_dmc", "Has data monitoring committee"),
    ("oversight_info/is_fda_regulated_device", "FDA-regulated device"),
    ("oversight_info/is_fda_regulated_drug", "FDA-regulated drug"),
)

# Columns that must NEVER be read by _render_body -- disease content (slot
# only) or a field that leaks the disease in free text (L7's slot only).
DISEASE_LEAK_COLS = frozenset({
    "condition", "condition_browse/mesh_term", "icdcode",
    "brief_summary/textblock", "brief_title",
})


_DISEASE_SLOT = "\x00DISEASE_SLOT\x00"


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none") else "unknown"


def _scrub_disease(text: str, terms: set) -> str:
    """Same masking ``disease_blind_text`` already uses elsewhere in this
    repo (P10) -- exact phrase, case-insensitive, longest phrase first so a
    longer match isn't shadowed by a shorter substring of itself. Reused
    here, not reimplemented, so both scrubbers share one fix if the
    word-boundary gap (wave1-preflight-review.md L7) is ever tightened."""
    for t in sorted(terms, key=len, reverse=True):
        text = re.sub(re.escape(t), " ", text, flags=re.IGNORECASE)
    return text


# Measured 2026-08-26 on 500-trial samples from four task/phase folders:
# eligibility/criteria/textblock names the trial's own condition or a MeSH
# synonym in 45-61% of trials before scrubbing (mortality/Phase1 45.4%, SAE/
# Phase2 60.6%, dropout/Phase3 59.7%, outcome/Phase1 52.8%) -- large enough
# that L0 (the masked null) would not actually be blind to the disease for
# roughly half the grid. The same free-text-in-a-mandated-body-field problem
# also showed up in study_design_info/intervention_model_description (e.g.
# "Patients with novel coronavirus (COVID-19) will be treated with...").
# masking_description is the same shape of field (free-text prose, not a
# short enum) and is scrubbed for the same reason even though a leak wasn't
# directly observed in it. Scrubbed here with the same phrase-match masking
# P10's disease_blind arm already uses (_scrub_disease above); like that
# arm, this under-masks rather than over-masks (L7: no word boundaries, and
# no MeSH synonym expansion), so a residual leak rate should still be
# measured on the actual T28a/T28 sample and recorded in their artifacts
# rather than assumed to be zero.
_SCRUB_COLS = frozenset({
    "eligibility/criteria/textblock",
    "study_design_info/intervention_model_description",
    "study_design_info/masking_description",
})


def _render_body(row: pd.Series) -> str:
    mask_terms = _disease_mask_terms(row.get("condition"), row.get("condition_browse/mesh_term"))
    lines = []
    for col, label in BODY_COLS:
        value = _fmt(row.get(col))
        if col in _SCRUB_COLS and mask_terms:
            value = _scrub_disease(value, mask_terms)
        lines.append(f"{label}: {value}")
    # A sentinel, not "{DISEASE}" + str.format: free-text fields (eligibility
    # criteria, intervention-model descriptions) routinely contain literal
    # curly braces (e.g. a BMI formula, "weight (kg) / {height (m)}^2"),
    # which str.format would misparse as an unrelated placeholder.
    lines.append(f"Condition: {_DISEASE_SLOT}")
    return "\n".join(lines)


def _parse_terms(row: pd.Series, col: str) -> list:
    if col not in row.index:
        return []
    return sorted(set(_recursive_parse_terms(row[col])))


def _parse_codes(row: pd.Series) -> list:
    if "icdcode" not in row.index:
        return []
    return sorted({c.strip().upper() for c in _recursive_parse_terms(row["icdcode"]) if c.strip()})


def _disease_filler(row: pd.Series, arm: str) -> str:
    if arm == "L0":
        return L0_MASK_TEXT

    if arm == "L1":
        terms = _parse_terms(row, "condition")
        return "; ".join(terms) if terms else L0_MASK_TEXT

    if arm == "L2":
        codes = _parse_codes(row)
        return "; ".join(codes) if codes else L0_MASK_TEXT

    if arm == "L3":
        codes = _parse_codes(row)
        if not codes:
            return L0_MASK_TEXT
        parts = []
        for c in codes:
            desc = icd10_full_descriptor(c)
            parts.append(f"{c} ({desc})" if desc else c)
        return "; ".join(parts)

    if arm == "L4":
        terms = _parse_terms(row, "condition_browse/mesh_term")
        return "; ".join(terms) if terms else L0_MASK_TEXT

    if arm == "L5":
        codes = _parse_codes(row)
        if not codes:
            return L0_MASK_TEXT
        cats = sorted({cat for c in codes for cat in icd10_ccsr(c)})
        names = [icd10_ccsr_name(c) or c for c in cats]
        return "; ".join(names) if names else L0_MASK_TEXT

    if arm == "L6":
        raise NotImplementedError(
            "L6 needs a pinned MONDO/PrimeKG description source and a scrub list "
            "committed to the repo before it can render -- see P13.6 in "
            "wave2-start-plan.md. Not yet decided."
        )

    if arm == "L7":
        names = _parse_terms(row, "condition")
        name_str = "; ".join(names) if names else L0_MASK_TEXT
        summary = _fmt(row.get("brief_summary/textblock"))
        return f"{name_str}. {summary}" if summary != "unknown" else name_str

    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


@dataclass
class Rendered:
    nct_id: str
    arm: str
    text: str
    disease: str


def render_arm(row: pd.Series, arm: str, nct_id: str) -> Rendered:
    """Render one trial row for one arm. ``disease`` is the exact substring
    inserted at the ``{DISEASE}`` slot -- callers that need to verify
    standing rule 9 use it to strip the slot back out (see
    :func:`assert_body_identical`)."""
    disease = _disease_filler(row, arm)
    text = _render_body(row).replace(_DISEASE_SLOT, disease)
    return Rendered(nct_id=nct_id, arm=arm, text=text, disease=disease)


def render_arm_with_disease_override(row: pd.Series, arm: str, nct_id: str,
                                     slot_row: pd.Series) -> Rendered:
    """Like :func:`render_arm`, but the disease SLOT's content comes from
    ``slot_row`` while the shared body -- and critically, `_render_body`'s
    scrubbing -- still uses ``row``'s own ``condition``/
    ``condition_browse/mesh_term``.

    Needed for a disease-*swap* probe (docs/t28b_l0_implementation_plan.md
    P2): swapping ``row["condition"]`` to a donor disease and rendering
    that single swapped row with plain :func:`render_arm` scrubs the
    *donor's* terms from ``_SCRUB_COLS`` (which appear nowhere in this
    trial's real eligibility criteria) while leaving the REAL disease's
    mentions there unredacted -- silently leaking the original disease
    back into what was supposed to be a swapped prompt. Passing the
    ORIGINAL row for scrubbing and a separate ``slot_row`` (a copy with
    only the disease fields swapped) for the slot fixes this: the body is
    scrubbed against what could actually leak, the slot shows the donor.
    """
    disease = _disease_filler(slot_row, arm)
    text = _render_body(row).replace(_DISEASE_SLOT, disease)
    return Rendered(nct_id=nct_id, arm=arm, text=text, disease=disease)


def assert_body_identical(df: pd.DataFrame, arms=ARMS) -> None:
    """Standing rule 9, as an assertion: for every row in ``df``, render
    every arm (except L6, not yet implemented), strip that arm's own disease
    substring out of the rendering, and require the residual to be
    byte-identical across arms. Raises ``AssertionError`` naming the first
    trial and arm pair that disagrees.

    Also asserts the disease string doesn't leak into L0's rendering (the
    masked null) -- if it did, the serialization itself would be reading a
    disease-bearing column into the shared body.
    """
    check_arms = [a for a in arms if a != "L6"]
    for nct_id, row in df.iterrows():
        residuals = {}
        l0_disease_terms = _parse_terms(row, "condition") + _parse_codes(row) + _parse_terms(row, "condition_browse/mesh_term")
        for arm in check_arms:
            rendered = render_arm(row, arm, str(nct_id))
            # The slot is always the last thing rendered ("Condition:
            # {DISEASE}" is BODY_COLS's final line), so strip it by
            # position -- not by searching for the disease string's text,
            # which can also occur earlier in the body by coincidence (e.g.
            # condition="Healthy" collides with the unrelated
            # eligibility/healthy_volunteers field's own wording) and would
            # strip the wrong occurrence.
            if not rendered.text.endswith(rendered.disease):
                raise AssertionError(
                    f"trial {nct_id}: arm {arm!r} rendering doesn't end with its own "
                    f"disease slot content -- template changed under render_arm?"
                )
            residual = rendered.text[: len(rendered.text) - len(rendered.disease)]
            residuals[arm] = residual
            if arm == "L0":
                for term in l0_disease_terms:
                    if term and term.lower() not in GENERIC_NON_DISEASE_TERMS and term in rendered.text:
                        raise AssertionError(
                            f"trial {nct_id}: L0 (masked) rendering leaks disease term "
                            f"{term!r} -- {rendered.text!r}"
                        )
        base_arm = check_arms[0]
        base = residuals[base_arm]
        for arm, residual in residuals.items():
            if residual != base:
                raise AssertionError(
                    f"trial {nct_id}: arm {arm!r} residual differs from {base_arm!r} "
                    f"after stripping the disease slot -- serialization leaked a "
                    f"non-slot difference.\n{arm}: {residual!r}\n{base_arm}: {base!r}"
                )
