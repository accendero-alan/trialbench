"""P14.4 (wave2-start-plan.md): emit a fresh-slice DataFrame in TrialBench's
own ~60-column tabular/raw schema, reconstructed from the AACT snapshot, so
T29's frozen classical arms can run `TabularFeaturizer`/`CodeFeaturizer` on
it unmodified.

Column-for-column strategy, decided per P14.4:

- Most columns map onto a single AACT table (``studies``, ``designs``,
  ``eligibilities``, ...) directly, or a per-trial join with a
  ``"; "``-joined string where TrialBench's own column can hold more than
  one value per trial (``condition``, ``keyword``, intervention name, ...).
- **``icdcode`` is reproduced, not skipped** (P14.4's explicit recommendation
  over T27's cleaner-but-different SapBERT re-mapper, since T27 itself is
  out of scope this wave): each AACT ``conditions.name`` string is looked up
  against the NLM Clinical Tables ICD-10-CM lexical search API --
  ``https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search`` -- the same
  public, free lexical service TrialBench's own condition-to-ICD mapping is
  built on (disease-representation-test-plan.md's T27 section). This is a
  best-effort reproduction, not a validated match to TrialBench's exact
  mapper (that validation is T27's job and stays out of scope) -- flagged
  here, not glossed over.
- Columns whose source table was never extracted from the AACT archive
  (``intervention_browse/mesh_term``, the 5 ``ipd_info_type-*`` flags,
  ``responsible_party/responsible_party_type``, ``smiless``) are emitted
  as an all-missing column, per P14.4's own instruction: "columns that
  cannot be reconstructed are emitted missing the way the training data
  encodes missing, and the count reported."
- ``location/facility/address/city`` is a many-to-one approximation
  (TrialBench's own join convention for multi-facility trials is unknown;
  this module takes the first facility row per trial, alphabetically by
  city, and flags the column as approximate below rather than exact).

See ``docs/p14_4_schema_slice.md`` for the missing-rate report this module
produced against the P14.5 fresh slice, and which columns are approximate
vs. exact vs. all-missing.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pandas as pd

from .aact import load_table_columns, load_table_rows, results_posted_date

NLM_ICD10CM_SEARCH_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

# Columns whose AACT source table was never extracted (aact.py's module
# docstring lists the 17 tables actually on disk) -- always emitted missing.
ALWAYS_MISSING_COLUMNS = (
    "intervention_browse/mesh_term",
    "ipd_info_type-Analytic Code",
    "ipd_info_type-Clinical Study Report (CSR)",
    "ipd_info_type-Informed Consent Form (ICF)",
    "ipd_info_type-Statistical Analysis Plan (SAP)",
    "ipd_info_type-Study Protocol",
    "responsible_party/responsible_party_type",
    "smiless",
)

# Columns reconstructed via a many-to-one approximation (first/joined row,
# not necessarily TrialBench's own tie-breaking rule) -- reported separately
# from exact reconstructions in the missing-rate writeup, not silently
# treated the same.
APPROXIMATE_COLUMNS = (
    "location/facility/address/city", "condition", "condition_browse/mesh_term",
    "intervention/description", "intervention/intervention_name", "keyword", "icdcode",
)

_INTERVENTION_TYPE_COLS = {
    "Behavioral": "Behavioral intervention Number", "Biological": "Biological intervention Number",
    "Combination Product": "Combination Product intervention Number", "Device": "Device intervention Number",
    "Diagnostic Test": "Diagnostic Test intervention Number", "Dietary Supplement": "Dietary Supplement intervention Number",
    "Drug": "Drug intervention Number", "Genetic": "Genetic intervention Number",
    "Other": "Other intervention Number", "Procedure": "Procedure intervention Number",
    "Radiation": "Radiation intervention Number",
}
_ARM_TYPE_COLS = {
    "Active Comparator": "Active Comparator Arm Number", "Experimental": "Experimental Arm Number",
    "No Intervention": "No Intervention Arm Number", "Other": "Other Arm Number",
    "Placebo Comparator": "Placebo Comparator Arm Number", "Sham Comparator": "Sham Comparator Arm Number",
}
_MASKING_ROLE_COLS = {
    "subject_masked": "MaskingType-Participant", "caregiver_masked": "MaskingType-Care Provider",
    "investigator_masked": "MaskingType-Investigator", "outcomes_assessor_masked": "MaskingType-Outcomes Assessor",
}
_YESNO = {"t": "Yes", "f": "No"}


def _icd_lookup(term: str, max_codes: int = 7, timeout_secs: float = 8.0) -> list:
    """One NLM Clinical Tables lexical search call. Returns up to
    ``max_codes`` ICD-10-CM code strings (empty list on no match, a network
    error, or an empty/NaN term -- never raises, since one bad condition
    string must not fail the whole slice)."""
    if not term or not isinstance(term, str) or not term.strip():
        return []
    url = f"{NLM_ICD10CM_SEARCH_URL}?sf=code,name&terms={urllib.request.quote(term.strip())}&maxList={max_codes}"
    req = urllib.request.Request(url, headers={"User-Agent": "trialbench-p14.4/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
            payload = json.loads(resp.read())
        return list(payload[1]) if payload and payload[0] else []
    except (urllib.error.URLError, TimeoutError, ValueError, IndexError, TypeError):
        return []


def reproduce_icdcode_column(condition_lists: pd.Series, cache: dict = None,
                             rate_limit_secs: float = 0.34) -> pd.Series:
    """``condition_lists`` maps nct_id -> list of condition-name strings
    (from AACT ``conditions.name``, one row per trial). Returns a Series in
    TrialBench's own nested-repr format (a stringified list of stringified
    per-condition code lists) so ``features.py``'s ``_recursive_parse_terms``
    parses it identically to the real column. ``cache`` lets a caller reuse
    lookups across calls (condition strings repeat heavily across trials).
    ``rate_limit_secs`` is a courtesy delay against the free public API --
    at ~3 req/s, a few hundred distinct condition strings is well under a
    minute."""
    cache = {} if cache is None else cache
    out = []
    for conditions in condition_lists:
        per_condition_lists = []
        for term in conditions:
            if term not in cache:
                cache[term] = _icd_lookup(term)
                time.sleep(rate_limit_secs)
            codes = cache[term]
            if codes:
                per_condition_lists.append(repr(codes))
        out.append(repr(per_condition_lists) if per_condition_lists else None)
    return pd.Series(out, index=condition_lists.index)


def slice_ab_nct_ids(cutoff: "pd.Timestamp | str", snapshot_dir: str = None) -> dict:
    """P14.5's slice (a)/(b) partition (`docs/p14_5_n_gate.md`), for an
    arbitrary cutoff -- previously only ever run ad-hoc, output never
    checked in (that document's own Data section says so;
    `docs/t28b_opus_recall_spec.md` flagged the gap this closes).

    - **(a)** registered pre-cutoff, results posted post-cutoff:
      `study_first_posted_date < cutoff` and `results_first_posted_date >
      cutoff` -- trial identity known, outcome not memorisable.
    - **(b)** fully post-cutoff, both dates after cutoff -- trial unknown.

    `cutoff` should already be a resolved month-end timestamp (e.g.
    `experiments/t28a_contamination_probes.py`'s `_cutoff_to_date`,
    `pd.Period("2025-03", freq="M").end_time`), not a `"YYYY-MM"` string,
    to match p14_5's own reading exactly -- passed through
    `pd.Timestamp(cutoff)` either way so a raw string still works, just
    without the month-end rounding.

    Returns `{"a": [nct_id, ...], "b": [nct_id, ...]}` -- nct_id lists,
    not rows. Feed either list into `emit_trialbench_schema` for
    TrialBench-shaped columns.
    """
    cutoff = pd.Timestamp(cutoff)
    kw = {} if snapshot_dir is None else {"snapshot_dir": snapshot_dir}
    studies = load_table_columns("studies", ["nct_id", "study_first_posted_date"], **kw)
    studies = studies.drop_duplicates("nct_id").set_index("nct_id")["study_first_posted_date"]
    registration = pd.to_datetime(studies, errors="coerce")
    results = results_posted_date(**kw)

    both = pd.DataFrame({"reg": registration, "res": results})
    slice_a = both.index[(both["reg"] < cutoff) & (both["res"] > cutoff)].tolist()
    slice_b = both.index[(both["reg"] > cutoff) & (both["res"] > cutoff)].tolist()
    return {"a": sorted(slice_a), "b": sorted(slice_b)}


def emit_trialbench_schema(nct_ids: list, snapshot_dir: str = None, do_icdcode: bool = True) -> pd.DataFrame:
    """The P14.4 deliverable: one row per ``nct_id``, columns matching
    TrialBench's own ``train_x.csv`` schema as closely as the extracted AACT
    tables allow. ``do_icdcode=False`` skips the (slow, network-bound) ICD
    reconstruction -- useful for a quick featurizer smoke-pass; the real
    P14.4 report runs it.
    """
    kw = {} if snapshot_dir is None else {"snapshot_dir": snapshot_dir}
    idx = pd.Index(nct_ids, name="nct_id")
    out = pd.DataFrame(index=idx)

    # load_table_rows throughout, not load_table's/load_table_columns's
    # full-table read -- confirmed live (T28b's OOM, 2026-08-28): reading
    # every row of eligibilities (900MB on disk, includes the large
    # free-text `criteria` column), brief_summaries (421MB, `description`),
    # facilities (389MB), interventions (217MB) and the rest, THEN
    # filtering down to a handful of nct_ids after the fact, still means
    # holding the whole table in memory first regardless of how few
    # columns are requested -- column-narrowing alone only got a 3.78GB
    # kill down to ~4GB peak, not a fix. Row-filtering during a chunked
    # read (load_table_rows) bounds peak memory to one chunk at a time
    # instead of the whole table, which is what this function actually
    # needs: it only ever wants a handful of trials.
    studies_cols = ["nct_id", "brief_title", "enrollment", "number_of_arms", "phase", "study_type",
                    "has_expanded_access", "has_dmc", "is_fda_regulated_device",
                    "is_fda_regulated_drug", "plan_to_share_ipd"]
    studies = load_table_rows("studies", nct_ids, studies_cols, **kw).set_index("nct_id").reindex(idx)
    out["brief_title"] = studies["brief_title"] if "brief_title" in studies else pd.NA
    out["enrollment"] = studies.get("enrollment")
    out["number_of_arms"] = studies.get("number_of_arms")
    out["phase"] = studies.get("phase")
    out["study_type"] = studies.get("study_type")
    out["has_expanded_access"] = studies.get("has_expanded_access").map(_YESNO)
    out["oversight_info/has_dmc"] = studies.get("has_dmc").map(_YESNO)
    out["oversight_info/is_fda_regulated_device"] = studies.get("is_fda_regulated_device").map(_YESNO)
    out["oversight_info/is_fda_regulated_drug"] = studies.get("is_fda_regulated_drug").map(_YESNO)
    out["patient_data/sharing_ipd"] = studies.get("plan_to_share_ipd").map(
        {"YES": "Yes", "NO": "No", "UNDECIDED": "Undecided"})
    del studies

    bs = load_table_rows("brief_summaries", nct_ids, ["nct_id", "description"], **kw).set_index("nct_id")
    out["brief_summary/textblock"] = bs["description"].reindex(idx) if "description" in bs else pd.NA
    del bs

    elig_cols = ["nct_id", "criteria", "gender", "healthy_volunteers", "minimum_age", "maximum_age"]
    elig = load_table_rows("eligibilities", nct_ids, elig_cols, **kw).set_index("nct_id").reindex(idx)
    out["eligibility/criteria/textblock"] = elig.get("criteria")
    out["eligibility/gender"] = elig.get("gender")
    out["eligibility/healthy_volunteers"] = elig.get("healthy_volunteers")
    out["eligibility/minimum_age"] = elig.get("minimum_age")
    out["eligibility/maximum_age"] = elig.get("maximum_age")
    del elig

    designs_cols = ["nct_id", "allocation", "intervention_model", "intervention_model_description",
                    "masking", "masking_description", "primary_purpose", *_MASKING_ROLE_COLS]
    designs = load_table_rows("designs", nct_ids, designs_cols, **kw).set_index("nct_id").reindex(idx)
    out["study_design_info/allocation"] = designs.get("allocation")
    out["study_design_info/intervention_model"] = designs.get("intervention_model")
    out["study_design_info/intervention_model_description"] = designs.get("intervention_model_description")
    out["study_design_info/masking"] = designs.get("masking")
    out["study_design_info/masking_description"] = designs.get("masking_description")
    out["study_design_info/primary_purpose"] = designs.get("primary_purpose")
    n_masked_roles = sum((designs.get(role) == "t").astype(int) for role in _MASKING_ROLE_COLS if role in designs)
    out["study_design_info/masking_num"] = n_masked_roles if isinstance(n_masked_roles, pd.Series) else pd.NA
    for role, col in _MASKING_ROLE_COLS.items():
        out[col] = (designs[role] == "t").astype(float) if role in designs else pd.NA
    del designs

    sponsors = load_table_rows("sponsors", nct_ids, ["nct_id", "lead_or_collaborator", "agency_class"], **kw)
    lead = sponsors[sponsors["lead_or_collaborator"] == "lead"].drop_duplicates("nct_id").set_index("nct_id")
    out["sponsors/lead_sponsor/agency_class"] = lead["agency_class"].reindex(idx)
    del sponsors, lead

    dg = load_table_rows("design_groups", nct_ids, ["nct_id", "group_type"], **kw)
    arm_counts = dg.groupby(["nct_id", "group_type"]).size().unstack(fill_value=0)
    for group_type, col in _ARM_TYPE_COLS.items():
        out[col] = arm_counts[group_type].reindex(idx).fillna(0) if group_type in arm_counts else 0.0
    del dg, arm_counts

    iv = load_table_rows("interventions", nct_ids, ["nct_id", "intervention_type", "name", "description"], **kw)
    iv_counts = iv.groupby(["nct_id", "intervention_type"]).size().unstack(fill_value=0)
    for itype, col in _INTERVENTION_TYPE_COLS.items():
        out[col] = iv_counts[itype].reindex(idx).fillna(0) if itype in iv_counts else 0.0
    iv_join = iv.groupby("nct_id").agg({"name": lambda s: "; ".join(sorted(set(s.dropna()))),
                                        "description": lambda s: "; ".join(sorted(set(s.dropna())))})
    out["intervention/intervention_name"] = iv_join["name"].reindex(idx)
    out["intervention/description"] = iv_join["description"].reindex(idx)
    del iv, iv_counts, iv_join

    kwds = load_table_rows("keywords", nct_ids, ["nct_id", "name"], **kw)
    out["keyword"] = kwds.groupby("nct_id")["name"].apply(lambda s: "; ".join(sorted(set(s.dropna())))).reindex(idx)
    del kwds

    fac = load_table_rows("facilities", nct_ids, ["nct_id", "city"], **kw).sort_values("city")
    out["location/facility/address/city"] = fac.drop_duplicates("nct_id").set_index("nct_id")["city"].reindex(idx)
    del fac

    cond = load_table_rows("conditions", nct_ids, ["nct_id", "name"], **kw)
    cond_lists = cond.groupby("nct_id")["name"].apply(list).reindex(idx).apply(
        lambda v: v if isinstance(v, list) else [])
    out["condition"] = cond_lists.apply(lambda v: repr(v) if v else None)
    del cond

    bcond = load_table_rows("browse_conditions", nct_ids, ["nct_id", "mesh_term"], **kw)
    out["condition_browse/mesh_term"] = bcond.groupby("nct_id")["mesh_term"].apply(
        lambda s: "; ".join(sorted(set(s.dropna())))).reindex(idx)
    del bcond

    if do_icdcode:
        out["icdcode"] = reproduce_icdcode_column(cond_lists)
    else:
        out["icdcode"] = pd.NA

    for c in ALWAYS_MISSING_COLUMNS:
        out[c] = pd.NA

    return out


def missing_rate_report(slice_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-rate comparison, slice vs. training distribution --
    P14.4's acceptance check. Columns present in ``train_df`` but not in
    ``slice_df`` (shouldn't happen; ``emit_trialbench_schema`` emits every
    known TrialBench column, missing or not) are reported at 100% missing.
    """
    rows = []
    for c in train_df.columns:
        train_rate = float(train_df[c].isna().mean())
        if c in slice_df.columns:
            slice_rate = float(slice_df[c].isna().mean())
        else:
            slice_rate = 1.0
        rows.append({
            "column": c, "train_missing_rate": train_rate, "slice_missing_rate": slice_rate,
            "delta": slice_rate - train_rate,
            "status": ("always_missing" if c in ALWAYS_MISSING_COLUMNS
                      else "approximate" if c in APPROXIMATE_COLUMNS else "reconstructed"),
        })
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)
