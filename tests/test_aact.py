"""src/data/aact.py's results_posted_date(), unit-tested against the real
pinned AACT snapshot (docs/t28b_opus_recall_spec.md's blocking gap: Arm A's
partition needs a results-posting date, and nothing in the repo read one
before this). Needs data/external/aact_20260826/studies.txt present
locally -- skips cleanly if it isn't, rather than failing the whole suite
on a missing multi-hundred-MB data file.

Run:  python tests/test_aact.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.aact import SNAPSHOT_DIR, load_table, results_posted_date  # noqa: E402
from src.data.aact_slice import slice_ab_nct_ids  # noqa: E402

_SNAPSHOT_PRESENT = os.path.exists(os.path.join(SNAPSHOT_DIR, "studies.txt"))


def test_shape_and_dtype():
    s = results_posted_date()
    assert s.index.name == "nct_id"
    assert pd.api.types.is_datetime64_any_dtype(s.dtype)
    assert len(s) > 0
    print("shape+dtype OK:", s.shape, s.dtype)


def test_null_safe_not_fabricated():
    """A trial with no posted results must come back NaT, never a guessed
    date -- most of the snapshot's trials have no results (still running,
    or never reported), so this should be the common case, not an edge
    case."""
    s = results_posted_date()
    n_null, n_total = s.isna().sum(), len(s)
    assert n_null > 0, "expected some trials with no posted results -- got none, suspicious"
    assert n_null < n_total, "expected some trials WITH posted results -- got none, suspicious"
    print(f"null-safe OK: {n_null}/{n_total} null ({n_null / n_total:.1%})")


def test_distinct_from_registration_date():
    """results_first_posted_date and study_first_posted_date are different
    AACT columns; a bug that reads the wrong one would silently make Arm
    A's partition (docs/t28b_opus_recall_spec.md) meaningless. A results
    date can never precede its own trial's registration date -- that's a
    real invariant on the data, not a coincidence of this snapshot, and
    catches a column-swap bug directly rather than by comparing column
    names."""
    results = results_posted_date()
    studies = load_table("studies")[["nct_id", "study_first_posted_date"]]
    studies = studies.drop_duplicates("nct_id").set_index("nct_id")["study_first_posted_date"]
    registration = pd.to_datetime(studies, errors="coerce")

    both = pd.DataFrame({"reg": registration, "res": results}).dropna()
    assert len(both) > 0, "no trial has both dates -- can't check the invariant"
    n_backwards = int((both["res"] < both["reg"]).sum())
    assert n_backwards == 0, (
        f"{n_backwards} trial(s) have a results-posted date before their own registration "
        f"date -- either a genuine AACT data anomaly worth investigating, or (more likely) "
        f"results_posted_date() is reading the wrong column."
    )
    # Not every trial should have byte-identical reg/results dates -- if this
    # were ~100%, the two columns would be suspiciously indistinguishable.
    n_identical = int((both["res"] == both["reg"]).sum())
    assert n_identical < len(both) * 0.5, (
        f"{n_identical}/{len(both)} trials have identical registration and results dates -- "
        f"too many to be same-day coincidence; check the two columns aren't aliased."
    )
    print(f"distinct-from-registration OK: {len(both)} trials with both dates, "
         f"{n_backwards} backwards, {n_identical} identical")


def test_dedup_by_nct_id():
    """One value per nct_id -- a duplicated raw row must not silently
    produce two conflicting dates for the same trial."""
    s = results_posted_date()
    assert not s.index.duplicated().any()
    print("dedup OK: no duplicate nct_id in the result")


def test_slice_ab_matches_p14_5_n_gate():
    """docs/p14_5_n_gate.md's own documented counts for Claude Opus 4.5's
    2025-03 cutoff: slice (a) = 9,046, slice (b) = 208. That document's own
    numbers came from an ad-hoc script never checked in -- this is the
    first committed, reproducible version, so matching those exact counts
    (not just "close") is the test."""
    cutoff = pd.Period("2025-03", freq="M").end_time
    s = slice_ab_nct_ids(cutoff)
    assert len(s["a"]) == 9046, f"slice (a): expected 9046, got {len(s['a'])}"
    assert len(s["b"]) == 208, f"slice (b): expected 208, got {len(s['b'])}"
    assert len(set(s["a"]) & set(s["b"])) == 0, "slice (a) and (b) must be disjoint by construction"
    assert s["a"] == sorted(s["a"]) and s["b"] == sorted(s["b"]), "both slices should be sorted, deterministic"
    print(f"slice_ab OK: |a|={len(s['a'])}, |b|={len(s['b'])}, disjoint")


if __name__ == "__main__":
    if not _SNAPSHOT_PRESENT:
        print(f"SKIPPED: {SNAPSHOT_DIR}/studies.txt not present locally -- "
             "download and extract the pinned AACT snapshot first (see src/data/aact.py's "
             "module docstring for the URL/checksum) to run this test.")
    else:
        test_shape_and_dtype()
        test_null_safe_not_fabricated()
        test_distinct_from_registration_date()
        test_dedup_by_nct_id()
        test_slice_ab_matches_p14_5_n_gate()
        print("aact tests passed")
