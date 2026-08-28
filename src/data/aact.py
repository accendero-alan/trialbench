"""P14.1 (wave2-start-plan.md): the AACT fresh-slice builder's data layer --
snapshot pinning, table loaders with a schema check, and (P14.2) the label
recomputation functions, citing ``docs/aact_label_rule.md``'s section numbers
in every function's docstring.

Pinned snapshot (standing rule 10 -- record date, URL, checksum, quote in
every artifact that reads it)
------------------------------------------------------------------------
AACT daily flat-file export, 2026-08-26::

    URL:    https://aact.ctti-clinicaltrials.org/static/exported_files/daily/2026-08-26
    File:   20260826_export_ctgov.zip
    Size:   2,513,044,058 bytes
    sha256: d2ef247566ec080a4788f7556ab7fe1dad630e87af5a783f3664be0ddee75e57
    Format: pipe-delimited flat text files (not the PostgreSQL dump variant
            -- no local database server needed, loads directly with pandas).

Extracted (a focused subset, not all 49 tables -- the full archive is
~16.6 GB uncompressed) to ``data/external/aact_20260826/``: ``studies``,
``browse_conditions``, ``conditions``, ``reported_events``,
``reported_event_totals``, ``drop_withdrawals``, ``milestones``,
``eligibilities``, ``designs``, ``design_groups``, ``interventions``,
``keywords``, ``sponsors``, ``brief_summaries``, ``id_information``,
``countries``, ``facilities``, ``calculated_values``.

Not a live database: a fixed, dated, checksummed snapshot per P14.1 -- "the
slice must be reproducible six months from now when Part 3 is being
defended." A second machine reproduces the same row counts by downloading
the same URL and verifying the same checksum before extracting.

Table layout is asserted against this snapshot's own headers at load time
(``_verify_schema``), not trusted from memory or from this docstring, per
P14.1's requirement to verify against CTTI's current layout at build time.
"""
from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

SNAPSHOT_DATE = "2026-08-26"
SNAPSHOT_URL = "https://aact.ctti-clinicaltrials.org/static/exported_files/daily/2026-08-26"
SNAPSHOT_ZIP = os.path.join("data", "external", "20260826_export_ctgov.zip")
SNAPSHOT_ZIP_SHA256 = "d2ef247566ec080a4788f7556ab7fe1dad630e87af5a783f3664be0ddee75e57"
SNAPSHOT_ZIP_SIZE = 2_513_044_058
SNAPSHOT_DIR = os.path.join("data", "external", "aact_20260826")

# Columns this module actually reads from each table -- the schema check
# below asserts these are present, not that the table has exactly these
# columns (AACT tables carry more than we use).
_EXPECTED_COLUMNS = {
    "reported_event_totals": {"nct_id", "ctgov_group_code", "event_type", "subjects_affected", "subjects_at_risk"},
    "milestones": {"nct_id", "result_group_id", "title", "period", "count"},
    "drop_withdrawals": {"nct_id", "result_group_id", "period", "reason", "count"},
    "studies": {"nct_id", "phase", "overall_status", "study_first_posted_date", "results_first_posted_date"},
    "browse_conditions": {"nct_id", "mesh_term"},
    "conditions": {"nct_id", "name"},
    "eligibilities": {"nct_id", "gender", "minimum_age", "maximum_age", "healthy_volunteers", "criteria"},
    "designs": {"nct_id", "allocation", "intervention_model", "masking", "primary_purpose"},
}


def snapshot_info() -> dict:
    """The pinning record every P14 artifact should quote verbatim."""
    return {
        "source": "AACT (aact.ctti-clinicaltrials.org)", "format": "pipe-delimited flat files",
        "date": SNAPSHOT_DATE, "url": SNAPSHOT_URL,
        "zip_sha256": SNAPSHOT_ZIP_SHA256, "zip_size_bytes": SNAPSHOT_ZIP_SIZE,
    }


def _verify_schema(name: str, columns: list) -> None:
    expected = _EXPECTED_COLUMNS.get(name)
    if expected is None:
        return
    missing = expected - set(columns)
    if missing:
        raise ValueError(
            f"AACT table {name!r} is missing expected column(s) {sorted(missing)} in this "
            f"snapshot's header -- CTTI's table layout may have changed since "
            f"{SNAPSHOT_DATE}; re-verify against their current data dictionary before trusting "
            f"this module's column names."
        )


@lru_cache(maxsize=None)
def load_table(name: str, snapshot_dir: str = SNAPSHOT_DIR) -> pd.DataFrame:
    """One AACT table, pipe-delimited, from the extracted snapshot directory.
    Cached per (name, snapshot_dir) -- these tables are large (``reported_events``
    is ~5 GB uncompressed) and P14's slice-building reads several of them
    more than once."""
    path = os.path.join(snapshot_dir, f"{name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download {SNAPSHOT_ZIP} from {SNAPSHOT_URL} (verify sha256 "
            f"{SNAPSHOT_ZIP_SHA256}), then extract {name}.txt into {snapshot_dir}."
        )
    df = pd.read_csv(path, sep="|", low_memory=False)
    _verify_schema(name, list(df.columns))
    return df


def load_table_columns(name: str, columns, snapshot_dir: str = SNAPSHOT_DIR) -> pd.DataFrame:
    """Like :func:`load_table`, but reads only ``columns`` from disk --
    for the tables where a full read is a real memory problem, not just
    wasteful. Confirmed live (T28b's OOM, 2026-08-28): ``eligibilities.txt``
    is 900MB on disk with dozens of columns this repo never reads;
    ``brief_summaries.txt`` 421MB, ``studies.txt`` 407MB, and
    :func:`src.data.aact_slice.emit_trialbench_schema` loads several more
    full-width tables the same way -- loading all of them in full, and
    ``load_table``'s ``lru_cache`` keeping every one resident for the rest
    of the process, is what pushed a single ``python3`` process to 3.78GB
    resident and into the kernel OOM killer on a modest instance, before
    any model call.

    Verifies the on-disk schema first via a cheap header-only read
    (``nrows=0``), the same guarantee ``load_table`` gives, so a column
    this module expects going missing from a future snapshot is still
    caught -- narrowing ``usecols`` must not also narrow that check.

    Deliberately **not** cached: a caller that needs the same narrow slice
    repeatedly should hold onto the returned DataFrame itself. Caching every
    distinct ``columns`` tuple here would reproduce exactly the unbounded-
    retention problem this function exists to avoid.
    """
    path = os.path.join(snapshot_dir, f"{name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download {SNAPSHOT_ZIP} from {SNAPSHOT_URL} (verify sha256 "
            f"{SNAPSHOT_ZIP_SHA256}), then extract {name}.txt into {snapshot_dir}."
        )
    header = pd.read_csv(path, sep="|", nrows=0).columns.tolist()
    _verify_schema(name, header)
    columns = list(columns)
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"requested column(s) {missing} not in {name}'s on-disk header: {header}")
    return pd.read_csv(path, sep="|", usecols=columns, low_memory=False)


def load_table_rows(name: str, nct_ids, columns, snapshot_dir: str = SNAPSHOT_DIR,
                    chunksize: int = 50_000) -> pd.DataFrame:
    """Like :func:`load_table_columns`, but also filters to a small set of
    ``nct_ids`` while reading, via chunked reads -- for a lookup against a
    handful of trials out of a huge table (e.g. 50 ids out of
    ``brief_summaries``' ~600k rows, confirmed the dominant remaining
    contributor to T28b's OOM after narrowing columns everywhere else:
    even 2-3 columns of free-text ``description`` for 600k rows is real
    memory, and this function's whole job is to never hold more than
    ``chunksize`` rows of it at once). ``columns`` must include
    ``"nct_id"``.
    """
    if "nct_id" not in columns:
        raise ValueError("load_table_rows requires 'nct_id' in columns")
    path = os.path.join(snapshot_dir, f"{name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download {SNAPSHOT_ZIP} from {SNAPSHOT_URL} (verify sha256 "
            f"{SNAPSHOT_ZIP_SHA256}), then extract {name}.txt into {snapshot_dir}."
        )
    header = pd.read_csv(path, sep="|", nrows=0).columns.tolist()
    _verify_schema(name, header)
    columns = list(columns)
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"requested column(s) {missing} not in {name}'s on-disk header: {header}")

    wanted = set(nct_ids)
    matches = []
    for chunk in pd.read_csv(path, sep="|", usecols=columns, chunksize=chunksize, low_memory=False):
        hit = chunk[chunk["nct_id"].isin(wanted)]
        if len(hit):
            matches.append(hit)
    return pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=columns)


# ----------------------------------------------------------------------------
# P14.2 label recomputation -- docs/aact_label_rule.md
# ----------------------------------------------------------------------------
def _pooled_event_rate(event_type: str, snapshot_dir: str = SNAPSHOT_DIR) -> pd.DataFrame:
    """docs/aact_label_rule.md §1/§2: reported_event_totals, pooled sum of
    subjects_affected / subjects_at_risk across every ctgov_group_code (arm)
    per trial, filtered to ``event_type``. Returns a DataFrame indexed by
    nct_id with ``subjects_affected``, ``subjects_at_risk``, ``rate`` columns."""
    ret = load_table_columns("reported_event_totals",
                             ["nct_id", "ctgov_group_code", "event_type", "subjects_affected", "subjects_at_risk"],
                             snapshot_dir)
    ret = ret[ret["event_type"] == event_type]
    pooled = ret.groupby("nct_id")[["subjects_affected", "subjects_at_risk"]].sum()
    pooled["rate"] = pooled["subjects_affected"] / pooled["subjects_at_risk"].replace(0, pd.NA)
    return pooled


def mortality_yn(snapshot_dir: str = SNAPSHOT_DIR) -> pd.Series:
    """docs/aact_label_rule.md §1: at least one all-cause-mortality event
    reported, pooled across arms. Measured 99.87% agreement with
    TrialBench's mortality_rate_yn Y/N on the 17,915-trial overlap."""
    pooled = _pooled_event_rate("deaths", snapshot_dir)
    return (pooled["subjects_affected"] > 0).astype(int).rename("mortality_yn")


def sae_yn(snapshot_dir: str = SNAPSHOT_DIR) -> pd.Series:
    """docs/aact_label_rule.md §2: at least one serious-adverse-event
    reported, pooled across arms. Measured 99.94% agreement with
    TrialBench's serious_adverse_rate_yn Y/N on the 17,915-trial overlap."""
    pooled = _pooled_event_rate("serious", snapshot_dir)
    return (pooled["subjects_affected"] > 0).astype(int).rename("sae_yn")


def dropout_yn(snapshot_dir: str = SNAPSHOT_DIR) -> pd.Series:
    """docs/aact_label_rule.md §3: at least one participant recorded as
    NOT COMPLETED (period='Overall Study'), pooled across arms. Measured
    99.96% agreement with TrialBench's patient_dropout_rate_yn Y/N on the
    32,050-trial overlap."""
    mil = load_table_columns("milestones", ["nct_id", "result_group_id", "title", "period", "count"], snapshot_dir)
    overall = mil[mil["period"] == "Overall Study"]
    pivot = overall.groupby(["nct_id", "title"])["count"].sum().unstack(fill_value=0)
    not_completed = pivot.get("NOT COMPLETED", pd.Series(0, index=pivot.index, dtype=float))
    return (not_completed > 0).astype(int).rename("dropout_yn")


def results_posted_date(snapshot_dir: str = SNAPSHOT_DIR) -> pd.Series:
    """``results_first_posted_date`` per nct_id, parsed to datetime, from the
    pinned AACT snapshot -- null (``NaT``) for any trial with no results
    posted yet, never a fabricated date.

    Distinct from ``study_first_posted_date`` (registration date, what
    ``experiments/t28a_contamination_probes.py``'s
    ``_join_registration_dates`` reads): that field says when a trial's
    *existence* became public, this one says when its *results* did. A
    model's training cutoff can only have seen a reported outcome if the
    results-posting date precedes the cutoff -- registration date alone
    only bounds whether the model could have seen the trial exists,
    which is a different and weaker claim
    (docs/t28b_opus_recall_spec.md's Arm A/B/C partition turns on this
    distinction; ``docs/p14_5_n_gate.md``'s option 1 framing is the same
    idea applied as a slice-selection rule rather than an instrument).

    Deduplicated by nct_id before parsing (matching
    ``_join_registration_dates``'s defensive pattern) since AACT does not
    itself guarantee one row per trial in every export."""
    studies = load_table_columns("studies", ["nct_id", "results_first_posted_date"], snapshot_dir)
    studies = studies.drop_duplicates("nct_id").set_index("nct_id")["results_first_posted_date"]
    return pd.to_datetime(studies, errors="coerce").rename("results_posted_date")
