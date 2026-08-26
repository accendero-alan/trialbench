"""ICD-10-CM -> CCSR (Clinical Classifications Software Refined) crosswalk,
for the P9 granularity ladder's ``ccsr`` rung
(disease-representation-test-plan.md, T23).

Parsed at runtime from the HCUP DXCCSR csv
(``data/external/DXCCSR_v2026-1.csv``, from ``DXCCSR-v2026-1.zip`` at
hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp). Free, no license required
(P9 standing rule 10: pin the version -- v2026.1, in the filename and this
module's default path).
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache

DXCCSR_CSV_PATH = os.path.join("data", "external", "DXCCSR_v2026-1.csv")

# Columns holding *all* CCSR categories a code maps to -- most codes map to
# one, some to several. "Default CCSR CATEGORY IP/OP" (columns 2 and 4) is a
# single context-dependent pick for cost/utilization studies; using only
# that would silently drop the categories it didn't pick.
_CATEGORY_COLS = (6, 8, 10, 12, 14, 16)


def _unquote(field: str) -> str:
    return field.strip().strip("'").strip()


@lru_cache(maxsize=1)
def _load(csv_path: str = DXCCSR_CSV_PATH) -> dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. P9's 'ccsr' granularity needs the HCUP DXCCSR "
            f"crosswalk: download the current DXCCSR zip from "
            f"hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp and copy its csv here "
            f"(or pass csv_path)."
        )
    mapping = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) <= max(_CATEGORY_COLS):
                continue
            code = _unquote(row[0])
            cats = {_unquote(row[i]) for i in _CATEGORY_COLS}
            cats.discard("")
            mapping[code] = sorted(cats)
    return mapping


def icd10_ccsr(code: str, csv_path: str = DXCCSR_CSV_PATH) -> list:
    """CCSR categories for a full ICD-10-CM code (e.g. ``"J45.909"``), or
    ``[]`` if the code isn't in the crosswalk. CCSR is keyed at the
    full-code level (dot removed) -- unlike chapter/block, it isn't
    derivable from a char3 rollup."""
    key = code.strip().upper().replace(".", "")
    return _load(csv_path).get(key, [])


# Category id -> description columns are one to the right of each of
# _CATEGORY_COLS in the same row (e.g. col 6 "CCSR CATEGORY 1" / col 7
# "CCSR CATEGORY 1 DESCRIPTION").
_CATEGORY_NAME_COLS = tuple(c + 1 for c in _CATEGORY_COLS)


@lru_cache(maxsize=1)
def _load_names(csv_path: str = DXCCSR_CSV_PATH) -> dict:
    """P13.6, the L5 arm: CCSR category *names* (e.g. ``"END011"`` ->
    ``"Diabetes mellitus"``), not just codes -- ``L5`` renders category
    names into the disease slot, and this is the crosswalk's own source for
    them rather than a second, hand-maintained code->name table that could
    drift from the pinned DXCCSR version."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. See icd10_ccsr's docstring for how to obtain it."
        )
    names: dict = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) <= max(_CATEGORY_NAME_COLS):
                continue
            for code_col, name_col in zip(_CATEGORY_COLS, _CATEGORY_NAME_COLS):
                code = _unquote(row[code_col])
                name = row[name_col].strip()
                if code and name:
                    names[code] = name
    return names


def icd10_ccsr_name(category: str, csv_path: str = DXCCSR_CSV_PATH) -> str | None:
    """The human-readable name for a CCSR category id (e.g.
    ``"DIG001"`` -> ``"Intestinal infection"``), or ``None`` if unknown."""
    return _load_names(csv_path).get(_unquote(category))
