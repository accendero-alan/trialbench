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
