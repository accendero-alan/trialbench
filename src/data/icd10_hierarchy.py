"""ICD-10-CM chapter boundaries, for the P9 granularity ladder
(disease-representation-test-plan.md, T23).

The 22 chapters are a fixed part of the classification's design and, unlike
``block`` (~280 category ranges) or CCSR (~540 categories, HCUP crosswalk),
require no external file: chapter membership is a pure function of a code's
3-character prefix. Cross-checked against two independent official sources
(CMS FY2026 ICD-10-CM Official Guidelines for Coding and Reporting; HCUP
Statistical Brief #248, Table 3) — see disease-representation-test-plan.md P9.

``block`` and ``ccsr`` granularities are NOT implemented here: block ranges
and CCSR categories both require an authoritative external reference file
(the ICD-10-CM Tabular List / HCUP CCSR crosswalk respectively) that hasn't
been pulled into ``data/external/`` yet. Hand-reconstructing ~280 block
ranges from web sources proved unreliable during research for this module
(sources disagreed and silently dropped rows) — exactly the kind of error
the plan's "get this exactly right" standard rules out.
"""
from __future__ import annotations

# (start_char3, end_char3, title) -- ranges sort correctly under plain string
# comparison because ICD-10-CM's letter-then-digit prefixes are designed to,
# including "U00-U85" (chapter 22) sitting lexicographically between "T88"
# and "V00" despite being numbered last.
ICD10_CHAPTERS = (
    ("A00", "B99", "Certain infectious and parasitic diseases"),
    ("C00", "D49", "Neoplasms"),
    ("D50", "D89", "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"),
    ("E00", "E89", "Endocrine, nutritional and metabolic diseases"),
    ("F01", "F99", "Mental, Behavioral and Neurodevelopmental disorders"),
    ("G00", "G99", "Diseases of the nervous system"),
    ("H00", "H59", "Diseases of the eye and adnexa"),
    ("H60", "H95", "Diseases of the ear and mastoid process"),
    ("I00", "I99", "Diseases of the circulatory system"),
    ("J00", "J99", "Diseases of the respiratory system"),
    ("K00", "K95", "Diseases of the digestive system"),
    ("L00", "L99", "Diseases of the skin and subcutaneous tissue"),
    ("M00", "M99", "Diseases of the musculoskeletal system and connective tissue"),
    ("N00", "N99", "Diseases of the genitourinary system"),
    ("O00", "O9A", "Pregnancy, childbirth and the puerperium"),
    ("P00", "P96", "Certain conditions originating in the perinatal period"),
    ("Q00", "Q99", "Congenital malformations, deformations and chromosomal abnormalities"),
    ("R00", "R99", "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"),
    ("S00", "T88", "Injury, poisoning and certain other consequences of external causes"),
    ("V00", "Y99", "External causes of morbidity"),
    ("Z00", "Z99", "Factors influencing health status and contact with health services"),
    ("U00", "U85", "Codes for special purposes"),
)


def icd10_chapter(char3: str) -> str | None:
    """Map a 3-character ICD-10-CM prefix (e.g. ``"J45"``) to its chapter's
    range label (e.g. ``"J00-J99"``), or ``None`` if it matches no chapter."""
    c = char3.strip().upper()
    for start, end, _title in ICD10_CHAPTERS:
        if start <= c <= end:
            return f"{start}-{end}"
    return None
