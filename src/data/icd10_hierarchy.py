"""ICD-10-CM chapter and block boundaries, for the P9 granularity ladder
(disease-representation-test-plan.md, T23/T24).

Both are parsed at runtime from the official CDC/NCHS Tabular List XML
(``data/external/icd10c-tabular-April-1-2026.xml``, from
``icd10cm-April-1-2026-XML.zip`` at
ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2026-update/),
not hand-transcribed: an early hand-typed version of the chapter table (built
from web search results during this session, cross-checked against two
independent secondary sources) turned out to have one wrong boundary --
chapter 17 ends at ``QA0``, not ``Q99`` -- that only the primary source
caught. Every downstream table -- chapter, and the ~280 finer "block"
ranges (``<sectionRef>`` in the XML, nested one level inside each
``<chapter>``) -- is parsed from the same file so there's one source of
truth and no place for a second transcription error to hide.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache

TABULAR_XML_PATH = os.path.join("data", "external", "icd10c-tabular-April-1-2026.xml")

_CHAPTER_DESC_RE = re.compile(r"^(.*)\s+\(([A-Z0-9]+)-([A-Z0-9]+)\)\s*$")


@lru_cache(maxsize=1)
def _load(xml_path: str = TABULAR_XML_PATH):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"{xml_path} not found. P9 needs the official ICD-10-CM Tabular List XML: "
            f"download icd10cm-<release>-XML.zip from "
            f"ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/<year>-update/ "
            f"and unzip the *-tabular-*.xml file to this path (or pass xml_path)."
        )
    root = ET.parse(xml_path).getroot()
    chapters, blocks = [], []
    for chapter in root.findall("chapter"):
        desc = chapter.findtext("desc", default="").strip()
        m = _CHAPTER_DESC_RE.match(desc)
        if not m:
            continue
        title, start, end = m.group(1), m.group(2), m.group(3)
        chapters.append((start, end, title))
        for ref in chapter.findall("./sectionIndex/sectionRef"):
            blocks.append((ref.get("first"), ref.get("last"), ref.get("id"),
                            (ref.text or "").strip()))
    if len(chapters) != 22:
        raise ValueError(f"expected 22 ICD-10-CM chapters in {xml_path}, parsed {len(chapters)}")
    return tuple(chapters), tuple(blocks)


def _range_lookup(char3: str, ranges) -> str | None:
    c = char3.strip().upper()
    for start, end, *_rest in ranges:
        if start <= c <= end:
            return f"{start}-{end}"
    return None


def icd10_chapter(char3: str, xml_path: str = TABULAR_XML_PATH) -> str | None:
    """Map a 3-character ICD-10-CM prefix (e.g. ``"J45"``) to its chapter's
    range label (e.g. ``"J00-J99"``), or ``None`` if it matches no chapter."""
    chapters, _blocks = _load(xml_path)
    return _range_lookup(char3, chapters)


def icd10_block(char3: str, xml_path: str = TABULAR_XML_PATH) -> str | None:
    """Map a 3-character ICD-10-CM prefix to its block's range label (e.g.
    ``"J40-J47"``), or ``None`` if it matches no block. ~280 blocks are
    finer than the 22 chapters and coarser than the 509 char3 categories."""
    _chapters, blocks = _load(xml_path)
    return _range_lookup(char3, blocks)
