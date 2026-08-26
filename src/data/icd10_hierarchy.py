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
    """Build explicit ``char3 -> label`` maps for both chapter and block by
    walking the XML's actual ``<section>``/``<diag>`` tree rather than
    comparing a code against ``<sectionRef first="..." last="...">`` bounds
    as strings. String comparison mis-handles any code whose third character
    is a letter under a numerically-bounded section (``"M1A" > "M14"``
    because ``'A' > '4'``), silently dropping it. Every top-level ``<diag>``
    under a ``<section>`` is confirmed 3 characters (verified against this
    release), so ``diag/name`` is a direct, unambiguous key.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"{xml_path} not found. P9 needs the official ICD-10-CM Tabular List XML: "
            f"download icd10cm-<release>-XML.zip from "
            f"ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/<year>-update/ "
            f"and unzip the *-tabular-*.xml file to this path (or pass xml_path)."
        )
    root = ET.parse(xml_path).getroot()
    n_chapters = 0
    char3_to_chapter: dict[str, str] = {}
    char3_to_block: dict[str, str] = {}
    for chapter in root.findall("chapter"):
        desc = chapter.findtext("desc", default="").strip()
        m = _CHAPTER_DESC_RE.match(desc)
        if not m:
            continue
        n_chapters += 1
        chapter_label = f"{m.group(2)}-{m.group(3)}"
        for section in chapter.findall("section"):
            block_label = section.get("id")
            for diag in section.findall("diag"):
                code = (diag.findtext("name") or "").strip().upper()
                if len(code) != 3:
                    continue
                char3_to_chapter[code] = chapter_label
                char3_to_block[code] = block_label
    if n_chapters != 22:
        raise ValueError(f"expected 22 ICD-10-CM chapters in {xml_path}, parsed {n_chapters}")
    return char3_to_chapter, char3_to_block


def icd10_chapter(char3: str, xml_path: str = TABULAR_XML_PATH) -> str | None:
    """Map a 3-character ICD-10-CM prefix (e.g. ``"J45"``) to its chapter's
    range label (e.g. ``"J00-J99"``), or ``None`` if it matches no chapter."""
    char3_to_chapter, _char3_to_block = _load(xml_path)
    return char3_to_chapter.get(char3.strip().upper())


def icd10_block(char3: str, xml_path: str = TABULAR_XML_PATH) -> str | None:
    """Map a 3-character ICD-10-CM prefix to its block's range label (e.g.
    ``"J40-J47"``), or ``None`` if it matches no block. ~280 blocks are
    finer than the 22 chapters and coarser than the 509 char3 categories."""
    _char3_to_chapter, char3_to_block = _load(xml_path)
    return char3_to_block.get(char3.strip().upper())


@lru_cache(maxsize=1)
def _load_full_descriptors(xml_path: str = TABULAR_XML_PATH) -> dict[str, str]:
    """P13.6 (wave2-start-plan.md), the L3 arm: official descriptor text for
    every full (dotted) ICD-10-CM code, not just the char3 rollups ``_load``
    builds. The Tabular List XML nests ``<diag>`` recursively -- A00 -> A00.0,
    A00.1, ... -- so a full code's descriptor is only reachable by walking
    each section's diag tree to its leaves, not by a single top-level pass."""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"{xml_path} not found. See icd10_chapter's docstring for how to obtain it."
        )
    root = ET.parse(xml_path).getroot()
    out: dict[str, str] = {}

    def walk(diag):
        code = (diag.findtext("name") or "").strip().upper()
        desc = (diag.findtext("desc") or "").strip()
        if code and desc:
            out[code] = desc
        for child in diag.findall("diag"):
            walk(child)

    for chapter in root.findall("chapter"):
        for section in chapter.findall("section"):
            for diag in section.findall("diag"):
                walk(diag)
    return out


def icd10_full_descriptor(code: str, xml_path: str = TABULAR_XML_PATH) -> str | None:
    """Official descriptor for a full ICD-10-CM code (e.g. ``"E11.9"`` ->
    ``"Type 2 diabetes mellitus without complications"``), or ``None`` if the
    code (with or without its decimal point) isn't in the Tabular List."""
    descriptors = _load_full_descriptors(xml_path)
    key = code.strip().upper()
    return descriptors.get(key) or descriptors.get(key.replace(".", ""))
