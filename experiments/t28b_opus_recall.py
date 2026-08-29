"""T28b -- Is Opus 4.5's predictive signal recall or reading?
(docs/t28b_opus_recall_spec.md)

T28a found Opus 4.5 discriminating well above chance on three of four tasks
with the full serialized trial in the prompt, and title recall (NCT ID ->
title) did not beat its shuffled-ID floor. Title recall asks for a title
from a bare ID; the outcome probe hands the model the whole trial. A null on
title recall does not exclude description-recognition followed by outcome
recall -- this test closes that gap directly, by holding trial identity
constant while removing whether the *result* could have been read.

Three arms, partitioned by when a trial's RESULTS became public relative to
Opus's 2025-03 training cutoff, not by whether the trial existed:

    A  TrialBench trials, results posted before cutoff  -- known, memorisable
    B  P14 slice (a): registered pre-cutoff, results post-cutoff -- known, NOT memorisable
    C  P14 slice (b): registered and resulted post-cutoff -- unknown

A -> B is the decisive contrast: it holds identity knowledge constant and
removes only outcome knowledge. Every arm is also scored by a frozen
TF-IDF+LogReg reference (fit once on TrialBench's training split, no memory
of anything) so any A->B drop can be read against how much of it is
distribution shift rather than lost recall.

The primary A/B/C reference is restricted to R1_TEXT_COLS below, not
src.methods.text_nlp.TfidfLogReg's default TEXT_COLS, which includes
brief_title/brief_summary/condition -- three of the five DISEASE_LEAK_COLS
(src/data/serialize.py) -- since taking the default would let the
"cannot be recalling" reference silently read exactly the
disease-identity signal a disease-swap probe needs to control for. The
disease-swap arm itself, and its own matched reference, moved out of this
module entirely (run_l0_null_check below, docs/t28b_l0_implementation_plan.md)
after the original swap arm turned out to be invalid as rendered.

Usage:
    python -m experiments.t28b_opus_recall --n-arm-a 500 --n-arm-b 500 \\
        --model anthropic.claude-opus-4-5
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.bedrock.client import BedrockClient, elicit_probability
from src.bedrock.meter import Meter
from src.bedrock.prices import is_verified, load_price_table, resolve_model_id
from src.data.aact import load_table_rows, mortality_yn, sae_yn
from src.data.aact_slice import emit_trialbench_schema, slice_ab_nct_ids
from src.data.features import _recursive_parse_terms, concat_text
from src.data.icd10_hierarchy import icd10_chapter
from src.data.loader import load_task_phase
from src.data.serialize import (
    BODY_COLS,
    DISEASE_LEAK_COLS,
    _SCRUB_COLS,
    _fmt,
    render_arm,
    render_arm_with_disease_override,
)
from src.eval.pooled_bootstrap import (
    diff_in_diff_bootstrap,
    one_sample_cluster_bootstrap,
    pooled_paired_bootstrap,
    two_sample_cluster_bootstrap,
)
from src.methods.llm import TASK_QUESTIONS, _build_verbalized_prompt
from src.methods.text_nlp import _TfidfLogRegBase

OUT_PATH = "results/experiments/t28b_opus_recall.json"
L0_OUT_PATH = "results/experiments/t28b_l0_null.json"
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]

# Endpoint -> (TrialBench task-folder-independent loader.py task key,
# P14's AACT-recomputed label function). Only these two: P14 has no
# recomputation rule for `outcome` (Opus's third significant T28a task).
ENDPOINTS = {
    "mortality_rate_yn": mortality_yn,
    "serious_adverse_rate_yn": sae_yn,
}

# R1 (distribution-shift calibration, all three arms): restricted to fields
# that reconstruct verbatim from AACT (docs/p14_4_schema_slice.md: phase
# 78.1% missing on the fresh slice, responsible_party 100%, icdcode 51%
# reconstructed -- a tabular/code reference would be reading those gaps).
# Disease leakage is NOT a concern here; R1 only has to be a memory-free
# predictor, and reading the disease is exactly what tfidf_logreg does
# elsewhere in this campaign (T22/T25).
R1_TEXT_COLS = ("condition", "brief_summary/textblock", "eligibility/criteria/textblock")

# The disease-swap arm's own reference is no longer a text_cols-restricted
# tfidf_logreg -- docs/t28b_l0_implementation_plan.md P3: text_cols=
# ("condition",) made the reference a pure disease model, so a swap
# changed 100% of its input while changing one line of Opus's, and that
# input-composition asymmetry (not memorisation) explained most of the
# gap. See _RenderedTextTfidfLogReg below: fit on render_arm's actual L1
# string instead, for exact input parity with Opus.

DISEASE_SWAP_MOVE_THRESHOLD = 0.05

OPUS_MODEL_KEY = "anthropic.claude-opus-4-5"

# docs/t28b_reanalysis_plan.md R1: balanced_accuracy thresholds the
# verbalized probability at a hard 0.5, and verbalized probabilities are
# usually badly calibrated (LLM_CONTAMINATION_PLAN.md Sec1, disqualifier 3)
# -- a fixed threshold interacting with a base-rate shift between arms can
# manufacture a "drop" that is a calibration artifact, not lost
# discrimination. AUROC/PR-AUC score the same continuous probabilities
# without a threshold, so they're what R1/R2 actually decide the verdict
# on; balanced_accuracy is still computed and reported (at this recorded
# threshold), but moved to a secondary column, per the reanalysis plan's
# explicit instruction.
BALANCED_ACCURACY_THRESHOLD = 0.5
DECISION_METRIC = "auroc"
REPORTED_METRICS = ("auroc", "prauc", "balanced_accuracy")

# T28a's per-task point estimates (results/experiments/t28a_probe_gate.json,
# git_sha 1f68a6d7's lineage) -- what funded T28b and what R4 checks T28b's
# Arm A against.
T28A_OPUS_POINT_ESTIMATES = {
    "mortality_rate_yn": 0.829,
    "serious_adverse_rate_yn": 0.769,
}


# ----------------------------------------------------------------------------
# The frozen reference -- one per (endpoint, text_cols configuration)
# ----------------------------------------------------------------------------
class _FrozenTfidfLogReg(_TfidfLogRegBase):
    """`_TfidfLogRegBase` (src/methods/text_nlp.py) with an explicit,
    caller-chosen `text_cols` instead of `TfidfLogReg`'s hardcoded default
    -- see module docstring for why the default is unsafe here."""

    def __init__(self, text_cols, task_type="binary", num_classes=2, seed=42):
        super().__init__(task_type=task_type, num_classes=num_classes, seed=seed)
        self.text_cols = text_cols

    def _texts(self, X):
        return concat_text(X, text_cols=self.text_cols)


def fit_frozen_reference(endpoint: str, text_cols, data_root: str, seed: int) -> _FrozenTfidfLogReg:
    """Fit once on TrialBench's own training split for `endpoint`, pooled
    across all four phases (train only -- never valid/test, so no arm this
    reference scores was seen during fit). Frozen afterward: every arm
    (including P14-slice rows from a different data source entirely) is
    scored with this same fitted vectorizer+classifier, never refit."""
    X_parts, y_parts = [], []
    for phase in PHASES:
        try:
            td = load_task_phase(data_root, endpoint, phase, seed=seed)
        except FileNotFoundError:
            continue
        X_parts.append(td.X_train)
        y_parts.append(td.y_train)
    if not X_parts:
        raise FileNotFoundError(f"no TrialBench train data found for {endpoint!r} under {data_root!r}")
    X_train = pd.concat(X_parts, ignore_index=True)
    y_train = np.concatenate(y_parts)
    method = _FrozenTfidfLogReg(text_cols=text_cols, seed=seed)
    method.fit(X_train, y_train)
    return method


# ----------------------------------------------------------------------------
# Arm A -- TrialBench's own test split, results posted before cutoff,
# P14's recomputed label (not TrialBench's own Y/N -- see module docstring)
# ----------------------------------------------------------------------------
def _results_dates_for(nct_ids, snapshot_dir=None) -> pd.Series:
    from src.data.aact import results_posted_date
    kw = {} if snapshot_dir is None else {"snapshot_dir": snapshot_dir}
    all_dates = results_posted_date(**kw)
    return all_dates.reindex(pd.Index(nct_ids, name="nct_id"))


def _largest_remainder_alloc(sizes: dict, total: int) -> dict:
    """Same technique src/data/subset.py's label-stratification and
    experiments/t28a_contamination_probes.py's `_pool_trials` use: allocate
    `total` across strata proportional to `sizes`, largest-remainder to
    hit the exact total rather than rounding error."""
    keys = list(sizes)
    counts = pd.Series(sizes, dtype=float)
    if counts.sum() == 0:
        return {k: 0 for k in keys}
    raw = counts / counts.sum() * total
    alloc = np.floor(raw).astype(int)
    shortfall = total - alloc.sum()
    remainders = raw - alloc
    for k in remainders.sort_values(ascending=False).index[:shortfall]:
        alloc[k] += 1
    return {k: min(int(alloc[k]), int(sizes[k])) for k in keys}


def sample_arm_a(data_root: str, cutoff: pd.Timestamp, n_total: int, seed: int) -> pd.DataFrame:
    """TrialBench test-split rows for mortality_rate_yn/serious_adverse_rate_yn,
    pooled across phases, restricted to results_posted_date < cutoff,
    labelled with P14's recomputed rule (constant label definition across
    all three arms -- see module docstring), stratified by (endpoint, label)
    via largest-remainder to n_total.
    """
    rows = []
    for endpoint, label_fn in ENDPOINTS.items():
        p14_labels = label_fn()
        for phase in PHASES:
            try:
                td = load_task_phase(data_root, endpoint, phase, seed=seed)
            except FileNotFoundError:
                continue
            df = td.X_test.copy()
            df["nct_id"] = df.index.astype(str)
            df["endpoint"] = endpoint
            df["phase"] = phase
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no TrialBench test data found under {data_root!r}")
    pooled = pd.concat(rows, ignore_index=True)

    results_dates = _results_dates_for(pooled["nct_id"].tolist())
    pooled["results_posted_date"] = results_dates.values
    pooled = pooled[pooled["results_posted_date"] < cutoff].copy()

    p14_by_endpoint = {ep: fn() for ep, fn in ENDPOINTS.items()}
    pooled["p14_label"] = [
        p14_by_endpoint[ep].get(nct, np.nan) for ep, nct in zip(pooled["endpoint"], pooled["nct_id"])
    ]
    pooled = pooled.dropna(subset=["p14_label"]).copy()
    pooled["p14_label"] = pooled["p14_label"].astype(int)
    pooled["stratum"] = pooled["endpoint"] + "|" + pooled["p14_label"].astype(str)

    rng = np.random.default_rng(seed)
    strata_sizes = pooled.groupby("stratum", observed=True).size().to_dict()
    alloc = _largest_remainder_alloc(strata_sizes, n_total)
    chosen = []
    for stratum, n in alloc.items():
        if n <= 0:
            continue
        pool = pooled[pooled["stratum"] == stratum]
        idx = rng.choice(pool.index.values, size=min(n, len(pool)), replace=False)
        chosen.append(pool.loc[idx])
    sample = pd.concat(chosen, ignore_index=True) if chosen else pooled.iloc[0:0]
    return sample.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Arm B / C -- P14 slices, reconstructed to TrialBench schema, P14 labels
# ----------------------------------------------------------------------------
def sample_slice_arm(slice_nct_ids: list, n_total: "int | None", seed: int, snapshot_dir=None) -> pd.DataFrame:
    """`n_total=None` takes every id (Arm C: all of slice (b)). Otherwise
    stratifies by (endpoint, label) to `n_total`, same technique as Arm A.
    One row per (nct_id, endpoint) -- a trial with both labels computable
    contributes to both endpoints' pools, matching Arm A's per-endpoint
    framing (each is a separate probe with its own prompt)."""
    kw = {} if snapshot_dir is None else {"snapshot_dir": snapshot_dir}
    p14_by_endpoint = {ep: fn(**kw) for ep, fn in ENDPOINTS.items()}

    per_endpoint = []
    for endpoint, labels in p14_by_endpoint.items():
        ids = [n for n in slice_nct_ids if n in labels.index and pd.notna(labels.get(n))]
        if not ids:
            continue
        sub = pd.DataFrame({"nct_id": ids})
        sub["endpoint"] = endpoint
        sub["p14_label"] = labels.reindex(ids).astype(int).values
        per_endpoint.append(sub)
    if not per_endpoint:
        return pd.DataFrame(columns=["nct_id", "endpoint", "p14_label"])
    pooled = pd.concat(per_endpoint, ignore_index=True)
    pooled["stratum"] = pooled["endpoint"] + "|" + pooled["p14_label"].astype(str)

    rng = np.random.default_rng(seed)
    if n_total is None:
        chosen_ids = pooled
    else:
        strata_sizes = pooled.groupby("stratum", observed=True).size().to_dict()
        alloc = _largest_remainder_alloc(strata_sizes, n_total)
        parts = []
        for stratum, n in alloc.items():
            if n <= 0:
                continue
            pool = pooled[pooled["stratum"] == stratum]
            idx = rng.choice(pool.index.values, size=min(n, len(pool)), replace=False)
            parts.append(pool.loc[idx])
        chosen_ids = pd.concat(parts, ignore_index=True) if parts else pooled.iloc[0:0]

    rows_by_nct = emit_trialbench_schema(sorted(set(chosen_ids["nct_id"])), snapshot_dir=snapshot_dir,
                                         do_icdcode=False)
    merged = chosen_ids.merge(rows_by_nct, left_on="nct_id", right_index=True, how="inner")
    return merged.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Pre-flight checks -- zero model calls, all blocking (spec section)
# ----------------------------------------------------------------------------
def preflight_text_identity(n_check: int = 50, data_root: str = "data", seed: int = 42,
                            snapshot_dir=None) -> dict:
    """TrialBench's brief_summary/textblock vs AACT's brief_summaries.description
    for the same trials, verbatim. Both derive from the same upstream
    record -- if they differ systematically, the A->B contrast would be
    reading a reconstruction difference, not a recall difference."""
    tb_rows = []
    for phase in PHASES:
        try:
            td = load_task_phase(data_root, "mortality_rate_yn", phase, seed=seed)
        except FileNotFoundError:
            continue
        df = td.X_test[["brief_summary/textblock"]].copy()
        df["nct_id"] = df.index.astype(str)
        tb_rows.append(df)
    tb = pd.concat(tb_rows, ignore_index=True).drop_duplicates("nct_id") if tb_rows else pd.DataFrame()
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(tb["nct_id"].values, size=min(n_check, len(tb)), replace=False) if len(tb) else []

    kw = {} if snapshot_dir is None else {"snapshot_dir": snapshot_dir}
    # load_table_rows, not a full load: brief_summaries is 600k rows of free
    # text (~421MB on disk) for a lookup against ~50 ids -- confirmed the
    # dominant remaining contributor to T28b's OOM (2026-08-28) even after
    # narrowing every other table's columns; a full read here, cached
    # forever by load_table's lru_cache, pinned ~4.2GB resident on its own.
    bs = load_table_rows("brief_summaries", sample_ids, ["nct_id", "description"], **kw)
    bs = bs.set_index("nct_id")["description"]
    tb_indexed = tb.set_index("nct_id")["brief_summary/textblock"]

    def _normalize(text) -> str:
        # Confirmed-benign, systematic formatting differences between the
        # two pipelines, never a content difference -- checked by hand on
        # every mismatch sampled during development, not assumed:
        # (1) TrialBench keeps raw \r\n-wrapped whitespace from the
        # original XML, AACT's copy is whitespace-normalized -- collapse
        # all whitespace runs to single spaces. (2) AACT's pipe-delimited
        # export encodes an embedded paragraph break as a literal "~" (to
        # avoid a real newline breaking the row-oriented file format)
        # where TrialBench collapses it to a space (NCT02563561). (3) AACT
        # backslash-escapes markdown-significant characters TrialBench
        # leaves bare: brackets (NCT04399551, "\[PSP\]"), a line-leading
        # hyphen bullet (NCT01248455, "\- To evaluate"), and angle brackets
        # (NCT01844765, "\<18"; NCT00640159, "p \< 0.001") -- a regex
        # covering the standard markdown-escapable set, rather than adding
        # one more literal .replace() per character discovered by sampling,
        # since more will exist that this sample didn't happen to hit.
        # (4) AACT uses "*" for a bullet marker where TrialBench uses "-"
        # for the identical list (NCT01727414, 1933 chars, word-for-word
        # identical apart from the marker). (5) AACT strips literal
        # double-quote characters TrialBench keeps around an emphasized
        # word (NCT00640159, `"OFF"` vs `OFF`) -- drop both sides' quote
        # marks entirely rather than guess which side is "original."
        # (6) AACT also escapes "^" (superscript markdown, NCT02747043:
        # "mg/m\^2" vs "mg/m^2").
        t = str(text).replace("~", " ").replace('"', "").replace(" * ", " - ")
        t = re.sub(r"\\([\[\]()*_`<>#+.!^-])", r"\1", t)
        return " ".join(t.split())

    n_identical, n_checked, mismatches = 0, 0, []
    for nct in sample_ids:
        tb_text = tb_indexed.get(nct)
        aact_text = bs.get(nct)
        if tb_text is None or aact_text is None or pd.isna(tb_text) or pd.isna(aact_text):
            continue
        n_checked += 1
        if _normalize(tb_text) == _normalize(aact_text):
            n_identical += 1
        elif len(mismatches) < 3:
            mismatches.append(nct)
    return {"n_checked": n_checked, "n_identical": n_identical, "example_mismatches": mismatches}


def preflight_n_and_label_balance(arm_a, arm_b, arm_c) -> dict:
    def _summary(df, name):
        if len(df) == 0:
            return {"n": 0}
        out = {"n": len(df)}
        for endpoint in ENDPOINTS:
            sub = df[df["endpoint"] == endpoint]
            out[endpoint] = {"n": len(sub), "positive_rate": float(sub["p14_label"].mean()) if len(sub) else None}
        return out
    return {"arm_a": _summary(arm_a, "A"), "arm_b": _summary(arm_b, "B"), "arm_c": _summary(arm_c, "C")}


def preflight_covariates(arm_a, arm_b, arm_c) -> dict:
    """Report only -- do not match on these, the frozen reference absorbs
    distribution shift. Flag anything that separates the arms starkly."""
    def _stats(df):
        if len(df) == 0 or "brief_summary/textblock" not in df.columns:
            return {}
        token_len = df["brief_summary/textblock"].fillna("").astype(str).str.split().str.len()
        out = {"summary_token_len_median": float(token_len.median()) if len(token_len) else None}
        for col in ("phase", "enrollment"):
            if col in df.columns:
                if col == "enrollment":
                    vals = pd.to_numeric(df[col], errors="coerce")
                    out["enrollment_median"] = float(vals.median()) if vals.notna().any() else None
                else:
                    out["phase_distribution"] = df[col].value_counts(normalize=True).round(3).to_dict()
        return out
    return {"arm_a": _stats(arm_a), "arm_b": _stats(arm_b), "arm_c": _stats(arm_c)}


def preflight_power(arm_a, arm_b, seed: int, n_resamples: int = 500) -> dict:
    """Smallest A->B balanced-accuracy drop detectable at the chosen n, via
    a synthetic-effect check: inject a known drop into Arm A's own labels
    (relabel a fraction to look like chance) and see the smallest injected
    drop whose two_sample_cluster_bootstrap CI excludes 0. Cheap since it's
    pure resampling, no model calls."""
    if len(arm_a) == 0 or len(arm_b) == 0:
        return {"detectable_drop": None, "note": "empty arm(s), power check skipped"}
    rng = np.random.default_rng(seed)
    nct_a = arm_a["nct_id"].values
    y_a = arm_a["p14_label"].values.astype(int)
    nct_b = arm_b["nct_id"].values
    y_b = arm_b["p14_label"].values.astype(int)
    # A synthetic "perfect" proba (== label, i.e. balanced accuracy 1.0) on
    # A, degraded by flipping a fraction `frac` of A's scores toward 0.5.
    base_proba_a = y_a.astype(float)
    for frac in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        n_flip = int(round(frac * len(base_proba_a)))
        flip_idx = rng.choice(len(base_proba_a), size=n_flip, replace=False)
        proba_a = base_proba_a.copy()
        proba_a[flip_idx] = 0.5
        proba_b = y_b.astype(float)  # B unchanged reference point (perfect)
        r = two_sample_cluster_bootstrap(nct_a, y_a, proba_a, nct_b, y_b, proba_b,
                                         metric="balanced_accuracy", n_resamples=n_resamples, seed=seed)
        if r["lo"] is not None and not np.isnan(r["lo"]) and r["hi"] < 1.0:
            # detectable once the CI no longer touches the ceiling
            return {"detectable_drop_frac_relabelled": frac, "n_arm_a": len(arm_a), "n_arm_b": len(arm_b)}
    return {"detectable_drop_frac_relabelled": None, "n_arm_a": len(arm_a), "n_arm_b": len(arm_b),
           "note": "power check inconclusive at tested fractions -- raise n"}


def run_preflight(data_root: str, cutoff: pd.Timestamp, n_arm_a: int, n_arm_b: int, seed: int,
                  snapshot_dir=None, n_arm_c=None) -> dict:
    """`n_arm_c=None` (the spec default) takes every trial in slice (b) --
    up to 416 rows (208 trials x 2 endpoints), which is real API spend on
    a live run. Pass an int to cap it for a cheap smoke test; the real
    invocation should leave it at None ("all 208")."""
    slices = slice_ab_nct_ids(cutoff, snapshot_dir=snapshot_dir)
    arm_a = sample_arm_a(data_root, cutoff, n_arm_a, seed)
    arm_b = sample_slice_arm(slices["a"], n_arm_b, seed, snapshot_dir=snapshot_dir)
    arm_c = sample_slice_arm(slices["b"], n_arm_c, seed, snapshot_dir=snapshot_dir)
    return {
        "text_identity": preflight_text_identity(data_root=data_root, seed=seed, snapshot_dir=snapshot_dir),
        "n_and_label_balance": preflight_n_and_label_balance(arm_a, arm_b, arm_c),
        "covariates": preflight_covariates(arm_a, arm_b, arm_c),
        "power": preflight_power(arm_a, arm_b, seed),
        "slice_sizes": {"a": len(slices["a"]), "b": len(slices["b"])},
    }, arm_a, arm_b, arm_c


# ----------------------------------------------------------------------------
# Disease swap -- secondary arm
# ----------------------------------------------------------------------------
def _row_chapters(row: pd.Series) -> set:
    codes = sorted({c.strip().upper() for c in _recursive_parse_terms(row.get("icdcode")) if c.strip()})
    chapters = {icd10_chapter(c.split(".")[0]) for c in codes}
    return {c for c in chapters if c}


def build_donor_pool(arm_a: pd.DataFrame) -> dict:
    """chapter -> [condition name, ...], from Arm A's own single-condition,
    single-chapter trials (donor side kept unambiguous deliberately)."""
    pool = {}
    for _, row in arm_a.iterrows():
        terms = _recursive_parse_terms(row.get("condition"))
        if len(terms) != 1:
            continue
        chapters = _row_chapters(row)
        if len(chapters) != 1:
            continue
        chapter = next(iter(chapters))
        pool.setdefault(chapter, []).append(terms[0])
    return pool


def swap_disease(row: pd.Series, donor_pool: dict, rng: np.random.Generator) -> "pd.Series | None":
    """Returns a copy of `row` with `condition` replaced by a donor disease
    name from a DIFFERENT ICD chapter, or None if no eligible donor chapter
    exists for this trial (skipped, not forced).

    Found live on real Arm B data (icdcode is null-heavy on the fresh AACT
    slice per docs/p14_4_schema_slice.md): if `_row_chapters(row)` comes
    back empty because this trial's icdcode is unresolvable, the original
    `[c for c in donor_pool if c not in own_chapters]` excludes nothing --
    an unverifiable chapter was silently treated as "no conflicts" instead
    of "can't tell," so the donor could be drawn from the trial's own
    actual disease area, including (as happened for NCT00782431, a real
    Influenza trial with icdcode=None) a same-NAMED donor from a different
    Arm A trial. That is not a swap; it is a no-op with an extra join,
    exactly what assert_no_disease_leak exists to catch, but the trial
    should never have reached that assertion. Two independent guards, not
    one: an empty own_chapters can't be verified cross-chapter, so it is
    skipped outright; and even within a genuinely-different chapter, a
    donor pool built from many trials can still contain the original
    disease's own name (data-quality overlap between chapters), so
    same-named candidates are excluded before drawing."""
    own_chapters = _row_chapters(row)
    if not own_chapters:
        return None
    own_terms = {t.strip().lower() for t in _recursive_parse_terms(row.get("condition"))}
    eligible_chapters = [c for c in donor_pool if c not in own_chapters]
    if not eligible_chapters:
        return None
    chapter = rng.choice(eligible_chapters)
    candidates = [n for n in donor_pool[chapter] if n.strip().lower() not in own_terms]
    if not candidates:
        return None
    # str(...): rng.choice on a Python list of str returns a numpy str_
    # scalar, not a plain str. numpy 2.x's repr() on that scalar is
    # "np.str_('Asthma')", not "'Asthma'" -- repr([donor_name]) would
    # write condition as the literal text "[np.str_('Asthma')]", which
    # ast.literal_eval can't parse, so _disease_filler's parser falls back
    # to treating that whole garbled string as the "disease name" shown to
    # the model. Silent, not a crash -- would not have been caught by
    # assert_no_disease_leak (which only checks for the ORIGINAL disease's
    # terms leaking, not for the donor's name being malformed).
    donor_name = str(rng.choice(candidates))
    swapped = row.copy()
    swapped["condition"] = repr([donor_name])
    return swapped


# ----------------------------------------------------------------------------
# R1/R2 -- bootstrap every reported metric, and align opus/reference onto
# the same rows for the paired difference-in-differences test
# ----------------------------------------------------------------------------
def bootstrap_all_metrics(x_df: pd.DataFrame, y_df: pd.DataFrame, n_resamples: int, seed: int) -> dict:
    """`two_sample_cluster_bootstrap` under each of REPORTED_METRICS, so
    every contrast reports AUROC/PR-AUC (threshold-free, R1) alongside
    balanced_accuracy (thresholded at BALANCED_ACCURACY_THRESHOLD,
    secondary/contextual only) rather than balanced_accuracy alone."""
    return {
        metric: two_sample_cluster_bootstrap(
            x_df["nct_id"].values, x_df["p14_label"].values, x_df["proba"].values,
            y_df["nct_id"].values, y_df["p14_label"].values, y_df["proba"].values,
            metric=metric, n_resamples=n_resamples, seed=seed,
        )
        for metric in REPORTED_METRICS
    }


def align_opus_and_reference(opus_df: pd.DataFrame, ref_df: pd.DataFrame) -> tuple:
    """`score_arm` and `score_arm_reference` iterate rows in different
    orders (the latter groups by endpoint first), so opus_df and ref_df
    are NOT positionally aligned even though they cover the same trials --
    an explicit join on (nct_id, endpoint, p14_label) is required before
    treating same-index entries as the same row, which
    diff_in_diff_bootstrap depends on (it needs proba_method[i] and
    proba_ref[i] to be the same trial on every draw, not just the same
    arm)."""
    merged = opus_df.merge(ref_df, on=["nct_id", "endpoint", "p14_label"], suffixes=("_opus", "_ref"))
    return (merged["nct_id"].values, merged["p14_label"].values,
           merged["proba_opus"].values, merged["proba_ref"].values)


# ----------------------------------------------------------------------------
# R3 -- disease-swap granularity. Opus emits coarse verbalized integers;
# the reference is continuous and essentially never moves less than 0.05
# by chance -- that resolution mismatch alone could produce an
# insensitivity gap with no memorisation involved, per
# docs/t28b_reanalysis_plan.md R3.
# ----------------------------------------------------------------------------
def _distinct_value_profile(values) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "n_distinct": 0, "largest_tie_block": 0, "largest_tie_block_frac": None}
    uniq, counts = np.unique(values, return_counts=True)
    return {
        "n": int(len(values)), "n_distinct": int(len(uniq)),
        "largest_tie_block": int(counts.max()),
        "largest_tie_block_frac": float(counts.max() / len(values)),
    }


# ----------------------------------------------------------------------------
# Elicitation -- identical to what T28 actually runs (src/methods/llm.py)
# ----------------------------------------------------------------------------
def elicit_row(client, results_dir, model_id, endpoint, row, meter, arm="L7", slot_row=None):
    """`arm` defaults to "L7" -- the primary A/B/C contrasts must render
    byte-identically to what already ran (docs/t28b_l0_implementation_plan.md
    P1's regression guard: re-running them after this change must show
    meter.calls == 0, cache_hits == 1616, or a prompt changed and those
    results are no longer the ones already reported).

    `slot_row`, if given, renders via `render_arm_with_disease_override`:
    `row` supplies the shared body (and its scrubbing), `slot_row` supplies
    the disease slot's content -- required for a disease-swap probe, where
    rendering the already-swapped row directly (the original bug) leaves
    the real disease unscrubbed from the shared body and/or, for L7,
    verbatim in the untouched brief summary."""
    question = TASK_QUESTIONS[endpoint]
    nct_id = str(row["nct_id"])
    if slot_row is not None:
        rendered = render_arm_with_disease_override(row, arm, nct_id, slot_row)
    else:
        rendered = render_arm(row, arm, nct_id)
    prompt = _build_verbalized_prompt(question, rendered.text)
    result = elicit_probability(client, results_dir, model_id, prompt, temperature=0.0,
                               primary="verbalized", meter=meter)
    return result


_SCORE_ARM_COLUMNS = ("nct_id", "endpoint", "p14_label", "arm", "proba", "parse_ok", "refused", "raw_text")


def score_arm(client, results_dir, model_id, arm_df: pd.DataFrame, meter, arm_name: str,
             arm: str = "L7", slot_df: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """`arm_name` is a free-text label carried into the artifact for
    readability (e.g. "A", "B", "swap"); `arm` is the actual serialize.py
    arm code rendered (defaults "L7", matching every call site before P1).
    `slot_df`, if given, must share `arm_df`'s index (row-for-row) and
    supplies each row's disease-slot override -- see `elicit_row`.

    Always returns a DataFrame with `_SCORE_ARM_COLUMNS`, even when
    `arm_df` is empty: a swap arm can legitimately have zero rows (every
    candidate had no eligible cross-chapter donor, e.g. an icdcode-null
    Arm B trial -- see `swap_disease`'s docstring), and
    `pd.DataFrame.from_records([])` with no explicit columns returns a
    columnless frame that KeyErrors on the merge in
    `_paired_arm_comparison` instead of degrading to an empty result."""
    records = []
    for idx, row in arm_df.iterrows():
        slot_row = slot_df.loc[idx] if slot_df is not None else None
        result = elicit_row(client, results_dir, model_id, row["endpoint"], row, meter,
                            arm=arm, slot_row=slot_row)
        records.append({
            "nct_id": row["nct_id"], "endpoint": row["endpoint"], "p14_label": int(row["p14_label"]),
            "arm": arm, "proba": result.primary_prob, "parse_ok": result.parse_ok,
            "refused": result.refused, "raw_text": result.raw_text,
        })
    return pd.DataFrame.from_records(records, columns=list(_SCORE_ARM_COLUMNS))


def score_arm_reference(method_by_endpoint: dict, arm_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for endpoint, sub in arm_df.groupby("endpoint", observed=True):
        method = method_by_endpoint[endpoint]
        proba = method.predict_proba(sub)
        for (_, row), p in zip(sub.iterrows(), proba):
            records.append({"nct_id": row["nct_id"], "endpoint": endpoint,
                           "p14_label": int(row["p14_label"]), "proba": float(p)})
    return pd.DataFrame.from_records(records)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def run(data_root="data", results_dir="results", n_arm_a=500, n_arm_b=500,
       seed=42, region="us-west-2", boto_client=None, snapshot_dir=None,
       model_key=OPUS_MODEL_KEY, out_path=OUT_PATH, n_resamples=1000, n_arm_c=None) -> dict:
    """The primary A-vs-B(-vs-C) AUROC-based recall test only. The disease-
    swap secondary arm moved out entirely (see run_l0_null_check below) --
    it was rendered via a hardcoded L7, which concatenates the swapped
    condition with the trial's own UNSWAPPED brief summary, so the
    "swapped" prompt still named the real disease
    (docs/t28b_l0_implementation_plan.md). That is a data-collection bug,
    not fixable by reanalyzing the scores already collected; the
    replacement is a from-scratch, correctly-rendered probe, not a patch
    to this function. Keeping this function scoped to A/B/C also means
    re-running it now still hits the response cache for those (unaffected)
    contrasts."""
    price_table = load_price_table()
    cutoff_str = price_table["models"][model_key]["cutoff"]
    cutoff = pd.Period(cutoff_str, freq="M").end_time

    with Timer() as t:
        preflight, arm_a, arm_b, arm_c = run_preflight(data_root, cutoff, n_arm_a, n_arm_b, seed,
                                                        snapshot_dir=snapshot_dir, n_arm_c=n_arm_c)

        r1_by_endpoint = {ep: fit_frozen_reference(ep, R1_TEXT_COLS, data_root, seed) for ep in ENDPOINTS}

        api_model_id = resolve_model_id(model_key, price_table)
        client = BedrockClient(region=region, boto_client=boto_client)
        meter = Meter()

        opus_a = score_arm(client, results_dir, api_model_id, arm_a, meter, "A")
        opus_b = score_arm(client, results_dir, api_model_id, arm_b, meter, "B")
        opus_c = score_arm(client, results_dir, api_model_id, arm_c, meter, "C") if len(arm_c) else pd.DataFrame()

        ref1_a = score_arm_reference(r1_by_endpoint, arm_a)
        ref1_b = score_arm_reference(r1_by_endpoint, arm_b)
        ref1_c = score_arm_reference(r1_by_endpoint, arm_c) if len(arm_c) else pd.DataFrame()

        # R1: every reported metric (AUROC/PR-AUC threshold-free, plus
        # balanced_accuracy for context), both the primary A-vs-B contrast
        # and the secondary A-vs-C one.
        primary_opus = bootstrap_all_metrics(opus_a, opus_b, n_resamples, seed)
        primary_ref = bootstrap_all_metrics(ref1_a, ref1_b, n_resamples, seed)

        secondary_opus, secondary_ref = None, None
        if len(opus_c):
            secondary_opus = bootstrap_all_metrics(opus_a, opus_c, n_resamples, seed)
            secondary_ref = bootstrap_all_metrics(ref1_a, ref1_c, n_resamples, seed)

        # R2: the paired difference-in-differences on DECISION_METRIC,
        # aligned row-for-row between Opus and the reference within each
        # arm (see align_opus_and_reference's docstring for why this
        # can't just zip the two DataFrames positionally).
        nct_a, y_a, opus_proba_a, ref_proba_a = align_opus_and_reference(opus_a, ref1_a)
        nct_b, y_b, opus_proba_b, ref_proba_b = align_opus_and_reference(opus_b, ref1_b)
        diff_in_diff = diff_in_diff_bootstrap(
            nct_a, y_a, opus_proba_a, ref_proba_a, nct_b, y_b, opus_proba_b, ref_proba_b,
            metric=DECISION_METRIC, n_resamples=n_resamples, seed=seed,
        )

        # R4: per-endpoint split of Arm A, reconciled against T28a's
        # per-task point estimates (n=34 each) -- distinguishes
        # small-sample optimism from an elicitation-format effect.
        per_endpoint_arm_a = {}
        for endpoint in ENDPOINTS:
            sub = opus_a[opus_a["endpoint"] == endpoint]
            if len(sub) == 0:
                continue
            ci = one_sample_cluster_bootstrap(sub["nct_id"].values, sub["p14_label"].values,
                                              sub["proba"].values, metric=DECISION_METRIC,
                                              n_resamples=n_resamples, seed=seed)
            ci_bal_acc = one_sample_cluster_bootstrap(sub["nct_id"].values, sub["p14_label"].values,
                                                      sub["proba"].values, metric="balanced_accuracy",
                                                      n_resamples=n_resamples, seed=seed)
            t28a_point = T28A_OPUS_POINT_ESTIMATES.get(endpoint)
            per_endpoint_arm_a[endpoint] = {
                DECISION_METRIC: ci, "balanced_accuracy": ci_bal_acc,
                "t28a_point_estimate_balanced_accuracy": t28a_point,
                "t28a_point_estimate_inside_ci": (
                    t28a_point is not None and ci_bal_acc["lo"] <= t28a_point <= ci_bal_acc["hi"]
                ),
            }

        branch, reason = decide(primary_opus, primary_ref, diff_in_diff)

    artifact = {
        "test_id": "T28b",
        "claim_at_stake": "whether Opus 4.5's T28a predictive signal on TrialBench (mortality, "
                          "SAE) survives removal of outcome knowledge, holding trial identity "
                          "knowledge constant -- recall vs genuine reading",
        "inputs": {
            "model_key": model_key, "api_model_id": api_model_id, "cutoff": cutoff_str,
            "n_arm_a": n_arm_a, "n_arm_b": n_arm_b, "seed": seed,
            "data_root": data_root, "r1_text_cols": list(R1_TEXT_COLS),
            "n_resamples": n_resamples, "decision_metric": DECISION_METRIC,
            "balanced_accuracy_threshold": BALANCED_ACCURACY_THRESHOLD,
            "reported_metrics": list(REPORTED_METRICS),
            "phase_string_format_note": (
                "Arm A carries TrialBench's title-case phase strings (Phase2); Arms B/C carry "
                "AACT's uppercase ones (PHASE2) plus categories absent from A entirely "
                "(PHASE1/PHASE2, EARLY_PHASE1 -- 15.4% of Arm B per the covariates report). Not "
                "normalized: doing so changes prompt text, changes the prompt hash, misses the "
                "response cache, and re-bills roughly the full run's cost "
                "(docs/t28b_reanalysis_plan.md). Deferred pending whether R1/R2 leaves a verdict "
                "worth re-running for."
            ),
            "arm_c_prevalence_note": (
                "Arm C's positive rate (mortality 9.6%, SAE 12.5%, both at n=208) is far below "
                "Arm A's (39.6%, 67.6%), leaving roughly 20 positive trials per endpoint -- the "
                "secondary A-vs-C contrast is uninformative by construction at this n, not a null "
                "result, and should be read that way (docs/t28b_reanalysis_plan.md)."
            ),
        },
        "preflight": preflight,
        "n_sampled": {"arm_a": len(arm_a), "arm_b": len(arm_b), "arm_c": len(arm_c)},
        "primary_a_vs_b": {"opus": primary_opus, "reference": primary_ref},
        "secondary_a_vs_c": {"opus": secondary_opus, "reference": secondary_ref},
        "diff_in_diff": diff_in_diff,
        "per_endpoint_arm_a": per_endpoint_arm_a,
        "disease_swap": None,
        "disease_swap_note": (
            "Withdrawn -- the original disease-swap arm was rendered via a hardcoded L7, which "
            "concatenates the swapped condition with the trial's own UNSWAPPED brief summary, so "
            "the 'swapped' prompt still named the real disease. quarantine_ordering: true was not "
            "evidence of memorisation; it was this rendering bug. See "
            "docs/t28b_l0_implementation_plan.md and results/experiments/t28b_l0_null.json for the "
            "corrected, from-scratch disease-sensitivity probe."
        ),
        "branch": branch, "branch_reason": reason,
        "per_trial": {
            "opus_arm_a": opus_a.to_dict(orient="records"), "opus_arm_b": opus_b.to_dict(orient="records"),
            "opus_arm_c": opus_c.to_dict(orient="records") if len(opus_c) else [],
            "reference_arm_a": ref1_a.to_dict(orient="records"), "reference_arm_b": ref1_b.to_dict(orient="records"),
            "reference_arm_c": ref1_c.to_dict(orient="records") if len(ref1_c) else [],
        },
        "meter": meter.summary(price_table, api_model_id, "sync"),
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(out_path, artifact)
    print(f"\nwrote {out_path}")
    print(f"branch={branch} ({reason})")
    return artifact


def decide(primary_opus: dict, primary_ref: dict, diff_in_diff: dict) -> tuple:
    """docs/t28b_reanalysis_plan.md's R1->R2 procedure, replacing the
    original decide()'s two-independent-significance-tests logic (a
    Gelman-Stern fallacy: testing Opus's drop and the reference's drop for
    significance SEPARATELY never tests whether the two drops DIFFER from
    each other, which is the actual claim -- the campaign's fourth
    verdict-layer bug after T28a's threshold, T26's floor arm, T27's
    direction check).

    `primary_opus`/`primary_ref` are `bootstrap_all_metrics` dicts (one
    per DECISION_METRIC, here `auroc`, plus context metrics);
    `diff_in_diff` is `diff_in_diff_bootstrap`'s result on the same metric.

    R1 -- is Opus's own AUROC drop from A to B real, not a byproduct of
    thresholding a poorly-calibrated verbalized probability at a fixed
    0.5 while the base rate shifts between arms? If Opus's AUROC (a
    threshold-free discrimination measure) is flat across A and B, any
    earlier balanced-accuracy drop was a calibration artifact, not
    evidence of anything -- reported and the verdict stops here.

    R2 -- if AUROC does drop, does it drop MORE than the reference's own
    AUROC drop on the identical rows? That is the actual "recall
    demonstrated" claim; diff_in_diff's CI excluding zero is the
    criterion, not two separately-significant deltas.
    """
    opus_auroc = primary_opus[DECISION_METRIC]
    opus_drops = opus_auroc["lo"] > 0  # CI excludes 0, delta > 0 -- A significantly beats B on AUROC

    if not opus_drops:
        return ("NO_RECALL_FINDING_CALIBRATION_ARTIFACT",
                f"R1: Opus's own {DECISION_METRIC} does not significantly drop from A to B "
                f"(delta={opus_auroc['mean_delta']:.3f}, CI=[{opus_auroc['lo']:.3f}, "
                f"{opus_auroc['hi']:.3f}]) -- any earlier balanced-accuracy drop (thresholded at "
                f"{BALANCED_ACCURACY_THRESHOLD}) is read as a calibration/base-rate artifact, not "
                f"evidence of recall. Report the threshold sensitivity as the finding.")

    d = diff_in_diff
    if d["lo"] > 0:
        return ("OUTCOME_RECALL_DEMONSTRATED",
                f"R1: Opus's {DECISION_METRIC} drops significantly (delta="
                f"{opus_auroc['mean_delta']:.3f}, CI=[{opus_auroc['lo']:.3f}, {opus_auroc['hi']:.3f}]). "
                f"R2: that drop significantly exceeds the reference's own drop on the identical "
                f"rows (diff={d['mean_diff']:.3f}, CI=[{d['lo']:.3f}, {d['hi']:.3f}], observed "
                f"correlation rho={d['rho']}) -- Opus's TrialBench figures are recall-inflated. No "
                f"absolute Opus number from TrialBench is quoted; T28's Opus arm is reported on "
                f"fresh data or not at all.")

    return ("INCONCLUSIVE",
            f"R1: Opus's {DECISION_METRIC} drops significantly (delta={opus_auroc['mean_delta']:.3f}, "
            f"CI=[{opus_auroc['lo']:.3f}, {opus_auroc['hi']:.3f}]). R2: that drop does not "
            f"significantly exceed the reference's own drop on the identical rows "
            f"(diff={d['mean_diff']:.3f}, CI=[{d['lo']:.3f}, {d['hi']:.3f}], observed correlation "
            f"rho={d['rho']}) -- underpowered to separate Opus's drop from the reference's. Report "
            f"both drops and the difference with its CI; do not read this as recall demonstrated "
            f"or as cleared.")


# ----------------------------------------------------------------------------
# T28b-L0 -- the corrected disease-sensitivity null (docs/t28b_l0_
# implementation_plan.md), replacing the withdrawn swap arm above.
# ----------------------------------------------------------------------------
class _RenderedTextTfidfLogReg(_TfidfLogRegBase):
    """P3: fit on the ACTUAL string `render_arm` produces for the L1 arm,
    not `concat_text`'s column-based text -- input parity with Opus is
    then exact, byte for byte, since the reference reads the same string
    Opus reads for the baseline arm. `nct_id` doesn't affect the rendered
    text at all (render_arm only uses it to label the returned
    ``Rendered``), so a placeholder is fine here; only used to fit
    ``vec_``/``clf_`` -- scoring an arbitrary arm or a disease-swap
    override goes through `score_rendered_texts` below with this same
    frozen pair, since which arm to render varies per call in a way a
    single fixed `self.arm` wouldn't support."""

    def _texts(self, X):
        return [render_arm(row, "L1", "").text for _, row in X.iterrows()]


def fit_rendered_text_reference(endpoint: str, data_root: str, seed: int) -> _RenderedTextTfidfLogReg:
    X_parts, y_parts = [], []
    for phase in PHASES:
        try:
            td = load_task_phase(data_root, endpoint, phase, seed=seed)
        except FileNotFoundError:
            continue
        X_parts.append(td.X_train)
        y_parts.append(td.y_train)
    if not X_parts:
        raise FileNotFoundError(f"no TrialBench train data found for {endpoint!r} under {data_root!r}")
    X_train = pd.concat(X_parts, ignore_index=True)
    y_train = np.concatenate(y_parts)
    method = _RenderedTextTfidfLogReg(seed=seed)
    method.fit(X_train, y_train)
    return method


def score_rendered_texts(method: _RenderedTextTfidfLogReg, texts: list) -> np.ndarray:
    Xt = method.vec_.transform(texts)
    proba = method.clf_.predict_proba(Xt)
    return method._binary_scores(proba)


def build_swap_slot(candidates: pd.DataFrame, donor_pool: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Per-row donor-swapped `condition` only (see `swap_disease`) -- one
    row per successfully-swapped candidate (a row with no eligible donor
    chapter is dropped, not forced), index-aligned with `candidates` so
    `.loc` joins work (`swap_disease` returns `row.copy()`, which keeps
    the row's original index label)."""
    swapped = [swap_disease(row, donor_pool, rng) for _, row in candidates.iterrows()]
    kept = [s for s in swapped if s is not None]
    return pd.DataFrame(kept) if kept else candidates.iloc[0:0].copy()


def assert_no_disease_leak(original_row: pd.Series, slot_row: pd.Series, rendered_text: str) -> None:
    """None of the ORIGINAL disease's terms may survive in a swapped
    rendering -- the exact failure docs/t28b_l0_implementation_plan.md
    found in the withdrawn L7-rendered swap (the untouched brief summary
    named the real disease verbatim regardless of what the slot said).

    Only scans the free-text fields `_render_body` actually scrubs
    (`_SCRUB_COLS`) plus the disease slot line itself -- every other body
    line is a fixed-vocabulary field (phase, allocation, masking, DMC/FDA
    yes-no, etc.) whose rendered text is independent of the trial's
    condition, EXCEPT for one real collision found on real data: a trial
    whose condition is literally "Healthy" (common for Phase 1
    healthy-volunteer studies) always matches the fixed body line "Healthy
    volunteers accepted: ..." regardless of what the disease was swapped
    to -- that line says nothing about the trial's actual disease, so
    counting it as a leak would crash on every such trial, not just the
    ones the swap genuinely failed to scrub."""
    orig_terms = [t for t in _recursive_parse_terms(original_row.get("condition")) if t]
    if not orig_terms:
        return
    safe_lines = {
        f"{label}: {_fmt(original_row.get(col))}" for col, label in BODY_COLS if col not in _SCRUB_COLS
    }
    search_text = "\n".join(line for line in rendered_text.split("\n") if line not in safe_lines)
    lowered = search_text.lower()
    leaked = [t for t in orig_terms if t.lower() in lowered]
    if leaked:
        raise AssertionError(
            f"original disease term(s) {leaked} leaked into a swapped rendering for "
            f"{original_row.get('nct_id')} -- the swap is not valid, do not score it"
        )


_SCORE_ARM_REFERENCE_TEXT_COLUMNS = ("nct_id", "endpoint", "p14_label", "arm", "proba")


def score_reference_arm_text(ref_by_endpoint: dict, candidates: pd.DataFrame, arm: str,
                             slot_df: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Always returns a DataFrame with `_SCORE_ARM_REFERENCE_TEXT_COLUMNS`,
    even when `candidates` is empty -- see `score_arm`'s docstring for why
    (a fully-empty swap arm is a real possibility, not a hypothetical)."""
    records = []
    for endpoint, sub in candidates.groupby("endpoint", observed=True):
        method = ref_by_endpoint[endpoint]
        texts = []
        for idx, row in sub.iterrows():
            if slot_df is not None and idx in slot_df.index:
                slot_row = row.copy()
                slot_row["condition"] = slot_df.loc[idx, "condition"]
                texts.append(render_arm_with_disease_override(row, arm, str(row["nct_id"]), slot_row).text)
            else:
                texts.append(render_arm(row, arm, str(row["nct_id"])).text)
        proba = score_rendered_texts(method, texts)
        for (_, row), p in zip(sub.iterrows(), proba):
            records.append({"nct_id": row["nct_id"], "endpoint": endpoint,
                           "p14_label": int(row["p14_label"]), "arm": arm, "proba": float(p)})
    return pd.DataFrame.from_records(records, columns=list(_SCORE_ARM_REFERENCE_TEXT_COLUMNS))


def run_three_point_curve(client, results_dir, model_id, candidates: pd.DataFrame, donor_pool: dict,
                         ref_by_endpoint: dict, rng: np.random.Generator, meter, label: str) -> dict:
    """L1 (baseline), L0 (masked), L1-swap (donor disease), for both Opus
    and the P3 reference, on the SAME `candidates` rows (a subset of Arm A
    or Arm B). Every swap rendering is checked for a disease leak
    BEFORE any model call -- P2's validity gate, zero-cost."""
    slot_df = build_swap_slot(candidates, donor_pool, rng)
    swappable = candidates.loc[slot_df.index]

    for idx in slot_df.index:
        row = candidates.loc[idx]
        slot_row = row.copy()
        slot_row["condition"] = slot_df.loc[idx, "condition"]
        rendered = render_arm_with_disease_override(row, "L1", str(row["nct_id"]), slot_row)
        assert_no_disease_leak(row, slot_row, rendered.text)

    opus_l1 = score_arm(client, results_dir, model_id, candidates, meter, f"{label}_l1", arm="L1")
    opus_l0 = score_arm(client, results_dir, model_id, candidates, meter, f"{label}_l0", arm="L0")
    opus_swap = score_arm(client, results_dir, model_id, swappable, meter, f"{label}_swap",
                          arm="L1", slot_df=slot_df)

    ref_l1 = score_reference_arm_text(ref_by_endpoint, candidates, arm="L1")
    ref_l0 = score_reference_arm_text(ref_by_endpoint, candidates, arm="L0")
    ref_swap = score_reference_arm_text(ref_by_endpoint, swappable, arm="L1", slot_df=slot_df)

    return {
        "opus_l1": opus_l1, "opus_l0": opus_l0, "opus_swap": opus_swap,
        "ref_l1": ref_l1, "ref_l0": ref_l0, "ref_swap": ref_swap,
        "n_candidates": len(candidates), "n_swap": len(swappable),
    }


def _paired_arm_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
                          n_resamples: int, seed: int) -> dict:
    """One arm-pair's full statistics block: paired AUROC delta (the two
    arms are the SAME trials scored differently, so `pooled_paired_bootstrap`
    -- same-row, different-method -- applies directly, not the independent
    two-sample bootstrap T28b's primary contrast needs), movement
    histogram, exact-zero/near-zero split, and rank-based movement
    (Opus's verbalized integers are coarse; a raw 0.05 threshold alone is
    scale-blind to that)."""
    merged = df_a.merge(df_b, on=["nct_id", "endpoint", "p14_label"], suffixes=(f"_{label_a}", f"_{label_b}"))
    nct = merged["nct_id"].values
    y = merged["p14_label"].values
    pa = merged[f"proba_{label_a}"].values
    pb = merged[f"proba_{label_b}"].values

    auroc_a = one_sample_cluster_bootstrap(nct, y, pa, metric="auroc", n_resamples=n_resamples, seed=seed)
    auroc_b = one_sample_cluster_bootstrap(nct, y, pb, metric="auroc", n_resamples=n_resamples, seed=seed)
    paired_delta = pooled_paired_bootstrap(nct, y, pa, pb, metric="auroc", n_resamples=n_resamples, seed=seed)

    move = np.abs(pa - pb)
    hist_counts, hist_edges = np.histogram(move, bins=10, range=(0.0, 1.0))
    spearman = float(pd.Series(pa).corr(pd.Series(pb), method="spearman")) if len(pa) > 2 else None
    rank_shift = float(np.mean(np.abs(pd.Series(pa).rank().values - pd.Series(pb).rank().values))) if len(pa) else None

    return {
        f"{label_a}_auroc": auroc_a, f"{label_b}_auroc": auroc_b, "paired_delta": paired_delta,
        "move_exactly_zero_rate": float((move == 0).mean()) if len(move) else None,
        "move_nonzero_below_threshold_rate": (
            float(((move > 0) & (move < DISEASE_SWAP_MOVE_THRESHOLD)).mean()) if len(move) else None
        ),
        "insensitivity_rate": float((move < DISEASE_SWAP_MOVE_THRESHOLD).mean()) if len(move) else None,
        "movement_histogram": {"counts": hist_counts.tolist(), "bin_edges": hist_edges.tolist()},
        "spearman_correlation": spearman, "mean_abs_rank_shift": rank_shift,
        "n": int(len(merged)),
    }


def curve_statistics(curve: dict, n_resamples: int, seed: int) -> dict:
    return {
        "opus_l1_vs_l0": _paired_arm_comparison(curve["opus_l1"], curve["opus_l0"], "l1", "l0", n_resamples, seed),
        "opus_l1_vs_swap": _paired_arm_comparison(curve["opus_l1"], curve["opus_swap"], "l1", "swap", n_resamples, seed),
        "reference_l1_vs_l0": _paired_arm_comparison(curve["ref_l1"], curve["ref_l0"], "l1", "l0", n_resamples, seed),
        "reference_l1_vs_swap": _paired_arm_comparison(curve["ref_l1"], curve["ref_swap"], "l1", "swap", n_resamples, seed),
        "opus_value_profile_l1": _distinct_value_profile(curve["opus_l1"]["proba"].values),
        "n_candidates": curve["n_candidates"], "n_swap": curve["n_swap"],
    }


def decide_l0_reading(stats_a: dict, stats_b: dict) -> tuple:
    """docs/t28b_l0_implementation_plan.md's reading matrix, on the L1->L0
    (masked) contrast -- also T28's own standing-rule-8 null on one cell,
    per the plan's own framing. 'sensitive' = the paired AUROC delta's CI
    excludes 0 (masking the disease measurably hurt discrimination);
    'insensitive' = it doesn't. Arm B (P4) overrides the Arm-A-only matrix
    if it shows sensitivity where TrialBench (Arm A) showed invariance --
    that specific pattern is the memorisation signature regardless of
    which matrix row fired on Arm A alone."""
    ref_sensitive_a = stats_a["reference_l1_vs_l0"]["paired_delta"]["lo"] > 0
    opus_sensitive_a = stats_a["opus_l1_vs_l0"]["paired_delta"]["lo"] > 0
    opus_sensitive_b = stats_b["opus_l1_vs_l0"]["paired_delta"]["lo"] > 0

    if opus_sensitive_b and not opus_sensitive_a:
        return ("MEMORISATION_SIGNATURE",
                "P4 override: Opus is sensitive to disease removal on Arm B (outcome not "
                "memorisable) but invariant on Arm A (TrialBench, outcome memorisable) -- "
                "consistent with substituted recall on Arm A, not genuine disease-blindness.")
    if not ref_sensitive_a and not opus_sensitive_a:
        return ("QUARANTINE_LIFTED_DISEASE_UNINFORMATIVE",
                "The reference is also insensitive to L1->L0 on Arm A -- disease carries little "
                "signal for these endpoints on these rows (consistent with T22's INCONCLUSIVE "
                "disease-share finding). Opus's invariance reads as innocent.")
    if ref_sensitive_a and not opus_sensitive_a:
        return ("NOT_SEPARABLE_ON_ARM_A_ALONE",
                "The reference is sensitive to disease removal but Opus is not -- Opus is not "
                "using predictive disease information. Weak reading and substituted recall are "
                "not separable from Arm A alone; see the Arm B (P4) result alongside this.")
    return ("NORMAL_BEHAVIOUR",
            "Both Opus and the reference are sensitive to disease removal on Arm A -- the swap "
            "probe behaves as designed; T28's own L0 null (standing rule 8) also passes on this "
            "cell, previewed at negligible extra cost.")


def run_l0_null_check(data_root="data", results_dir="results", n_arm_a=500, n_arm_b=500,
                      n_l0_a=200, n_l0_b=200, seed=42, region="us-west-2", boto_client=None,
                      snapshot_dir=None, model_key=OPUS_MODEL_KEY, out_path=L0_OUT_PATH,
                      n_resamples=1000, n_arm_c=None) -> dict:
    price_table = load_price_table()
    cutoff_str = price_table["models"][model_key]["cutoff"]
    cutoff = pd.Period(cutoff_str, freq="M").end_time

    with Timer() as t:
        # Same sampling as run() (same seed -> same rows), so this reuses
        # Arm A/B rather than drawing a fresh, incomparable sample.
        preflight, arm_a, arm_b, arm_c = run_preflight(data_root, cutoff, n_arm_a, n_arm_b, seed,
                                                        snapshot_dir=snapshot_dir, n_arm_c=n_arm_c)

        donor_pool = build_donor_pool(arm_a)
        rng = np.random.default_rng(seed)

        candidates_a = arm_a.sample(n=min(n_l0_a, len(arm_a)), random_state=seed)
        candidates_b = arm_b.sample(n=min(n_l0_b, len(arm_b)), random_state=seed)

        ref_by_endpoint = {ep: fit_rendered_text_reference(ep, data_root, seed) for ep in ENDPOINTS}

        api_model_id = resolve_model_id(model_key, price_table)
        client = BedrockClient(region=region, boto_client=boto_client)
        meter = Meter()

        curve_a = run_three_point_curve(client, results_dir, api_model_id, candidates_a, donor_pool,
                                        ref_by_endpoint, rng, meter, label="A")
        curve_b = run_three_point_curve(client, results_dir, api_model_id, candidates_b, donor_pool,
                                        ref_by_endpoint, rng, meter, label="B")

        stats_a = curve_statistics(curve_a, n_resamples, seed)
        stats_b = curve_statistics(curve_b, n_resamples, seed)

        reading, reading_reason = decide_l0_reading(stats_a, stats_b)

    def _per_trial(curve):
        return {k: v.to_dict(orient="records") for k, v in curve.items() if isinstance(v, pd.DataFrame)}

    artifact = {
        "test_id": "T28b-L0",
        "claim_at_stake": "whether Opus 4.5's near-total insensitivity to a disease-identity "
                          "swap (the withdrawn quarantine_ordering in t28b_opus_recall.json) "
                          "reflects genuine disease-blindness or was a rendering bug, and "
                          "whether Opus uses disease-slot information at all for these endpoints "
                          "-- also previews T28's own standing-rule-8 L0 null on one cell",
        "inputs": {
            "model_key": model_key, "api_model_id": api_model_id, "cutoff": cutoff_str,
            "n_arm_a": n_arm_a, "n_arm_b": n_arm_b, "n_l0_a": n_l0_a, "n_l0_b": n_l0_b,
            "seed": seed, "data_root": data_root, "n_resamples": n_resamples,
            "move_threshold": DISEASE_SWAP_MOVE_THRESHOLD,
            "supersedes": "results/experiments/t28b_opus_recall.json's disease_swap field "
                         "(withdrawn -- rendered via hardcoded L7, see "
                         "docs/t28b_l0_implementation_plan.md)",
        },
        "preflight": preflight,
        "curve_arm_a": stats_a, "curve_arm_b": stats_b,
        "reading": reading, "reading_reason": reading_reason,
        "per_trial": {"arm_a": _per_trial(curve_a), "arm_b": _per_trial(curve_b)},
        "meter": meter.summary(price_table, api_model_id, "sync"),
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(out_path, artifact)
    print(f"\nwrote {out_path}")
    print(f"reading={reading} ({reading_reason})")
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-arm-a", type=int, default=500)
    ap.add_argument("--n-arm-b", type=int, default=500)
    ap.add_argument("--n-arm-c", type=int, default=None,
                    help="cap Arm C below its spec default (all of slice (b), ~416 rows -- real "
                         "spend). Leave unset for the real run; pass a small int for a cheap "
                         "smoke test.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--model", default=OPUS_MODEL_KEY)
    ap.add_argument("--out-path", default=OUT_PATH)
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--preflight-only", action="store_true",
                    help="run only the zero-cost pre-flight checks and print them, no model calls")
    ap.add_argument("--l0-null-check", action="store_true",
                    help="run the corrected disease-sensitivity probe (docs/t28b_l0_"
                         "implementation_plan.md) instead of the primary A/B/C contrast -- new "
                         "prompts, real spend, writes results/experiments/t28b_l0_null.json by "
                         "default (--l0-out-path to change it)")
    ap.add_argument("--n-l0-a", type=int, default=200, help="rows drawn from Arm A for the L0/L1/swap curve")
    ap.add_argument("--n-l0-b", type=int, default=200, help="rows drawn from Arm B for the L0/L1/swap curve")
    ap.add_argument("--l0-out-path", default=L0_OUT_PATH)
    args = ap.parse_args()

    if args.preflight_only:
        price_table = load_price_table()
        cutoff = pd.Period(price_table["models"][args.model]["cutoff"], freq="M").end_time
        preflight, arm_a, arm_b, arm_c = run_preflight(args.data_root, cutoff, args.n_arm_a,
                                                        args.n_arm_b, args.seed, n_arm_c=args.n_arm_c)
        import json
        print(json.dumps(preflight, indent=2, default=str))
        return

    if args.l0_null_check:
        run_l0_null_check(data_root=args.data_root, results_dir=args.results_dir,
                          n_arm_a=args.n_arm_a, n_arm_b=args.n_arm_b, n_l0_a=args.n_l0_a,
                          n_l0_b=args.n_l0_b, seed=args.seed, region=args.region,
                          model_key=args.model, out_path=args.l0_out_path,
                          n_resamples=args.n_resamples, n_arm_c=args.n_arm_c)
        return

    run(data_root=args.data_root, results_dir=args.results_dir, n_arm_a=args.n_arm_a,
        n_arm_b=args.n_arm_b, seed=args.seed, region=args.region,
        model_key=args.model, out_path=args.out_path, n_resamples=args.n_resamples,
        n_arm_c=args.n_arm_c)


if __name__ == "__main__":
    main()
