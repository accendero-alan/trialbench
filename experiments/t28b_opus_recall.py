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

Two reference configurations, not one -- R1_TEXT_COLS/R2_TEXT_COLS below --
because src.methods.text_nlp.TfidfLogReg's default TEXT_COLS includes
brief_title/brief_summary/condition, three of the five DISEASE_LEAK_COLS
(src/data/serialize.py). Taking the default would let the "cannot be
recalling" reference silently read exactly the disease-identity signal the
secondary disease-swap arm exists to control for.

Usage:
    python -m experiments.t28b_opus_recall --n-arm-a 500 --n-arm-b 500 \\
        --model anthropic.claude-opus-4-5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.bedrock.client import BedrockClient, elicit_probability
from src.bedrock.meter import Meter
from src.bedrock.prices import is_verified, load_price_table, resolve_model_id
from src.data.aact import load_table, mortality_yn, sae_yn
from src.data.aact_slice import emit_trialbench_schema, slice_ab_nct_ids
from src.data.features import _recursive_parse_terms, concat_text
from src.data.icd10_hierarchy import icd10_chapter
from src.data.loader import load_task_phase
from src.data.serialize import DISEASE_LEAK_COLS, render_arm
from src.eval.pooled_bootstrap import two_sample_cluster_bootstrap
from src.methods.llm import TASK_QUESTIONS, _build_verbalized_prompt
from src.methods.text_nlp import _TfidfLogRegBase

OUT_PATH = "results/experiments/t28b_opus_recall.json"
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

# R2 (disease-swap sensitivity calibration, secondary arm only): the swap
# only rewrites the `condition` slot, not brief_summary/brief_title prose,
# which routinely still names the pre-swap disease. A reference reading
# those fields would see the ORIGINAL disease after the swap and its
# measured sensitivity would be attenuated for a reason that has nothing to
# do with recall -- see docs/t28b_opus_recall_spec.md's R2 section.
R2_TEXT_COLS = ("condition",)

DISEASE_SWAP_MOVE_THRESHOLD = 0.05
DISEASE_SWAP_INSENSITIVITY_THRESHOLD = 0.20

OPUS_MODEL_KEY = "anthropic.claude-opus-4-5"


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
    bs = load_table("brief_summaries", **kw).set_index("nct_id")["description"]
    tb_indexed = tb.set_index("nct_id")["brief_summary/textblock"]

    def _normalize(text) -> str:
        # Two confirmed-benign, systematic formatting differences between
        # the two pipelines, neither a content difference: (1) TrialBench
        # keeps raw \r\n-wrapped whitespace from the original XML, AACT's
        # copy is whitespace-normalized -- collapse all whitespace runs to
        # single spaces; (2) AACT's pipe-delimited export encodes an
        # embedded paragraph break as a literal "~" (to avoid a real
        # newline breaking the row-oriented file format) where TrialBench
        # collapses it to a space -- confirmed live on a real mismatch
        # (NCT02563561: identical 870-char content, "~" vs " " at the same
        # positions). Treat "~" as whitespace too before comparing content.
        # (3) AACT backslash-escapes literal square brackets ("\[PSP\]"
        # where TrialBench has plain "[PSP]", markdown-style) -- confirmed
        # live on NCT04399551. (4) AACT uses "*" for a bullet marker where
        # TrialBench uses "-" for the identical list -- confirmed live on
        # NCT01727414 (both renderings 1933 chars, word-for-word
        # identical apart from the marker). All four are confirmed-benign
        # formatting/markup differences between the two export pipelines,
        # never a content difference, checked by hand on every mismatch
        # sampled during development -- not assumed.
        t = (str(text).replace("~", " ").replace("\\[", "[").replace("\\]", "]")
            .replace(" * ", " - "))
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
    exists for this trial (skipped, not forced)."""
    own_chapters = _row_chapters(row)
    eligible_chapters = [c for c in donor_pool if c not in own_chapters]
    if not eligible_chapters:
        return None
    chapter = rng.choice(eligible_chapters)
    donor_name = rng.choice(donor_pool[chapter])
    swapped = row.copy()
    swapped["condition"] = repr([donor_name])
    return swapped


# ----------------------------------------------------------------------------
# Elicitation -- identical to what T28 actually runs (src/methods/llm.py)
# ----------------------------------------------------------------------------
def elicit_row(client, results_dir, model_id, endpoint, row, meter):
    question = TASK_QUESTIONS[endpoint]
    rendered = render_arm(row, "L7", str(row["nct_id"]))
    prompt = _build_verbalized_prompt(question, rendered.text)
    result = elicit_probability(client, results_dir, model_id, prompt, temperature=0.0,
                               primary="verbalized", meter=meter)
    return result


def score_arm(client, results_dir, model_id, arm_df: pd.DataFrame, meter, arm_name: str) -> pd.DataFrame:
    records = []
    for _, row in arm_df.iterrows():
        result = elicit_row(client, results_dir, model_id, row["endpoint"], row, meter)
        records.append({
            "nct_id": row["nct_id"], "endpoint": row["endpoint"], "p14_label": int(row["p14_label"]),
            "proba": result.primary_prob, "parse_ok": result.parse_ok, "refused": result.refused,
            "raw_text": result.raw_text,
        })
    return pd.DataFrame.from_records(records)


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
def run(data_root="data", results_dir="results", n_arm_a=500, n_arm_b=500, n_swap=200,
       seed=42, region="us-west-2", boto_client=None, snapshot_dir=None,
       model_key=OPUS_MODEL_KEY, out_path=OUT_PATH, n_resamples=1000, n_arm_c=None) -> dict:
    price_table = load_price_table()
    cutoff_str = price_table["models"][model_key]["cutoff"]
    cutoff = pd.Period(cutoff_str, freq="M").end_time

    with Timer() as t:
        preflight, arm_a, arm_b, arm_c = run_preflight(data_root, cutoff, n_arm_a, n_arm_b, seed,
                                                        snapshot_dir=snapshot_dir, n_arm_c=n_arm_c)

        donor_pool = build_donor_pool(arm_a)
        rng = np.random.default_rng(seed)
        swap_candidates = arm_a.sample(n=min(n_swap, len(arm_a)), random_state=seed)
        swapped_rows = []
        for _, row in swap_candidates.iterrows():
            swapped = swap_disease(row, donor_pool, rng)
            if swapped is not None:
                swapped_rows.append(swapped)
        arm_swap = pd.DataFrame(swapped_rows).reset_index(drop=True) if swapped_rows else pd.DataFrame()

        r1_by_endpoint = {ep: fit_frozen_reference(ep, R1_TEXT_COLS, data_root, seed) for ep in ENDPOINTS}
        r2_by_endpoint = {ep: fit_frozen_reference(ep, R2_TEXT_COLS, data_root, seed) for ep in ENDPOINTS}

        api_model_id = resolve_model_id(model_key, price_table)
        client = BedrockClient(region=region, boto_client=boto_client)
        meter = Meter()

        opus_a = score_arm(client, results_dir, api_model_id, arm_a, meter, "A")
        opus_b = score_arm(client, results_dir, api_model_id, arm_b, meter, "B")
        opus_c = score_arm(client, results_dir, api_model_id, arm_c, meter, "C") if len(arm_c) else pd.DataFrame()
        opus_swap = score_arm(client, results_dir, api_model_id, arm_swap, meter, "swap") if len(arm_swap) else pd.DataFrame()

        ref1_a = score_arm_reference(r1_by_endpoint, arm_a)
        ref1_b = score_arm_reference(r1_by_endpoint, arm_b)
        ref1_c = score_arm_reference(r1_by_endpoint, arm_c) if len(arm_c) else pd.DataFrame()
        ref2_swap_original = score_arm_reference(r2_by_endpoint, swap_candidates) if len(swap_candidates) else pd.DataFrame()
        ref2_swap = score_arm_reference(r2_by_endpoint, arm_swap) if len(arm_swap) else pd.DataFrame()

        primary = two_sample_cluster_bootstrap(
            opus_a["nct_id"].values, opus_a["p14_label"].values, opus_a["proba"].values,
            opus_b["nct_id"].values, opus_b["p14_label"].values, opus_b["proba"].values,
            metric="balanced_accuracy", n_resamples=n_resamples, seed=seed,
        )
        reference_ab = two_sample_cluster_bootstrap(
            ref1_a["nct_id"].values, ref1_a["p14_label"].values, ref1_a["proba"].values,
            ref1_b["nct_id"].values, ref1_b["p14_label"].values, ref1_b["proba"].values,
            metric="balanced_accuracy", n_resamples=n_resamples, seed=seed,
        )

        secondary_ac, reference_ac = None, None
        if len(opus_c):
            secondary_ac = two_sample_cluster_bootstrap(
                opus_a["nct_id"].values, opus_a["p14_label"].values, opus_a["proba"].values,
                opus_c["nct_id"].values, opus_c["p14_label"].values, opus_c["proba"].values,
                metric="balanced_accuracy", n_resamples=n_resamples, seed=seed,
            )
            reference_ac = two_sample_cluster_bootstrap(
                ref1_a["nct_id"].values, ref1_a["p14_label"].values, ref1_a["proba"].values,
                ref1_c["nct_id"].values, ref1_c["p14_label"].values, ref1_c["proba"].values,
                metric="balanced_accuracy", n_resamples=n_resamples, seed=seed,
            )

        swap_result = None
        if len(opus_swap):
            # swap_candidates is a subset of arm_a's own rows, so the
            # pre-swap Opus score for every swapped (nct_id, endpoint) pair
            # is already sitting in opus_a -- no separate call needed, just
            # join on (nct_id, endpoint), not nct_id alone (the same trial
            # can appear under both endpoints with different questions/
            # scores).
            orig_opus = opus_a.set_index(["nct_id", "endpoint"])["proba"]
            swap_opus = opus_swap.set_index(["nct_id", "endpoint"])["proba"]
            common = orig_opus.index.intersection(swap_opus.index)
            opus_move = (orig_opus.loc[common] - swap_opus.loc[common]).abs()
            opus_insensitivity = float((opus_move < DISEASE_SWAP_MOVE_THRESHOLD).mean()) if len(opus_move) else None

            ref_insensitivity = None
            if len(ref2_swap_original) and len(ref2_swap):
                orig_ref = ref2_swap_original.set_index(["nct_id", "endpoint"])["proba"]
                swap_ref = ref2_swap.set_index(["nct_id", "endpoint"])["proba"]
                ref_common = orig_ref.index.intersection(swap_ref.index)
                ref_move = (orig_ref.loc[ref_common] - swap_ref.loc[ref_common]).abs()
                ref_insensitivity = float((ref_move < DISEASE_SWAP_MOVE_THRESHOLD).mean()) if len(ref_move) else None

            swap_result = {
                "n": len(common),
                "opus_insensitivity_rate": opus_insensitivity,
                "reference_insensitivity_rate": ref_insensitivity,
                "quarantine_ordering": (opus_insensitivity is not None
                                       and opus_insensitivity > DISEASE_SWAP_INSENSITIVITY_THRESHOLD),
                "note": "T22 came back INCONCLUSIVE on disease share, so low sensitivity is not "
                       "recall on its own -- only interpretable against the reference's "
                       "sensitivity on the same rows (docs/t28b_opus_recall_spec.md).",
            }

        branch, reason = decide(primary, reference_ab)

    artifact = {
        "test_id": "T28b",
        "claim_at_stake": "whether Opus 4.5's T28a predictive signal on TrialBench (mortality, "
                          "SAE) survives removal of outcome knowledge, holding trial identity "
                          "knowledge constant -- recall vs genuine reading",
        "inputs": {
            "model_key": model_key, "api_model_id": api_model_id, "cutoff": cutoff_str,
            "n_arm_a": n_arm_a, "n_arm_b": n_arm_b, "n_swap_requested": n_swap, "seed": seed,
            "data_root": data_root, "r1_text_cols": list(R1_TEXT_COLS), "r2_text_cols": list(R2_TEXT_COLS),
            "n_resamples": n_resamples,
        },
        "preflight": preflight,
        "n_sampled": {"arm_a": len(arm_a), "arm_b": len(arm_b), "arm_c": len(arm_c), "swap": len(arm_swap)},
        "primary_a_vs_b": {"opus": primary, "reference": reference_ab},
        "secondary_a_vs_c": {"opus": secondary_ac, "reference": reference_ac},
        "disease_swap": swap_result,
        "branch": branch, "branch_reason": reason,
        "meter": meter.summary(price_table, api_model_id, "sync"),
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(out_path, artifact)
    print(f"\nwrote {out_path}")
    print(f"branch={branch} ({reason})")
    return artifact


def decide(primary: dict, reference: dict) -> tuple:
    """docs/t28b_opus_recall_spec.md's primary decision table, on the A->B
    contrast. `primary`/`reference` are two_sample_cluster_bootstrap results
    (delta = A - B; a positive delta with a CI excluding 0 is a real drop
    from A to B)."""
    opus_drops = primary["lo"] > 0  # CI excludes 0, delta > 0 -- A significantly beats B
    ref_drops = reference["lo"] > 0

    if opus_drops and not ref_drops:
        return ("OUTCOME_RECALL_DEMONSTRATED",
                f"Opus's A->B drop clears its CI (delta={primary['mean_delta']:.3f}, "
                f"CI=[{primary['lo']:.3f}, {primary['hi']:.3f}]) and the reference's does not "
                f"(delta={reference['mean_delta']:.3f}, CI=[{reference['lo']:.3f}, {reference['hi']:.3f}]) "
                f"-- Opus's TrialBench figures are recall-inflated.")
    if opus_drops and ref_drops:
        return ("DISTRIBUTION_SHIFT_NOT_RECALL",
                f"Both drop comparably (Opus delta={primary['mean_delta']:.3f}, "
                f"reference delta={reference['mean_delta']:.3f}) -- Opus's arm orderings stand, "
                f"shift noted.")
    if not opus_drops and not ref_drops:
        return ("NO_OUTCOME_RECALL_DETECTED",
                f"Neither drops (Opus CI=[{primary['lo']:.3f}, {primary['hi']:.3f}], "
                f"reference CI=[{reference['lo']:.3f}, {reference['hi']:.3f}]) -- Opus's signal "
                f"is reading, not recall. T28 proceeds with Opus as rung 1 as designed.")
    return ("OPUS_MORE_ROBUST_THAN_REFERENCE",
            f"Opus does not drop (CI=[{primary['lo']:.3f}, {primary['hi']:.3f}]) but the "
            f"reference does (delta={reference['mean_delta']:.3f}) -- reported as-is, not a "
            f"contamination finding.")


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
    ap.add_argument("--n-swap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--model", default=OPUS_MODEL_KEY)
    ap.add_argument("--out-path", default=OUT_PATH)
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--preflight-only", action="store_true",
                    help="run only the zero-cost pre-flight checks and print them, no model calls")
    args = ap.parse_args()

    if args.preflight_only:
        price_table = load_price_table()
        cutoff = pd.Period(price_table["models"][args.model]["cutoff"], freq="M").end_time
        preflight, arm_a, arm_b, arm_c = run_preflight(args.data_root, cutoff, args.n_arm_a,
                                                        args.n_arm_b, args.seed, n_arm_c=args.n_arm_c)
        import json
        print(json.dumps(preflight, indent=2, default=str))
        return

    run(data_root=args.data_root, results_dir=args.results_dir, n_arm_a=args.n_arm_a,
        n_arm_b=args.n_arm_b, n_swap=args.n_swap, seed=args.seed, region=args.region,
        model_key=args.model, out_path=args.out_path, n_resamples=args.n_resamples,
        n_arm_c=args.n_arm_c)


if __name__ == "__main__":
    main()
