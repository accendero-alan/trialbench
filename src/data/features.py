"""Build the dense **tabular view** used by classical / GBM / deep-tabular methods.

Mirrors TrialBench's own preprocessing spirit while staying dependency-light and
strictly leakage-safe (every transform is fit on TRAIN only):

- normalize age columns to months (repo's ``refine_year``);
- drop columns that are >50% missing on train;
- exclude the raw multimodal columns (text / SMILES / ICD / MeSH) — those belong
  to the "raw" view consumed by text/multimodal/LLM methods;
- numeric columns: coerce + median-impute;
- categorical columns:
    * binary task  -> smoothed target-mean encoding (≈ the repo's LeaveOneOut),
    * multiclass   -> one-hot with a cardinality cap;
- multi-hot columns (``MaskingType-*``, ``ipd_info_type-*``): pass through, NaN->0.

The text / SMILES / ICD / MeSH columns are intentionally NOT in this view;
methods that want them use ``feature_view = "raw"`` and read the DataFrame.
"""
from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd

from .mol_features import vocab_matrix

# As-shipped column list (kept as a *set* originally, which made
# ``concat_text``'s join order vary between processes -- see P1 in
# newsletter-part2-test-plan.md). "brief_summary" and "detailed_description"
# are also missing the "/textblock" suffix the actual CSV columns carry, so
# those two fields silently never make it into the text view at all (T7 in
# the test plan measures what that cost). This exact list is preserved here,
# unfixed, as the historical "as-shipped" comparison point -- do not repair
# the names in this constant; the live default below is where the repair
# lands once T7 confirms it.
TEXT_COLS_AS_SHIPPED = (
    "brief_title", "brief_summary", "detailed_description",
    "eligibility/study_pop/textblock", "eligibility/criteria/textblock",
    "intervention/description", "keyword",
    "study_design_info/intervention_model_description",
    "study_design_info/masking_description", "condition",
)

# Live default: an ordered tuple (not a set) so column order -- and therefore
# concat_text's joined string -- is deterministic across processes/seeds, with
# the name typo repaired (T7 in the test plan): "brief_summary" ->
# "brief_summary/textblock", "detailed_description" -> "detailed_description/textblock"
# (present only on trial-failure-reason-identification; concat_text already
# skips columns absent from a given task's frame). T7 measured the repair's
# effect at ~+0.003 to +0.036 PR-AUC per cell -- below the T1 noise floor on
# 19/20 cells, real only on failure_reason/Phase2 (+0.0285) -- but it's landed
# regardless because it's the objectively correct column mapping, and T8/T9
# depend on the tabular view no longer double-counting these columns as
# categoricals.
TEXT_COLS = tuple(
    {"brief_summary": "brief_summary/textblock",
     "detailed_description": "detailed_description/textblock"}.get(c, c)
    for c in TEXT_COLS_AS_SHIPPED
)
MOLECULE_COLS = {"smiless", "intervention/intervention_name"}
DISEASE_COLS = {"icdcode"}
AGE_COLS = ["eligibility/minimum_age", "eligibility/maximum_age"]


def refine_year(x):
    """Convert a duration string (e.g. '2 Years') to months; pass numbers through."""
    if isinstance(x, str):
        try:
            n = float(x.split(" ")[0])
        except (ValueError, IndexError):
            return np.nan
        if "Year" in x:
            return n * 12
        if "Month" in x:
            return n
        if "Week" in x:
            return n / 4.286
        if "Day" in x:
            return n / 30
        if "Hour" in x:
            return n / 30 / 24
        if "Minute" in x:
            return n / 30 / 24 / 60
    return x


def _is_multihot(col: str) -> bool:
    return ("MaskingType-" in col) or ("ipd_info_type-" in col)


def _is_raw_multimodal(col: str) -> bool:
    return (col in TEXT_COLS or col in MOLECULE_COLS or col in DISEASE_COLS
            or "mesh_term" in col)


class TabularFeaturizer:
    def __init__(self, task_type: str = "binary", max_onehot_card: int = 40,
                 numeric_parse_thresh: float = 0.7, smoothing: float = 20.0):
        self.task_type = task_type
        self.max_onehot_card = max_onehot_card
        self.numeric_parse_thresh = numeric_parse_thresh
        self.smoothing = smoothing

    # -- fit --------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X = self._prep_ages(X.copy())
        n = len(X)

        # candidate columns: drop raw-multimodal + >50% missing
        cols = [c for c in X.columns if not _is_raw_multimodal(c)]
        cols = [c for c in cols if X[c].isna().sum() <= 0.5 * n]

        self.multihot_ = [c for c in cols if _is_multihot(c)]
        rest = [c for c in cols if c not in self.multihot_]

        numeric, categorical = [], []
        for c in rest:
            coerced = pd.to_numeric(X[c], errors="coerce")
            if coerced.notna().mean() >= self.numeric_parse_thresh:
                numeric.append(c)
            else:
                categorical.append(c)
        self.numeric_ = numeric
        self.categorical_ = categorical

        # numeric medians
        self.medians_ = {c: float(pd.to_numeric(X[c], errors="coerce").median()) for c in numeric}
        self.medians_ = {c: (v if np.isfinite(v) else 0.0) for c, v in self.medians_.items()}

        # categorical encoders
        y = np.asarray(y)
        self.cat_encoders_ = {}
        self.onehot_levels_ = {}
        if self.task_type == "binary":
            self.global_mean_ = float(np.mean(y)) if len(y) else 0.0
            for c in categorical:
                s = X[c].astype("object").where(X[c].notna(), "__nan__").astype(str)
                df = pd.DataFrame({"cat": s.values, "y": y})
                agg = df.groupby("cat")["y"].agg(["sum", "count"])
                enc = (agg["sum"] + self.smoothing * self.global_mean_) / (agg["count"] + self.smoothing)
                self.cat_encoders_[c] = enc.to_dict()
        else:
            for c in categorical:
                s = X[c].astype("object").where(X[c].notna(), "__nan__").astype(str)
                levels = list(pd.Index(s.unique()))
                if len(levels) <= self.max_onehot_card:
                    self.onehot_levels_[c] = levels
                # else: dropped (too high cardinality for one-hot)

        self.feature_names_ = self._build_names()
        return self

    def _build_names(self):
        names = list(self.numeric_) + list(self.multihot_)
        if self.task_type == "binary":
            names += [f"{c}__te" for c in self.categorical_]
        else:
            for c, levels in self.onehot_levels_.items():
                names += [f"{c}__{lv}" for lv in levels]
        return names

    # -- transform --------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._prep_ages(X.copy())
        blocks = []

        # numeric
        for c in self.numeric_:
            col = pd.to_numeric(X.get(c), errors="coerce") if c in X else pd.Series([np.nan] * len(X))
            blocks.append(col.fillna(self.medians_[c]).to_numpy(dtype=float).reshape(-1, 1))

        # multi-hot
        for c in self.multihot_:
            col = pd.to_numeric(X.get(c), errors="coerce") if c in X else pd.Series([0] * len(X))
            blocks.append(col.fillna(0).to_numpy(dtype=float).reshape(-1, 1))

        # categorical
        if self.task_type == "binary":
            for c in self.categorical_:
                s = (X[c].astype("object").where(X[c].notna(), "__nan__").astype(str)
                     if c in X else pd.Series(["__nan__"] * len(X)))
                enc = self.cat_encoders_[c]
                vals = s.map(enc).fillna(self.global_mean_).to_numpy(dtype=float)
                blocks.append(vals.reshape(-1, 1))
        else:
            for c, levels in self.onehot_levels_.items():
                s = (X[c].astype("object").where(X[c].notna(), "__nan__").astype(str)
                     if c in X else pd.Series(["__nan__"] * len(X)))
                oh = np.zeros((len(X), len(levels)), dtype=float)
                idx = {lv: i for i, lv in enumerate(levels)}
                for r, v in enumerate(s.values):
                    if v in idx:
                        oh[r, idx[v]] = 1.0
                blocks.append(oh)

        if not blocks:
            return np.zeros((len(X), 0), dtype=float)
        M = np.hstack(blocks)
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    # -- helpers ----------------------------------------------------------
    def _prep_ages(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in AGE_COLS:
            if c in X.columns:
                X[c] = X[c].apply(refine_year)
        return X


def concat_text(X: pd.DataFrame, text_cols=None) -> list:
    """Concatenate available free-text columns into one string per row (raw view).

    ``text_cols`` defaults to the module-level ``TEXT_COLS``; pass an explicit
    ordered sequence (e.g. ``TEXT_COLS_AS_SHIPPED`` or a repaired variant) to
    probe alternate text configurations without mutating shared state.
    """
    cols = TEXT_COLS if text_cols is None else text_cols
    present = [c for c in cols if c in X.columns]
    if not present:
        return ["" for _ in range(len(X))]
    return (X[present].fillna("").astype(str).agg(" ".join, axis=1)).tolist()


# ----------------------------------------------------------------------------
# T21 (newsletter-part2-test-plan.md continuation, t21-code-channel-plan.md):
# ICD / MeSH code channel, currently dropped by ``_is_raw_multimodal`` from the
# default tabular view. ``CodeFeaturizer`` is deliberately separate from
# ``TabularFeaturizer`` and does not alter its output in any way (P5) -- the
# default path must keep reproducing today's leaderboard exactly.
# ----------------------------------------------------------------------------
def _recursive_parse_terms(value) -> list:
    """Parse a (possibly multiply-nested) stringified list into a flat list of
    scalar term strings. ``icdcode`` entries are commonly a list whose single
    element is itself a stringified list, e.g. ``["['J45.998', 'J82.83']"]`` --
    a single ``ast.literal_eval`` returns one opaque token. This recurses until
    the leaves are scalars, dropping ``"None"``/``"nan"``/``""`` at every level
    (present in the data and not real terms).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if s in ("", "None", "nan", "NaN", "[]"):
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return [s]  # scalar leaf -- a real term (e.g. "Filariasis")
    if isinstance(parsed, (list, tuple)):
        out = []
        for item in parsed:
            out.extend(_recursive_parse_terms(item))
        return out
    s2 = str(parsed).strip()
    return [] if s2 in ("", "None", "nan", "NaN") else [s2]


def _icd_chapter(code: str) -> str:
    """Roll an ICD-10-CM code up to its 3-character chapter, e.g. 'J45.998' -> 'J45'."""
    return code.strip()[:3].upper()


class CodeFeaturizer:
    """Three separate multi-hot code blocks -- ICD chapters, MeSH condition
    terms, MeSH intervention terms -- never concatenated into one vocabulary,
    so per-block importances and ablations stay readable (per the plan).

    Each block gets a per-row ``n_<block>_terms`` count and ``has_<block>``
    indicator (the presence controls, mirroring ``mol_features.aggregate``'s
    ``[has_molecule, n_molecules]`` -- the same control that made the
    chemistry probe's conclusions legible). Vocabulary is fit on train only at
    ``min_df`` frequency; terms unseen at valid/test are silently dropped
    (standard multi-hot behavior).
    """

    BLOCKS = ("icd", "mesh_cond", "mesh_int")
    SOURCE_COLS = {
        "icd": "icdcode",
        "mesh_cond": "condition_browse/mesh_term",
        "mesh_int": "intervention_browse/mesh_term",
    }

    def __init__(self, min_df: int = 10, icd_rollup: bool = True):
        self.min_df = min_df
        self.icd_rollup = icd_rollup

    def _parse_block(self, X: pd.DataFrame, block: str) -> list:
        col = self.SOURCE_COLS[block]
        if col not in X.columns:
            return [[] for _ in range(len(X))]
        raw = X[col].values
        if block == "icd" and self.icd_rollup:
            return [sorted({_icd_chapter(c) for c in _recursive_parse_terms(v) if c.strip()})
                    for v in raw]
        return [sorted(set(_recursive_parse_terms(v))) for v in raw]

    def fit(self, X: pd.DataFrame) -> "CodeFeaturizer":
        self.vocabs_ = {}
        for block in self.BLOCKS:
            terms_lists = self._parse_block(X, block)
            counts = Counter(t for terms in terms_lists for t in terms)
            self.vocabs_[block] = sorted(t for t, c in counts.items() if c >= self.min_df)
        self.feature_names_ = self._build_names()
        return self

    def _build_names(self) -> list:
        names = []
        for block in self.BLOCKS:
            names += [f"{block}__{t}" for t in self.vocabs_[block]]
            names += [f"n_{block}_terms", f"has_{block}"]
        return names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        blocks_out = []
        for block in self.BLOCKS:
            terms_lists = self._parse_block(X, block)
            mh = vocab_matrix(terms_lists, self.vocabs_[block])
            n_terms = np.array([len(t) for t in terms_lists], dtype=float).reshape(-1, 1)
            has = (n_terms > 0).astype(float)
            blocks_out.append(np.hstack([mh, n_terms, has]))
        return np.hstack(blocks_out) if blocks_out else np.zeros((len(X), 0))

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)
