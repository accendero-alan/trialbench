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

import numpy as np
import pandas as pd

TEXT_COLS = {
    "brief_title", "brief_summary", "detailed_description",
    "eligibility/study_pop/textblock", "eligibility/criteria/textblock",
    "intervention/description", "keyword",
    "study_design_info/intervention_model_description",
    "study_design_info/masking_description", "condition",
}
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


def concat_text(X: pd.DataFrame) -> list:
    """Concatenate available free-text columns into one string per row (raw view)."""
    present = [c for c in TEXT_COLS if c in X.columns]
    if not present:
        return ["" for _ in range(len(X))]
    return (X[present].fillna("").astype(str).agg(" ".join, axis=1)).tolist()
