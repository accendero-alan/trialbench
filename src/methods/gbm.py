"""Tier A: gradient-boosted trees — the SOTA tabular workhorses.

XGBoost / LightGBM / CatBoost are imported lazily inside ``fit`` so a missing
package yields a clean ImportError the runner records as "skipped (missing dep)"
rather than crashing the whole benchmark.
"""
from __future__ import annotations

import numpy as np

from .base import BaseMethod
from .registry import register


class _GBMBase(BaseMethod):
    feature_view = "tabular"

    def predict_proba(self, X):
        proba = self.model_.predict_proba(X)
        if self.task_type == "binary":
            return self._binary_scores(proba)
        full = np.zeros((len(X), self.num_classes), dtype=float)
        for j, c in enumerate(self._classes):
            full[:, int(c)] = proba[:, j]
        return full


@register("xgboost")
class XGBoost(_GBMBase):
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        from xgboost import XGBClassifier
        kw = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=self.params.get("n_jobs", -1),
                  random_state=self.seed, eval_metric="logloss", tree_method="hist")
        if self.task_type == "binary":
            pos = float(np.sum(y_train == 1)); neg = float(np.sum(y_train == 0))
            kw["scale_pos_weight"] = (neg / pos) if pos > 0 else 1.0
        else:
            kw.update(objective="multi:softprob", num_class=self.num_classes)
        self.model_ = XGBClassifier(**kw)
        self.model_.fit(X_train, y_train)
        self._classes = self.model_.classes_
        return self


@register("lightgbm")
class LightGBM(_GBMBase):
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        from lightgbm import LGBMClassifier
        kw = dict(n_estimators=600, num_leaves=63, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                  n_jobs=self.params.get("n_jobs", -1), random_state=self.seed)
        self.model_ = LGBMClassifier(**kw)
        self.model_.fit(X_train, y_train)
        self._classes = self.model_.classes_
        return self


@register("catboost")
class CatBoost(_GBMBase):
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        from catboost import CatBoostClassifier
        kw = dict(iterations=600, depth=6, learning_rate=0.05,
                  random_seed=self.seed, verbose=False, thread_count=self.params.get("n_jobs", -1),
                  auto_class_weights="Balanced",
                  # H1 (wave1-preflight-review.md): unset, every concurrent CatBoost
                  # fit writes to one shared ./catboost_info/ in the CWD, and
                  # concurrent processes interleave writes/cleanup there. Nothing
                  # in this repo reads those logs.
                  allow_writing_files=False)
        self.model_ = CatBoostClassifier(**kw)
        self.model_.fit(X_train, y_train)
        self._classes = self.model_.classes_
        return self
