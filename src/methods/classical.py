"""Tier A: classical / linear scikit-learn methods (always available, CPU-native).

All operate on the dense **tabular** view. A shared ``SklearnMethod`` handles
proba shaping for binary (-> 1-D P(y=1)) and multiclass (-> (n, C)).
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from .base import BaseMethod
from .registry import register


class SklearnMethod(BaseMethod):
    feature_view = "tabular"

    def _build(self):  # -> sklearn estimator with predict_proba
        raise NotImplementedError

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model_ = self._build()
        self.classes_ = np.unique(y_train)
        self.model_.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        proba = self.model_.predict_proba(X)
        if self.task_type == "binary":
            return self._binary_scores(proba)
        # align to full class set 0..num_classes-1
        full = np.zeros((len(X), self.num_classes), dtype=float)
        for j, c in enumerate(self.model_.classes_):
            full[:, int(c)] = proba[:, j]
        return full


@register("majority")
class Majority(SklearnMethod):
    def _build(self):
        return DummyClassifier(strategy="prior")


@register("logreg_l2")
class LogRegL2(SklearnMethod):
    def _build(self):
        return make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                               max_iter=2000, random_state=self.seed),
        )


@register("logreg_l1")
class LogRegL1(SklearnMethod):
    def _build(self):
        return make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                               class_weight="balanced", max_iter=2000, random_state=self.seed),
        )


@register("random_forest")
class RandomForest(SklearnMethod):
    def _build(self):
        return RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=self.seed)


@register("extra_trees")
class ExtraTrees(SklearnMethod):
    def _build(self):
        return ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=self.seed)


@register("hist_gbm")
class HistGBM(SklearnMethod):
    def _build(self):
        return HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=300, random_state=self.seed)


@register("knn")
class KNN(SklearnMethod):
    def _build(self):
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, n_jobs=-1))


@register("svm_linear")
class SVMLinear(SklearnMethod):
    def _build(self):
        # LinearSVC has no predict_proba -> calibrate.
        base = LinearSVC(class_weight="balanced", random_state=self.seed)
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(base, cv=3))
