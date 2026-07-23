"""Tier A/C: text methods over the trial free-text fields (raw view).

- ``tfidf_logreg`` (Tier A, always available): TF-IDF over concatenated trial
  text -> Logistic Regression. Cheap and a strong text-only baseline.
- ``clinical_embeddings`` (Tier C, stub): frozen BioBERT/ClinicalBERT/PubMedBERT
  embeddings (precompute once, cache) -> linear classifier.
"""
from __future__ import annotations

import numpy as np

from ..data.features import concat_text
from .base import BaseMethod
from .registry import register


@register("tfidf_logreg")
class TfidfLogReg(BaseMethod):
    feature_view = "raw"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec_ = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                    min_df=2, sublinear_tf=True)
        Xtr = self.vec_.fit_transform(concat_text(X_train))
        self.clf_ = LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=2000, random_state=self.seed)
        self.clf_.fit(Xtr, y_train)
        return self

    def predict_proba(self, X):
        Xt = self.vec_.transform(concat_text(X))
        proba = self.clf_.predict_proba(Xt)
        if self.task_type == "binary":
            return self._binary_scores(proba)
        full = np.zeros((Xt.shape[0], self.num_classes), dtype=float)
        for j, c in enumerate(self.clf_.classes_):
            full[:, int(c)] = proba[:, j]
        return full


@register("clinical_embeddings")
class ClinicalEmbeddings(BaseMethod):
    """STUB (Tier C). Implement, then enable in configs/benchmark.yaml.

    Plan:
      1. Load a frozen clinical encoder (e.g. 'emilyalsentzer/Bio_ClinicalBERT'
         or 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract').
      2. Encode concat_text(X) -> fixed vectors. CPU inference is slow, so
         embed ONCE per task×phase and cache to results/cache/<hash>.npy.
      3. Train a LogisticRegression / small MLP on the cached vectors.
    Set feature_view = "raw" and use src.data.features.concat_text for input.
    """
    feature_view = "raw"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        raise NotImplementedError(
            "clinical_embeddings is a Tier C stub — see docstring and PLAN.md §3/§4. "
            "Install transformers/sentence-transformers (requirements-extended.txt)."
        )

    def predict_proba(self, X):
        raise NotImplementedError
