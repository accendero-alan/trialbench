"""Tier A/C: text methods over the trial free-text fields (raw view).

- ``tfidf_logreg`` (Tier A, always available): TF-IDF over concatenated trial
  text -> Logistic Regression. Cheap and a strong text-only baseline.
- ``clinical_embeddings`` (Tier C): frozen clinical BERT embeddings
  (mean-pooled, precomputed once and cached to disk) -> linear classifier.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from ..data.features import concat_text
from .base import BaseMethod
from .registry import register

# Loaded lazily, cached at module scope so repeated fit()/predict_proba()
# calls across cells in the same run don't reload the same frozen model.
_ENCODER_CACHE = {}


def _get_encoder(model_name: str):
    if model_name not in _ENCODER_CACHE:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        _ENCODER_CACHE[model_name] = (tok, model)
    return _ENCODER_CACHE[model_name]


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


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
    """Frozen clinical BERT embeddings (mean-pooled over token vectors,
    masked by attention) -> LogisticRegression. The encoder is never
    fine-tuned -- CPU inference is the slow part, so each unique set of rows
    is embedded once and cached to ``results/cache/clinical_embeddings/`` by
    a hash of (model name, row NCT ids); a re-run or a shared task/phase with
    overlapping trials reuses the cache instead of re-embedding.
    """
    feature_view = "raw"

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    CACHE_DIR = os.path.join("results", "cache", "clinical_embeddings")

    def _cache_path(self, X):
        key_src = self.MODEL_NAME + "|" + ",".join(sorted(str(i) for i in X.index))
        key = hashlib.sha256(key_src.encode()).hexdigest()[:24]
        return os.path.join(self.CACHE_DIR, f"{key}.npy")

    def _embed(self, X):
        cache_path = self._cache_path(X)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        import torch

        tok, model = _get_encoder(self.MODEL_NAME)
        texts = concat_text(X)
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), self.BATCH_SIZE):
                batch = texts[i:i + self.BATCH_SIZE]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=self.MAX_LENGTH, return_tensors="pt")
                out = model(**enc)
                pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
                chunks.append(pooled.numpy())
        emb = np.concatenate(chunks, axis=0).astype(np.float32)

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        tmp_path = cache_path + ".tmp"
        np.save(tmp_path, emb)
        os.replace(tmp_path + ".npy", cache_path)  # np.save appends ".npy"
        return emb

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        Xtr = self._embed(X_train)
        self.clf_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=2000, random_state=self.seed),
        )
        self.clf_.fit(Xtr, y_train)
        return self

    def predict_proba(self, X):
        Xt = self._embed(X)
        proba = self.clf_.predict_proba(Xt)
        if self.task_type == "binary":
            return self._binary_scores(proba)
        full = np.zeros((Xt.shape[0], self.num_classes), dtype=float)
        for j, c in enumerate(self.clf_.classes_):
            full[:, int(c)] = proba[:, j]
        return full
