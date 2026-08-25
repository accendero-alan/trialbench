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

from ..data.features import DISEASE_TEXT_COLS, TEXT_COLS, concat_text, disease_blind_text
from .base import BaseMethod
from .registry import register

# Loaded lazily, cached at module scope so repeated fit()/predict_proba()
# calls across cells in the same run don't reload the same frozen model.
_ENCODER_CACHE = {}


def _get_encoder(model_name: str, revision: str = None):
    key = (model_name, revision)
    if key not in _ENCODER_CACHE:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModel.from_pretrained(model_name, revision=revision)
        model.eval()
        _ENCODER_CACHE[key] = (tok, model)
    return _ENCODER_CACHE[key]


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


class _TfidfLogRegBase(BaseMethod):
    """TF-IDF -> LogisticRegression over whatever ``_texts`` returns.
    Subclasses vary only in which text they hand the vectorizer -- the
    P10 text configurations (T22) are this with a narrower or scrubbed
    ``_texts``, everything else is shared.
    """
    feature_view = "raw"

    def _texts(self, X) -> list:
        raise NotImplementedError

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec_ = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                    min_df=2, sublinear_tf=True)
        Xtr = self.vec_.fit_transform(self._texts(X_train))
        self.clf_ = LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=2000, random_state=self.seed)
        self.clf_.fit(Xtr, y_train)
        return self

    def predict_proba(self, X):
        Xt = self.vec_.transform(self._texts(X))
        proba = self.clf_.predict_proba(Xt)
        if self.task_type == "binary":
            return self._binary_scores(proba)
        full = np.zeros((Xt.shape[0], self.num_classes), dtype=float)
        for j, c in enumerate(self.clf_.classes_):
            full[:, int(c)] = proba[:, j]
        return full


@register("tfidf_logreg")
class TfidfLogReg(_TfidfLogRegBase):
    def _texts(self, X) -> list:
        return concat_text(X)


@register("disease_text_only")
class DiseaseTextOnly(_TfidfLogRegBase):
    """P10 (T22 arm b): TF-IDF over only ``condition`` and
    ``condition_browse/mesh_term`` -- the disease-only text lower bound."""

    def _texts(self, X) -> list:
        return concat_text(X, text_cols=DISEASE_TEXT_COLS)


@register("disease_blind")
class DiseaseBlind(_TfidfLogRegBase):
    """P10 (T22 arm c): full ``concat_text`` with each row's own condition
    and MeSH-condition phrases scrubbed -- the disease-blind upper bound.
    Under-masks by construction (see ``disease_blind_text``'s docstring);
    ``mask_lists_`` holds the last call's per-row scrub list, for the T22
    artifact.
    """

    def _texts(self, X) -> list:
        texts, self.mask_lists_ = disease_blind_text(X)
        return texts


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

    POOLING = "mean"

    def _cache_path(self, X):
        # Folds in every knob that changes the resulting vectors -- model
        # name, MAX_LENGTH, pooling strategy, and the text-column set --
        # not just model name + row ids. Before this fix, changing
        # MAX_LENGTH (e.g. for a truncation ablation) silently reloaded
        # stale 256-token vectors from a still-matching cache path instead
        # of re-embedding, with no error (P3 in newsletter-part2-test-plan.md).
        text_cols_key = ",".join(TEXT_COLS)
        key_src = "|".join([
            self.MODEL_NAME, str(self.MAX_LENGTH), self.POOLING, text_cols_key,
            ",".join(sorted(str(i) for i in X.index)),
        ])
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


class SapBERTEncoder:
    """P11 (disease-representation-test-plan.md, T25/T26) infra: encodes
    condition strings to 768-d SapBERT vectors, cached **per unique string**
    (not per row-set like ``ClinicalEmbeddings`` -- condition strings repeat
    heavily across trials, and T25/T26 both need vectors for individual
    strings, not a fixed row-aligned matrix) under
    ``results/cache/sapbert/``. GPU (T25's budget note); falls back to CPU if
    none is available. Not a ``BaseMethod`` -- T25 composes this encoder's
    output with other blocks (multi-hot, MeSH graph embeddings) in ways that
    vary per arm, so it's a featurizer, built once here and wired up when
    T25 actually runs.
    """
    MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    # Pinned per P9/P11 standing rule 10 ("every external asset is pinned").
    # lastModified 2023-06-14 per the HF Hub API -- this model hasn't been
    # updated since, but pin the commit, not just the name, regardless.
    REVISION = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"
    MAX_LENGTH = 64  # condition strings are short (SapBERT's own recommendation)
    BATCH_SIZE = 64
    CACHE_DIR = os.path.join("results", "cache", "sapbert")
    POOLING = "mean"

    def _cache_path(self, text: str) -> str:
        key_src = "|".join([self.MODEL_NAME, self.REVISION, str(self.MAX_LENGTH), self.POOLING, text])
        key = hashlib.sha256(key_src.encode()).hexdigest()[:24]
        return os.path.join(self.CACHE_DIR, f"{key}.npy")

    def encode_strings(self, texts) -> dict:
        """``{text: 768-d np.ndarray}`` for every string in ``texts``,
        reading/populating the per-string cache. Encoding order is sorted,
        so a repeated call with a growing ``texts`` set re-batches
        deterministically."""
        unique = sorted(set(texts))
        out, to_encode, paths = {}, [], {}
        for t in unique:
            p = self._cache_path(t)
            if os.path.exists(p):
                out[t] = np.load(p)
            else:
                to_encode.append(t)
                paths[t] = p

        if to_encode:
            import torch

            tok, model = _get_encoder(self.MODEL_NAME, revision=self.REVISION)
            with torch.no_grad():
                for i in range(0, len(to_encode), self.BATCH_SIZE):
                    batch = to_encode[i:i + self.BATCH_SIZE]
                    enc = tok(batch, padding=True, truncation=True,
                              max_length=self.MAX_LENGTH, return_tensors="pt")
                    out_t = model(**enc)
                    pooled = _mean_pool(out_t.last_hidden_state, enc["attention_mask"]).numpy()
                    for t, vec in zip(batch, pooled):
                        vec = vec.astype(np.float32)
                        out[t] = vec
                        os.makedirs(self.CACHE_DIR, exist_ok=True)
                        tmp_path = paths[t] + ".tmp"
                        np.save(tmp_path, vec)
                        os.replace(tmp_path + ".npy", paths[t])
        return out

    def trial_vectors(self, condition_lists) -> np.ndarray:
        """``condition_lists``: one list of condition strings per trial ->
        ``(n, 768)``, each row the mean of that trial's condition-string
        vectors (all-zero if the trial has none -- pair with a
        ``has_condition`` presence control, per the T21 pattern)."""
        all_strings = [s for terms in condition_lists for s in terms]
        vecs = self.encode_strings(all_strings)
        dim = next(iter(vecs.values())).shape[0] if vecs else 768
        out = np.zeros((len(condition_lists), dim), dtype=np.float32)
        for i, terms in enumerate(condition_lists):
            rows = [vecs[t] for t in terms if t in vecs]
            if rows:
                out[i] = np.mean(rows, axis=0)
        return out
