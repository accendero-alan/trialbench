"""Tier D: multimodal + repo baseline (STUBS).

- ``fingerprint_fusion``: CPU-friendly multimodal stand-in. Morgan/ECFP
  fingerprints from ``smiless`` (RDKit) + mean MeSH/ICD embedding + the tabular
  matrix -> concatenate -> gradient boosting.
- ``hint_reference``: wrap TrialBench's own HINT-style ``MultiModel`` so its
  numbers appear in the same leaderboard (true apples-to-apples reference).
"""
from __future__ import annotations

from .base import BaseMethod
from .registry import register


@register("fingerprint_fusion")
class FingerprintFusion(BaseMethod):
    """Plan (raw view):
      1. tabular matrix via TabularFeaturizer;
      2. RDKit Morgan fingerprints (nBits=1024) from parsed `smiless`, averaged
         over multiple drugs per trial;
      3. mean MeSH embedding (load mesh_embeddings.txt.gz) + ICD presence bag;
      4. hstack all blocks -> LightGBM/XGBoost.
    Cache fingerprints/embeddings per task×phase.
    """
    feature_view = "raw"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        raise NotImplementedError("fingerprint_fusion stub — pip install rdkit; see docstring.")

    def predict_proba(self, X):
        raise NotImplementedError


@register("hint_reference")
class HintReference(BaseMethod):
    """Wrap the repo's MultiModel (Trialbench/models/model_multi.py).

    Plan:
      - clone ML2ClinicalTrials, install its requirements.txt (torch 1.13.1, cpu);
      - reuse its dataset.load_data(task, phase, data_format='dl') dataloaders;
      - build MultiModel with the multimodal encoders, `learn`, then read the
        test probabilities so metrics come from THIS harness (not its bootstrap).
    Slow on CPU — run on a capped subset first. feature_view left as 'raw';
    this method mostly bypasses the shared featurizer and drives the repo code.
    """
    feature_view = "raw"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        raise NotImplementedError("hint_reference stub — see docstring and PLAN.md §3 Tier D.")

    def predict_proba(self, X):
        raise NotImplementedError
