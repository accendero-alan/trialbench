"""Tier B: deep tabular methods (STUBS).

Each runs on the dense **tabular** view. Implement one at a time, keep networks
small and cap epochs (CPU!), then enable in configs/benchmark.yaml. The runner
records an unimplemented method as "skipped" so leaving these as stubs never
breaks a core run.
"""
from __future__ import annotations

from .base import BaseMethod
from .registry import register


@register("tabpfn")
class TabPFN(BaseMethod):
    """Tabular foundation model. Near-instant fit; CPU-runnable within limits
    (~<=10k train rows, <=500 features, <=10 classes for v2).

    Plan:
      from tabpfn import TabPFNClassifier
      self.model_ = TabPFNClassifier(device="cpu")
      # subsample train if it exceeds the row limit; reduce features via the
      # featurizer or a quick SelectKBest if > limit.
      self.model_.fit(X_train, y_train); proba = self.model_.predict_proba(X)
    """
    feature_view = "tabular"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        raise NotImplementedError("tabpfn stub — see docstring; pip install tabpfn.")

    def predict_proba(self, X):
        raise NotImplementedError


@register("tabnet")
class TabNet(BaseMethod):
    """pytorch-tabnet TabNetClassifier. Use eval_set=(X_valid,y_valid) for early
    stopping; keep n_d/n_a small and max_epochs modest on CPU."""
    feature_view = "tabular"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        raise NotImplementedError("tabnet stub — pip install pytorch-tabnet.")

    def predict_proba(self, X):
        raise NotImplementedError


@register("ft_transformer")
class FTTransformer(BaseMethod):
    """FT-Transformer / MLP / ResNet via the `rtdl` library (torch, CPU ok)."""
    feature_view = "tabular"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        raise NotImplementedError("ft_transformer stub — pip install rtdl torch.")

    def predict_proba(self, X):
        raise NotImplementedError
