"""The method contract every model implements.

A method takes a train (and optional validation) set and produces class
probabilities on a test set. Feature views:

- ``"tabular"``: a dense numeric matrix (numpy array) produced by
  :class:`src.data.features.TabularFeaturizer`. Used by classical, GBM, and
  deep-tabular methods.
- ``"raw"``: the original pandas DataFrame with all multimodal columns
  (text, SMILES, ICD codes, MeSH). Used by text/NLP, multimodal, and LLM
  methods that do their own encoding.
- ``"tabular+codes"`` / ``"codes"`` (T21, newsletter-part2-test-plan.md
  continuation): not a method-declared view -- no registered method sets
  ``feature_view`` to either of these. They're produced by ``run_cell``
  (src/run_benchmark.py) when ``feature_view_override`` is configured, which
  swaps in :class:`src.data.features.CodeFeaturizer` output (horizontally
  stacked with ``TabularFeaturizer`` for ``"tabular+codes"``, alone for
  ``"codes"``) for any method whose *declared* view is ``"tabular"`` --
  letting the same 12 Tier A method names run twice and pair exactly, without
  renaming them or duplicating method classes. ``"raw"`` methods are always
  left untouched by the override.

``predict_proba`` returns:
- binary tasks: a 1-D array of shape (n,) giving P(y = 1);
- multiclass tasks: a 2-D array of shape (n, num_classes).
"""
from __future__ import annotations

import numpy as np


class BaseMethod:
    #: registry key (set by @register)
    name: str = "base"
    #: "tabular" or "raw" -- see module docstring for the run-time-only
    #: "tabular+codes" / "codes" views ``feature_view_override`` can swap in.
    feature_view: str = "tabular"

    def __init__(self, task_type: str = "binary", num_classes: int = 2, seed: int = 42, **params):
        self.task_type = task_type          # "binary" | "multiclass"
        self.num_classes = num_classes
        self.seed = seed
        self.params = params

    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> "BaseMethod":
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    def _binary_scores(self, proba_2d: np.ndarray) -> np.ndarray:
        """Reduce an sklearn (n, 2) proba to a 1-D P(y=1)."""
        proba_2d = np.asarray(proba_2d)
        if proba_2d.ndim == 1:
            return proba_2d
        return proba_2d[:, 1]
