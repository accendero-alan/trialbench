"""Tier D: LLM-based classification (STUB).

Not CPU-bound — it's API/cost/rate limited, so cap the sample. Serialize each
trial's structured fields to a compact text record and ask a frontier model for
a calibrated probability (few-shot or zero-shot with reasoning).
"""
from __future__ import annotations

from .base import BaseMethod
from .registry import register


@register("llm_fewshot")
class LLMFewShot(BaseMethod):
    """Plan (raw view):
      1. Serialize key fields (phase, condition, intervention, enrollment,
         eligibility summary, sponsor class, ...) to a short text record.
      2. Few-shot prompt with k balanced examples drawn from TRAIN only.
      3. Parse a probability from the response; average a few samples for
         stability. Cache responses keyed by NCT id + prompt hash.
    Respect --max_test_rows to bound cost. Use the `anthropic` SDK.
    """
    feature_view = "raw"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        # "fit" = stash few-shot exemplars from train.
        raise NotImplementedError("llm_fewshot stub — see docstring and PLAN.md §3 Tier D.")

    def predict_proba(self, X):
        raise NotImplementedError
