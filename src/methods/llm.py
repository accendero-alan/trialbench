"""P13.1 (wave2-start-plan.md): ``llm_probability``, driven entirely by
config the way ``--icd-granularity`` drives the code rungs -- T28 runs
*through* ``run_benchmark.py``, not around it, so resume, per-fit prediction
persistence, the leaderboard, the pairing convention and the B3 guard all come
for free, and T28's analysis script reads the same ``results/predictions/``
layout T22-T24 already read.

Zero-shot only, per the plan's design: ``fit`` asserts it wasn't handed any
few-shot exemplars rather than quietly ignoring a ``llm_fewshot_k`` param
that would otherwise do nothing.
"""
from __future__ import annotations

import numpy as np

from ..data.serialize import ARMS, render_arm
from .base import BaseMethod
from .llm_backend import Meter, elicit_probability
from .registry import register

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_PRIMARY_ELICITATION = "logprob"  # per the §3 amendment's recommendation

# Per-task question the trial record is judged against. "outcome" here is
# TrialBench's trial-approval-forecasting task (loader.py TASKS), not the
# generic English word.
TASK_QUESTIONS = {
    "mortality_rate_yn": "an all-cause mortality event will occur among this trial's participants",
    "serious_adverse_rate_yn": "a serious adverse event will occur among this trial's participants",
    "patient_dropout_rate_yn": "this trial will experience a patient dropout event",
    "outcome": "this trial's outcome will be a success (approved / positive result)",
}


def _build_verbalized_prompt(question: str, trial_text: str) -> str:
    return (
        "You are a clinical trial risk assessor. Read the trial record below and estimate "
        f"the probability (0-100) that {question}.\n\n"
        f"Trial record:\n{trial_text}\n\n"
        'Respond with ONLY a JSON object: {"probability": <integer 0-100>}.'
    )


def _build_logprob_prompt(question: str, trial_text: str) -> str:
    return (
        "You are a clinical trial risk assessor. Read the trial record below and answer Yes or No: "
        f"is it true that {question}?\n\n"
        f"Trial record:\n{trial_text}\n\n"
        "Answer (Yes/No):"
    )


@register("llm_probability")
class LLMProbability(BaseMethod):
    feature_view = "raw"

    def __init__(self, task_type: str = "binary", num_classes: int = 2, seed: int = 42, **params):
        super().__init__(task_type=task_type, num_classes=num_classes, seed=seed, **params)
        self.task = params.get("task")
        self.arm = params.get("llm_arm")
        self.model = params.get("llm_model")
        self.temperature = params.get("llm_temperature", 0.0)
        self.primary_elicitation = params.get("primary_elicitation", DEFAULT_PRIMARY_ELICITATION)
        self.max_calls = params.get("llm_max_calls")
        self.base_url = params.get("llm_base_url", DEFAULT_BASE_URL)
        self.llm_meter = Meter()
        self.llm_n_scored = 0
        self.llm_n_parse_failures = 0

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if not self.arm or not self.model:
            raise NotImplementedError(
                "llm_probability requires --llm-arm and --llm-model (P13.1); neither was configured."
            )
        if self.arm not in ARMS:
            raise ValueError(f"unknown --llm-arm {self.arm!r}, expected one of {ARMS}")
        if self.params.get("llm_fewshot_k"):
            raise NotImplementedError(
                "llm_probability is zero-shot (verbalized probability / logprob) per the plan's "
                "design -- llm_fewshot_k exemplars are not implemented, not silently ignored."
            )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.task_type != "binary":
            raise NotImplementedError("llm_probability is scoped to binary tasks; multiclass is out of scope.")
        if self.max_calls is not None and len(X) > self.max_calls:
            raise ValueError(
                f"predict_proba got {len(X)} rows but --llm-max-calls={self.max_calls}. "
                f"Reduce the sample via --test-subset-file or --max-test-rows instead of silently "
                f"truncating -- a truncated proba array would break row alignment with y_test."
            )
        question = TASK_QUESTIONS.get(self.task)
        if question is None:
            raise NotImplementedError(
                f"no LLM question template for task {self.task!r}; add one to TASK_QUESTIONS."
            )

        probs = []
        for nct_id, row in X.iterrows():
            rendered = render_arm(row, self.arm, str(nct_id))
            prompt_v = _build_verbalized_prompt(question, rendered.text)
            prompt_l = _build_logprob_prompt(question, rendered.text)
            result = elicit_probability(
                self.base_url, prompt_v, prompt_l, self.model,
                primary=self.primary_elicitation, temperature=self.temperature,
                meter=self.llm_meter,
            )
            self.llm_n_scored += 1
            if not result.parse_ok:
                self.llm_n_parse_failures += 1
            probs.append(result.primary_prob)
        return np.asarray(probs, dtype=float)

    @property
    def llm_parse_failure_rate(self) -> float:
        return self.llm_n_parse_failures / self.llm_n_scored if self.llm_n_scored else 0.0
