"""P13.1, P13.3 (wave2-start-plan.md): the Bedrock Converse client and the
verbalized-probability elicitation path.

Converse, not InvokeModel: ``InvokeModel`` cannot report a router's invoked
model id (no equivalent to ``trace.promptRouter.invokedModelId``), so every
call in this harness -- routed or not -- goes through Converse for a
consistent response shape.

Verbalized probability is the only elicitation path implemented here.
Bedrock does not expose per-token logprobs over user-supplied text for the
Anthropic or Nova models; DeepSeek V3.2 and Llama 4 Maverick are unchecked
(W1.8, one Converse call each) and are a one-time upside if they return them
-- not built out speculatively here. ``primary_elicitation="logprob"``
raises rather than silently falling back, so a caller can't end up scoring a
model on an elicitation path nobody verified exists for it.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

from .cache import cache_get, cache_put

_THROTTLE_CODES = {"ThrottlingException", "ServiceUnavailableException"}
_FATAL_CODES = {"ValidationException"}

# Substring markers for a refused (vs. merely malformed) response. Coarse by
# design -- refusal-rate is a per-model quality signal T30 plots, not a
# safety classifier, so a false positive here costs a mislabeled row, not a
# missed detection.
_REFUSAL_MARKERS = ("i cannot", "i can't", "i'm unable", "as an ai", "i won't", "i will not")


class BedrockCallError(RuntimeError):
    pass


class ThrottleBudgetExceededError(BedrockCallError):
    pass


@dataclass
class ConverseResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_secs: float
    invoked_model_id: str
    retry_count: int = 0
    raw_response: dict = field(default_factory=dict)


def _error_code(exc: Exception) -> str | None:
    resp = getattr(exc, "response", None)
    if not resp:
        return None
    return resp.get("Error", {}).get("Code")


class BedrockClient:
    """Wraps ``boto3``'s ``bedrock-runtime`` Converse API.

    ``boto_client`` is an injection point: tests pass a fake object exposing
    a ``.converse(**kwargs)`` method instead of constructing a real
    ``boto3.client("bedrock-runtime")``, the same dependency-injection shape
    the predecessor local-server-backed client got via its ``base_url`` parameter.
    ``boto3`` is imported lazily inside ``__init__`` (only when no fake is
    injected) so this repo's CPU-only Tier A methods never need it installed
    -- golden rule 5, CLAUDE.md.
    """

    def __init__(self, region: str = "us-west-2", boto_client=None,
                max_retries: int = 6, backoff_base_secs: float = 1.0,
                backoff_max_secs: float = 60.0, max_throttle_fraction: float = 0.5):
        if boto_client is not None:
            self._client = boto_client
        else:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=region)
        self.max_retries = max_retries
        self.backoff_base_secs = backoff_base_secs
        self.backoff_max_secs = backoff_max_secs
        self.max_throttle_fraction = max_throttle_fraction
        self._calls = 0
        self._throttled_calls = 0

    @property
    def throttle_rate(self) -> float:
        return self._throttled_calls / self._calls if self._calls else 0.0

    def converse(self, model_id_or_router_arn: str, prompt: str, temperature: float = 0.0,
                max_tokens: int = 32) -> ConverseResult:
        """P13.1: retries with exponential backoff and full jitter on
        ``ThrottlingException``/``ServiceUnavailableException``;
        ``ValidationException`` is fatal (fails the call, not retried --
        retrying a malformed request just burns quota for the same result).
        Fails the whole run once the throttle rate exceeds
        ``max_throttle_fraction`` rather than continuing to hammer a rate
        limit -- W1's quota numbers say the real run spends hours near the
        ceiling, so throttling itself is expected and only a *sustained*
        rate is a problem worth stopping for.
        """
        self._calls += 1
        if self._calls > 20 and self.throttle_rate > self.max_throttle_fraction:
            raise ThrottleBudgetExceededError(
                f"throttle rate {self.throttle_rate:.1%} exceeds the configured "
                f"{self.max_throttle_fraction:.0%} over {self._calls} calls -- failing the "
                f"run rather than continuing to hammer a rate limit (P13.1)."
            )

        payload = dict(
            modelId=model_id_or_router_arn,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
        )

        retry_count = 0
        this_call_throttled = False
        t0 = time.time()
        while True:
            try:
                resp = self._client.converse(**payload)
                break
            except Exception as e:  # noqa: BLE001 -- botocore.exceptions.ClientError in production
                code = _error_code(e)
                if code in _FATAL_CODES:
                    raise BedrockCallError(
                        f"fatal Bedrock error ({code}) for {model_id_or_router_arn}: {e}"
                    ) from e
                if code in _THROTTLE_CODES and retry_count < self.max_retries:
                    this_call_throttled = True
                    retry_count += 1
                    delay = min(self.backoff_base_secs * (2 ** (retry_count - 1)), self.backoff_max_secs)
                    delay *= 0.5 + random.random()   # full jitter around the exponential envelope
                    time.sleep(delay)
                    continue
                raise BedrockCallError(
                    f"Bedrock call to {model_id_or_router_arn} failed after {retry_count} "
                    f"retries ({code}): {e}"
                ) from e
        latency = time.time() - t0
        if this_call_throttled:
            self._throttled_calls += 1

        usage = resp.get("usage", {})
        content = resp.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content)
        invoked_model_id = (
            resp.get("trace", {}).get("promptRouter", {}).get("invokedModelId")
            or model_id_or_router_arn
        )

        return ConverseResult(
            text=text, input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)), latency_secs=latency,
            invoked_model_id=invoked_model_id, retry_count=retry_count, raw_response=resp,
        )


def _parse_verbalized(text: str) -> float | None:
    """Batch inference supports neither structured output nor tool calling
    (P13.3), so the JSON contract is enforced by prompt and parser here, not
    an API parameter -- this must work identically for sync and batch
    responses."""
    try:
        obj = json.loads(text)
        p = float(obj["probability"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not (0 <= p <= 100):
        return None
    return p / 100.0


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


@dataclass
class ElicitationResult:
    primary_prob: float          # 0-1; 0.5 on parse failure (P13.3)
    parse_ok: bool
    refused: bool
    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_secs: float = 0.0
    invoked_model_id: str = ""
    cache_hit: bool = False


def elicit_verbalized_probability(client: BedrockClient, results_dir: str, model_id: str,
                                  prompt: str, temperature: float,
                                  meter=None) -> ElicitationResult:
    """The only *synchronous* elicitation path (verbalized 0-100 JSON,
    temperature 0), per revision 3 §6.3. Cache-checked first (P13.4); a
    cache hit still re-parses the stored text rather than trusting a stored
    probability, so a parser fix retroactively reinterprets old cache
    entries correctly on the next read.

    Always ``service_tier="sync"`` for both the cache key and the meter --
    hardcoded, not threaded from a caller, because this function performs a
    real-time ``Converse`` call every time it actually reaches the network;
    there is no way for it to be anything else. (Threading a caller-supplied
    tier through here is exactly the shape of bug found 2026-08-27,
    wave2-start-plan.md P13.8's status note: a cell configured for "batch"
    had every one of its calls actually run this function, tagged "batch"
    anyway. Batch submission is a different code path entirely --
    :mod:`src.bedrock.batch_formats` / :func:`src.methods.llm.LLMProbability._predict_batch`
    -- that never calls this function for the rows it successfully
    batches; it only falls back to this function, honestly, for rows below
    the per-model batch minimum.)
    """
    cached = cache_get(results_dir, model_id, prompt, temperature, "sync")
    if cached is not None:
        if meter is not None:
            meter.record_cache_hit()
        resp = cached["response"]
        text = resp.get("text", "")
        prob = _parse_verbalized(text)
        return ElicitationResult(
            primary_prob=prob if prob is not None else 0.5, parse_ok=prob is not None,
            refused=_looks_like_refusal(text), raw_text=text,
            input_tokens=resp.get("input_tokens", 0), output_tokens=resp.get("output_tokens", 0),
            invoked_model_id=resp.get("invoked_model_id", model_id), cache_hit=True,
        )

    result = client.converse(model_id, prompt, temperature=temperature)
    if meter is not None:
        meter.record_call(result.input_tokens, result.output_tokens, result.latency_secs,
                          throttle_count=result.retry_count, service_tier="sync")
    cache_put(results_dir, model_id, prompt, temperature, "sync", {
        "text": result.text, "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens, "invoked_model_id": result.invoked_model_id,
    })
    prob = _parse_verbalized(result.text)
    return ElicitationResult(
        primary_prob=prob if prob is not None else 0.5, parse_ok=prob is not None,
        refused=_looks_like_refusal(result.text), raw_text=result.text,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        latency_secs=result.latency_secs, invoked_model_id=result.invoked_model_id,
    )


def elicit_probability(client: BedrockClient, results_dir: str, model_id: str, prompt: str,
                       temperature: float, primary: str, meter=None) -> ElicitationResult:
    if primary != "verbalized":
        raise NotImplementedError(
            f"primary_elicitation={primary!r} is not available -- Bedrock exposes no logprob "
            f"path for Anthropic/Nova, and W1.8 (the DeepSeek V3.2 / Llama 4 Maverick logprob "
            f"check) hasn't run. 'verbalized' is the only implemented path; see client.py's "
            f"module docstring and wave2-start-plan.md §6.3."
        )
    return elicit_verbalized_probability(client, results_dir, model_id, prompt, temperature, meter=meter)
