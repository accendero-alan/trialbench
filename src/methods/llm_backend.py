"""P13.3-P13.5 (wave2-start-plan.md): the pieces `llm_probability`
(``llm.py``) is built on -- an HTTP client for a llama.cpp ``llama-server``
instance (W1's chosen serving stack: native Windows CUDA build, no WSL/Linux
dependency, exposes per-token logprobs and grammar/``json_schema``-constrained
decoding directly on its ``/completion`` endpoint), a response cache, and a
per-arm/per-model usage meter. Kept separate from ``llm.py`` so the method
class itself (the ``BaseMethod`` contract) stays readable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

CACHE_ROOT = os.path.join("results", "cache", "llm")

# Case/whitespace variants a tokenizer commonly splits "yes"/"no" into
# (leading-space forms are typical for BPE tokenizers mid-sentence). Matched
# against llama-server's top-k candidate list at the yes/no decision token.
_YES_VARIANTS = {"yes", " yes", "Yes", " Yes", "YES", " YES"}
_NO_VARIANTS = {"no", " no", "No", " No", "NO", " NO"}


# ----------------------------------------------------------------------------
# P13.4: response cache
# ----------------------------------------------------------------------------
def _safe_model_dirname(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in model)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def cache_path(model: str, prompt_hash: str, temperature: float, cache_root: str = CACHE_ROOT) -> str:
    return os.path.join(cache_root, _safe_model_dirname(model), f"{prompt_hash}_t{temperature}.json")


def cache_get(model: str, prompt: str, temperature: float, cache_root: str = CACHE_ROOT) -> dict | None:
    path = cache_path(model, prompt_sha256(prompt), temperature, cache_root)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    # prompt_sha256 already guarantees no stale hit on a template edit (the
    # hash is a function of the prompt text); the stored prompt is for
    # legibility -- so a cached call can be audited byte-for-byte later
    # without needing to re-run the pipeline that produced it.
    return rec


def cache_put(model: str, prompt: str, temperature: float, response: dict, cache_root: str = CACHE_ROOT) -> str:
    """Atomic write with a process-unique temp suffix (H3, wave1-preflight-review.md:
    a fixed ``.tmp`` name let two concurrent writers race and publish a
    truncated file). ``os.replace`` is atomic within one filesystem, so a kill
    mid-write leaves only a stray temp file, never a corrupted cache entry."""
    path = cache_path(model, prompt_sha256(prompt), temperature, cache_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"model": model, "prompt": prompt, "temperature": temperature, "response": response}
    tmp_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp_path, path)
    return path


# ----------------------------------------------------------------------------
# P13.5: the meter
# ----------------------------------------------------------------------------
@dataclass
class Meter:
    """Per-arm/per-model usage, accumulated across calls in one run. Local
    models cost $0.00 by construction; ``project_api_cost`` gives the
    separate, explicitly-computed figure P13.5 wants for the same call
    volume against a candidate API model's published per-token rate, so the
    dollar comparison at the actual GPU-seconds/call counts, not the plan's
    prior estimate."""
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_secs: float = 0.0

    def record_call(self, input_tokens: int, output_tokens: int, wall_clock_secs: float) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.wall_clock_secs += wall_clock_secs

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.calls + self.cache_hits
        return self.cache_hits / total if total else 0.0

    def project_api_cost(self, rate_per_1k_input: float, rate_per_1k_output: float) -> float:
        return (self.input_tokens / 1000) * rate_per_1k_input + (self.output_tokens / 1000) * rate_per_1k_output

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "gpu_seconds": round(self.wall_clock_secs, 2),
            "dollars": 0.00,  # local models only, per the 2026-08-26 decision (wave2-start-plan.md)
        }


# ----------------------------------------------------------------------------
# P13.3: the llama-server client and the two elicitation paths
# ----------------------------------------------------------------------------
class LlamaServerError(RuntimeError):
    pass


@dataclass
class ElicitationResult:
    verbalized_prob: float | None   # 0-1, from the parsed 0-100 verbalized number
    logprob_prob: float | None      # 0-1, from the yes/no token logprob
    primary_prob: float             # whichever of the above is configured primary; 0.5 on total parse failure
    parse_ok: bool
    raw_response: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


def _post(base_url: str, path: str, payload: dict, timeout: float = 120.0) -> dict:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise LlamaServerError(f"llama-server call to {url} failed: {e}") from e


_VERBALIZED_JSON_SCHEMA = {
    "type": "object",
    "properties": {"probability": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["probability"],
}


def _parse_verbalized(content: str) -> float | None:
    try:
        obj = json.loads(content)
        p = obj["probability"]
        p = float(p)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not (0 <= p <= 100):
        return None
    return p / 100.0


def elicit_verbalized(base_url: str, prompt: str, model: str, temperature: float = 0.0) -> tuple:
    """Ask for a JSON ``{"probability": 0-100}`` with a schema-constrained
    decode (grammar built from ``_VERBALIZED_JSON_SCHEMA`` server-side), per
    P13.3. Returns ``(prob_or_None, response_dict)``."""
    payload = {
        "prompt": prompt, "temperature": temperature, "n_predict": 32,
        "json_schema": _VERBALIZED_JSON_SCHEMA,
    }
    resp = _post(base_url, "/completion", payload)
    return _parse_verbalized(resp.get("content", "")), resp


def _extract_logprob_prob(resp: dict) -> float | None:
    """P(yes) from a llama-server ``/completion`` response's per-token
    candidate list (``n_probs``): find "yes"/"no" among the top-k candidates
    at the single decision token and softmax over just those two -- restricted
    to {yes, no}, not calibrated against the full vocabulary distribution.
    ``None`` if neither token appears in the top-k (parse failure)."""
    probs_list = resp.get("completion_probabilities") or []
    if not probs_list:
        return None
    candidates = probs_list[0].get("probs") or probs_list[0].get("top_logprobs") or []
    yes_lp = no_lp = None
    for cand in candidates:
        tok = cand.get("tok_str", cand.get("token", ""))
        lp = cand.get("logprob") if "logprob" in cand else math.log(max(cand.get("prob", 0.0), 1e-12))
        if tok in _YES_VARIANTS and yes_lp is None:
            yes_lp = lp
        elif tok in _NO_VARIANTS and no_lp is None:
            no_lp = lp
    if yes_lp is None and no_lp is None:
        return None
    yes_lp = yes_lp if yes_lp is not None else -50.0
    no_lp = no_lp if no_lp is not None else -50.0
    m = max(yes_lp, no_lp)
    return math.exp(yes_lp - m) / (math.exp(yes_lp - m) + math.exp(no_lp - m))


def elicit_logprob(base_url: str, prompt: str, model: str, temperature: float = 0.0, n_probs: int = 20) -> tuple:
    """Single-token continuation at a fixed yes/no cue, per P13.3's second
    path: the exact token logprob for "yes" vs "no", softmaxed into
    P(yes). Returns ``(prob_or_None, response_dict)``. Needs the serving
    stack to expose per-token candidate logprobs (``n_probs``) -- verified
    at W1, not re-verified per call."""
    payload = {
        "prompt": prompt, "temperature": temperature, "n_predict": 1, "n_probs": n_probs,
    }
    resp = _post(base_url, "/completion", payload)
    return _extract_logprob_prob(resp), resp


def elicit_probability(base_url: str, prompt_verbalized: str, prompt_logprob: str, model: str,
                       primary: str, temperature: float = 0.0,
                       cache_root: str = CACHE_ROOT, meter: Meter | None = None) -> ElicitationResult:
    """Both paths, every call, per the §3 amendment: compute and store both,
    primary chosen from config. Each path has its own cache entry (they're
    different prompts/requests). Parse failure -> that path's prob is
    ``None``; if the *primary* path fails, ``primary_prob`` falls back to
    0.5 and ``parse_ok`` is False, which the method-level caller aggregates
    into the run's parse-failure rate (P13.3's >2% warning)."""
    if primary not in ("verbalized", "logprob"):
        raise ValueError(f"primary must be 'verbalized' or 'logprob', got {primary!r}")

    def _call(prompt, elicit_fn):
        cached = cache_get(model, prompt, temperature, cache_root)
        if cached is not None:
            if meter is not None:
                meter.record_cache_hit()
            resp = cached["response"]
            if elicit_fn is elicit_verbalized:
                return _parse_verbalized(resp.get("content", "")), resp
            return _extract_logprob_prob(resp), resp
        t0 = time.time()
        prob, resp = elicit_fn(base_url, prompt, model, temperature)
        wall = time.time() - t0
        cache_put(model, prompt, temperature, resp, cache_root)
        if meter is not None:
            timings = resp.get("timings", {})
            in_tok = int(timings.get("prompt_n", resp.get("tokens_evaluated", 0)) or 0)
            out_tok = int(timings.get("predicted_n", resp.get("tokens_predicted", 0)) or 0)
            meter.record_call(in_tok, out_tok, wall)
        return prob, resp

    verbalized_prob, verbalized_resp = _call(prompt_verbalized, elicit_verbalized)
    logprob_prob, logprob_resp = _call(prompt_logprob, elicit_logprob)

    primary_prob = verbalized_prob if primary == "verbalized" else logprob_prob
    parse_ok = primary_prob is not None
    if primary_prob is None:
        primary_prob = 0.5

    return ElicitationResult(
        verbalized_prob=verbalized_prob, logprob_prob=logprob_prob, primary_prob=primary_prob,
        parse_ok=parse_ok, raw_response={"verbalized": verbalized_resp, "logprob": logprob_resp},
    )
