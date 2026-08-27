"""P13.4 (wave2-start-plan.md): the Bedrock response cache.

Keyed on ``(model_id, prompt_sha256, temperature, service_tier)`` -- service
tier is part of the key because a batched and a synchronous call to the same
model at the same temperature are priced differently and, for T31, are
different experimental conditions (a router call is always synchronous; a
ladder member's T28 prediction may be batched).

The cache root is an **absolute path derived from the results directory the
caller passes in**, never a bare relative literal -- the predecessor
(``src/methods/llm_backend.py``'s ``CACHE_ROOT = os.path.join("results",
"cache", "llm")``) was CWD-relative, H3's bug class (wave1-preflight-review.md):
whichever directory the process happened to be launched from silently became
part of the cache key's storage location, so two launches from different
CWDs against the same nominal ``results_dir`` missed each other's cache
entries.

Name this ``response_cache`` in call sites and artifacts, not just
"the cache" -- Bedrock's own prompt caching (off; see the plan §2, prompts
run 750-1,000 tokens, below every current minimum checkpoint) is a different
thing, and the two are easy to conflate.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _safe_model_dirname(model_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in model_id)


def cache_root(results_dir: str) -> str:
    """``results_dir`` may itself be relative (as ``--results-dir`` commonly
    is); ``os.path.abspath`` resolves it against the process's CWD exactly
    once, here, so every subsequent path built from the returned root is
    already absolute and can't drift if something downstream changes
    directory."""
    return os.path.join(os.path.abspath(results_dir), "cache", "bedrock")


def cache_path(results_dir: str, model_id: str, prompt_hash: str, temperature: float,
              service_tier: str) -> str:
    root = cache_root(results_dir)
    fname = f"{prompt_hash}_t{temperature}_{service_tier}.json"
    return os.path.join(root, _safe_model_dirname(model_id), fname)


def cache_get(results_dir: str, model_id: str, prompt: str, temperature: float,
             service_tier: str) -> dict | None:
    path = cache_path(results_dir, model_id, prompt_sha256(prompt), temperature, service_tier)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cache_put(results_dir: str, model_id: str, prompt: str, temperature: float,
             service_tier: str, response: dict) -> str:
    """Atomic write with a process-unique temp suffix (H3, wave1-preflight-review.md:
    a fixed ``.tmp`` name let two concurrent writers race and publish a
    truncated file). ``os.replace`` is atomic within one filesystem, so a
    kill mid-write leaves only a stray temp file, never a corrupted cache
    entry. The rendered prompt is stored beside the hash so a template edit
    is detectable by inspection and so a cached call can be audited
    byte-for-byte without re-running the pipeline that produced it."""
    path = cache_path(results_dir, model_id, prompt_sha256(prompt), temperature, service_tier)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {
        "model_id": model_id, "prompt": prompt, "temperature": temperature,
        "service_tier": service_tier, "response": response,
    }
    tmp_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp_path, path)
    return path
