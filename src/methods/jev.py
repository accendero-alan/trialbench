"""Zero-shot Jev (TypeSafe), via OpenRouter's decisions API.

``jev_zeroshot`` never trains: ``fit`` is a no-op, and ``predict_proba``
asks Jev one ``noul`` question per trial on the same L7 render and the
same risk-assessor framing as its T28b run (experiments/t28b_jev.py), where
contamination was cleared. A row Jev does not answer scores 0.5 and is
logged, per P13.

Jev is a decisions model, not a chat model: it returns a typed 0-1 value
and no text, takes no temperature, and was seen to move by up to 0.01 on
repeats. Responses are cached by request hash in
``results/cache/jev_decisions.jsonl``, shared with the T28a/T28b scripts.
Calls need ``OPENROUTER_API_KEY``; the key is never logged.

Per-call accounting goes to ``<results_dir>/jev_meta/<task>.jsonl``, one
line per predict_proba call, because run_cell has no generic hook for a
method's own metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..data.serialize import render_arm
from .base import BaseMethod
from .llm import TASK_QUESTIONS
from .registry import register

MODEL = "typesafe/jev-1.13-20260917"
URL = "https://openrouter.ai/api/alpha/decisions"
CACHE_PATH = os.path.join("results", "cache", "jev_decisions.jsonl")
WORKERS = 6
RETRIES = 3

_lock = threading.Lock()
_cache: dict = {}
_cache_loaded = False


def _load_cache():
    global _cache_loaded
    with _lock:
        if _cache_loaded:
            return
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    _cache[rec["key"]] = rec["call"]
        _cache_loaded = True


def _post(state: str, questions: dict) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
               "Content-Type": "application/json", "X-Title": "accendero-trialbench"}
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers=headers), timeout=120) as f:
            status, payload = f.status, json.load(f)
    except urllib.error.HTTPError as e:
        status, payload = e.code, {"error_body": e.read().decode(errors="replace")[:1000]}
    except Exception as e:  # network / timeout
        status, payload = None, {"error_body": repr(e)}
    return {"status": status, "latency_s": round(time.time() - t0, 3), "payload": payload}


def decide(state: str, questions: dict) -> dict:
    """Cached call; transient failures (no status, 429, 5xx) retried with backoff."""
    _load_cache()
    key = hashlib.sha256(json.dumps([MODEL, state, questions], sort_keys=True).encode()).hexdigest()
    with _lock:
        if key in _cache:
            return {**_cache[key], "cache_hit": True, "retries": 0}
    retries = 0
    call = _post(state, questions)
    while (call["status"] is None or call["status"] == 429 or call["status"] >= 500) and retries < RETRIES:
        time.sleep(2 ** (retries + 1))
        retries += 1
        call = _post(state, questions)
    if call["status"] == 200:
        with _lock:
            _cache[key] = call
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "call": call}) + "\n")
    return {**call, "cache_hit": False, "retries": retries}


@register("jev_zeroshot")
class JevZeroShot(BaseMethod):
    feature_view = "raw"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if self.task_type != "binary":
            raise NotImplementedError("jev_zeroshot asks a noul (0-1) question; binary tasks only")
        self.task_ = self.params.get("task")
        if self.task_ not in TASK_QUESTIONS:
            raise NotImplementedError(f"no question template for task {self.task_!r}")
        return self

    def _instructions(self) -> str:
        return ("You are a clinical trial risk assessor. Read the trial record and estimate the "
                f"probability that {TASK_QUESTIONS[self.task_]}.")

    def _score(self, item):
        nct_id, row = item
        call = decide(render_arm(row, "L7", str(nct_id)).text,
                      {"risk": {"type": "noul", "instructions": self._instructions()}})
        val = ((call["payload"].get("answers") or {}).get("risk") or {}).get("noul")
        ok = isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0
        return (float(val) if ok else 0.5), ok, call, str(nct_id)

    def predict_proba(self, X):
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            out = list(ex.map(self._score, list(X.iterrows())))
        proba = np.array([o[0] for o in out], dtype=float)
        calls = [o[2] for o in out]
        usage = [c["payload"].get("usage") or {} for c in calls]
        meta = {
            "task": self.task_, "model": MODEL, "n_rows": len(out),
            "first_nct_id": out[0][3] if out else None,
            "parse_failures": [o[3] for o in out if not o[1]],
            "cache_hits": sum(bool(c.get("cache_hit")) for c in calls),
            "retries": sum(c.get("retries", 0) for c in calls),
            "http_status_counts": {str(k): sum(c["status"] == k for c in calls)
                                   for k in sorted({c["status"] for c in calls}, key=str)},
            "input_tokens": int(sum(u.get("input_tokens") or 0 for u in usage)),
            "output_tokens": int(sum(u.get("output_tokens") or 0 for u in usage)),
            "dollars_realized": round(sum((u.get("cost") or 0.0) for u, c in zip(usage, calls)
                                          if not c.get("cache_hit")), 6),
        }
        self.parse_failure_rate_ = len(meta["parse_failures"]) / max(1, len(out))
        meta_dir = os.path.join(self.params.get("results_dir") or "results", "jev_meta")
        os.makedirs(meta_dir, exist_ok=True)
        with _lock, open(os.path.join(meta_dir, f"{self.task_}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        return proba
