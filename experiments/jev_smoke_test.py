"""Jev smoke test -- can T28a's two recall probes run on a decisions model?

Jev (typesafe/jev-1.13-20260917, via OpenRouter) is not a chat model: it
answers typed questions (choice / score / noul) on /api/alpha/decisions and
emits no free text, so T28a's probes are adapted, not reused:

- Outcome recall: T28a's recall framing and its L7 rendered trial as
  `state`, asked as a `noul` (0-1 probability) instead of Yes/No text.
- Title recall: free-text title from a bare NCT id is impossible, so it
  becomes multiple-choice recognition -- NCT id alone, pick the true
  `brief_title` among 4 decoy TrialBench titles (chance 0.2).

Plus 3 repeated outcome calls, since Jev exposes no temperature and
determinism is unknown.

This is a plumbing check before any spend decision: parse behaviour,
tokens, cost, latency. n=20 settles nothing about contamination.

Usage:
    python -m experiments.jev_smoke_test --n-trials 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

from experiments._common import git_sha
from experiments.t28a_contamination_probes import _pool_trials
from src.data.serialize import render_arm
from src.methods.llm import TASK_QUESTIONS

MODEL = "typesafe/jev-1.13-20260917"
URL = "https://openrouter.ai/api/alpha/decisions"
OUT_PATH = "results/experiments/jev_smoke_test.json"
N_DECOYS = 4
N_REPEATS = 3
LETTERS = "ABCDE"


def _decide(state: str, questions: dict) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
               "Content-Type": "application/json", "X-Title": "accendero-jev-smoke"}
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers=headers),
                                    timeout=120) as f:
            status, payload = f.status, json.load(f)
    except urllib.error.HTTPError as e:
        status, payload = e.code, {"error_body": e.read().decode(errors="replace")[:1000]}
    except Exception as e:  # network / timeout -- recorded, never retried silently
        status, payload = None, {"error_body": repr(e)}
    return {"status": status, "latency_s": round(time.time() - t0, 3), "payload": payload}


def outcome_probe(task: str, trial_text: str) -> dict:
    instructions = ("This clinical trial has already completed and its results have been reported. "
                    f"Based on what you know about this specific trial, state whether "
                    f"{TASK_QUESTIONS[task]}.")
    call = _decide(trial_text, {"outcome": {"type": "noul", "instructions": instructions}})
    ans = (call["payload"].get("answers") or {}).get("outcome") or {}
    val = ans.get("noul")
    parsed = isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0
    return {**call, "noul": float(val) if parsed else None, "parsed": parsed}


def title_probe(nct_id: str, true_title: str, decoys: list, rng) -> dict:
    titles = [true_title] + decoys
    order = rng.permutation(len(titles))
    criteria = {LETTERS[i]: titles[j] for i, j in enumerate(order)}
    true_letter = LETTERS[int(np.where(order == 0)[0][0])]
    call = _decide(f"ClinicalTrials.gov trial {nct_id}",
                   {"title": {"type": "choice",
                              "instructions": f"Which of these is the official title of the "
                                              f"ClinicalTrials.gov trial {nct_id}?",
                              "criteria": criteria}})
    ans = (call["payload"].get("answers") or {}).get("title") or {}
    choice, probs = ans.get("choice"), ans.get("probabilities") or {}
    parsed = choice in criteria and set(probs) == set(criteria)
    return {**call, "true_letter": true_letter, "choice": choice, "probabilities": probs,
            "confidence": ans.get("confidence"), "parsed": parsed,
            "hit": (choice == true_letter) if parsed else None,
            "p_true": float(probs[true_letter]) if parsed else None}


def _usage(call: dict) -> dict:
    u = call["payload"].get("usage") or {}
    return {"input_tokens": u.get("input_tokens") or 0, "output_tokens": u.get("output_tokens") or 0,
            "cost": u.get("cost") or 0.0}


def run(data_root="data", n_trials=20, seed=42):
    rng = np.random.default_rng(seed)
    sample = _pool_trials(data_root, n_trials, seed)
    decoy_pool = _pool_trials(data_root, 500, seed + 1)
    decoy_pool = decoy_pool[~decoy_pool["nct_id"].isin(sample["nct_id"])]
    decoy_titles = decoy_pool["brief_title"].dropna().drop_duplicates().tolist()

    per_trial, calls = [], []
    for _, row in sample.iterrows():
        rendered = render_arm(row, "L7", row["nct_id"])
        o = outcome_probe(row["task"], rendered.text)
        decoys = list(rng.choice(decoy_titles, size=N_DECOYS, replace=False))
        t = title_probe(row["nct_id"], row["brief_title"], decoys, rng)
        calls += [o, t]
        per_trial.append({
            "nct_id": row["nct_id"], "task": row["task"], "y": int(row["y"]),
            "outcome_status": o["status"], "outcome_noul": o["noul"], "outcome_parsed": o["parsed"],
            "outcome_hit": (round(o["noul"]) == int(row["y"])) if o["parsed"] else None,
            "outcome_usage": _usage(o), "outcome_latency_s": o["latency_s"],
            "outcome_error": None if o["parsed"] else o["payload"],
            "title_true": row["brief_title"], "title_status": t["status"],
            "title_true_letter": t["true_letter"], "title_choice": t["choice"],
            "title_probabilities": t["probabilities"], "title_confidence": t["confidence"],
            "title_parsed": t["parsed"], "title_hit": t["hit"], "title_p_true": t["p_true"],
            "title_usage": _usage(t), "title_latency_s": t["latency_s"],
            "title_error": None if t["parsed"] else t["payload"],
            "model_returned": o["payload"].get("model"),
        })

    repeats = []
    for rec, (_, row) in zip(per_trial[:N_REPEATS], sample.head(N_REPEATS).iterrows()):
        r = outcome_probe(row["task"], render_arm(row, "L7", row["nct_id"]).text)
        calls.append(r)
        repeats.append({"nct_id": rec["nct_id"], "first": rec["outcome_noul"], "repeat": r["noul"]})

    usage = [_usage(c) for c in calls]
    o_ok = [p for p in per_trial if p["outcome_parsed"]]
    t_ok = [p for p in per_trial if p["title_parsed"]]
    summary = {
        "calls": len(calls),
        "http_status_counts": {str(k): sum(c["status"] == k for c in calls)
                               for k in sorted({c["status"] for c in calls}, key=str)},
        "outcome_parse_rate": len(o_ok) / len(per_trial),
        "title_parse_rate": len(t_ok) / len(per_trial),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "cost_usd": round(sum(u["cost"] for u in usage), 8),
        "input_tokens_per_outcome_call": float(np.mean([p["outcome_usage"]["input_tokens"] for p in per_trial])),
        "input_tokens_per_title_call": float(np.mean([p["title_usage"]["input_tokens"] for p in per_trial])),
        "latency_s_median": float(np.median([c["latency_s"] for c in calls])),
        "outcome_noul_values": sorted({p["outcome_noul"] for p in o_ok}),
        "outcome_hit_rate": float(np.mean([p["outcome_hit"] for p in o_ok])) if o_ok else None,
        "title_hit_rate": float(np.mean([p["title_hit"] for p in t_ok])) if t_ok else None,
        "title_hits": int(sum(p["title_hit"] for p in t_ok)),
        "title_chance": 1 / (N_DECOYS + 1),
        "title_mean_p_true": float(np.mean([p["title_p_true"] for p in t_ok])) if t_ok else None,
        "title_choice_letter_counts": {L: sum(p["title_choice"] == L for p in t_ok) for L in LETTERS},
        "determinism": repeats,
        "deterministic": all(r["first"] == r["repeat"] for r in repeats),
    }
    artifact = {
        "test_id": "jev_smoke_test", "model": MODEL, "endpoint": URL, "n_trials": n_trials,
        "seed": seed, "n_decoys": N_DECOYS, "git_sha": git_sha(),
        "claim_at_stake": "Plumbing only: whether T28a's outcome and title recall probes, adapted to a "
                          "decisions model, parse reliably and what they cost. Not a contamination finding.",
        "summary": summary, "per_trial": per_trial,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1, default=str)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(run(a.data_root, a.n_trials, a.seed)["summary"], indent=1, default=str))
