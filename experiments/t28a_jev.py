"""T28a for Jev -- the contamination gate, adapted to a decisions model.

Jev (typesafe/jev-1.13-20260917, OpenRouter /api/alpha/decisions) emits no
free text, so T28a's two recall probes are adapted (see jev_smoke_test.py,
whose 20-trial run found the multiple-choice format dominated by position
bias: option A chosen 17/20):

- Outcome recall: T28a's recall framing and L7 trial text, asked as a
  `noul` (0-1). Binarised at 0.5 for T28a's per-task balanced accuracy and
  Fisher exact; AUROC reported beside it.
- Title recognition: NCT id alone, true `brief_title` + 4 decoy TrialBench
  titles, asked 5 times with the option list cyclically rotated so every
  title occupies every slot once. Score = mean probability on the true
  title across rotations (p_true); chance is 0.2 and position bias cancels.
- Shuffled-ID control: the identical rotated option set, asked under a
  DIFFERENT sampled trial's NCT id (a derangement). Same score. This floor
  catches a true title that wins on plausibility rather than recall.

Sample: T28a's own `_pool_trials(data, 200, seed=42)`, the same 200 trials
Opus 4.5 ran on, so per-task figures compare directly.

Decision rule, fixed before the run:
- Primary: paired Wilcoxon signed-rank on p_true(real) - p_true(shuffled),
  one-sided (real > shuffled), alpha 0.01 (T28a's title-recall alpha).
  Clears it -> recognition demonstrated -> RECALL_SHOWN: no Jev TrialBench
  number is quoted until T28b-style arms separate recall from reading.
- Otherwise, per task: balanced accuracy vs 0.5, Fisher exact on the 2x2,
  read at both T28a's family alpha (0.0025, 20 comparisons) and a
  Jev-only Bonferroni (0.0125, 4 tasks). Any task clearing it -> predictive
  signal that T28a cannot attribute to reading or recall -> NOT_CLEARED_
  NEEDS_T28B. None -> NO_SIGNAL.
- A null on title recognition is weak evidence (it does not exclude
  recognising a fully described trial); it never on its own clears Jev.

Usage:
    python -m experiments.t28a_jev --n-trials 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import binomtest, fisher_exact, wilcoxon
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from experiments._common import git_sha
from experiments.jev_smoke_test import MODEL, URL, _decide
from experiments.t28a_contamination_probes import _pool_trials
from src.data.serialize import render_arm
from src.methods.llm import TASK_QUESTIONS

OUT_PATH = "results/experiments/t28a_jev.json"
CACHE_PATH = "results/cache/jev_decisions.jsonl"
N_DECOYS = 4
LETTERS = "ABCDE"
TITLE_ALPHA = 0.01
FAMILY_ALPHA_T28A = 0.0025
JEV_ONLY_ALPHA = 0.05 / 4
N_BOOT = 2000
WORKERS = 6

_lock = threading.Lock()
_cache: dict = {}


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                _cache[rec["key"]] = rec["call"]


def _cached_decide(state: str, questions: dict) -> dict:
    key = hashlib.sha256(json.dumps([MODEL, state, questions], sort_keys=True).encode()).hexdigest()
    with _lock:
        if key in _cache:
            return {**_cache[key], "cache_hit": True}
    call = _decide(state, questions)
    if call["status"] == 200:  # only successes are cached; failures stay visible and re-runnable
        with _lock:
            _cache[key] = call
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "call": call}) + "\n")
    return {**call, "cache_hit": False}


def _outcome(task: str, trial_text: str) -> dict:
    instructions = ("This clinical trial has already completed and its results have been reported. "
                    f"Based on what you know about this specific trial, state whether "
                    f"{TASK_QUESTIONS[task]}.")
    call = _cached_decide(trial_text, {"outcome": {"type": "noul", "instructions": instructions}})
    val = ((call["payload"].get("answers") or {}).get("outcome") or {}).get("noul")
    ok = isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0
    return {"call": call, "noul": float(val) if ok else None}


def _title_rotation(nct_id: str, options: list, r: int) -> dict:
    """options[0] is the scored title. Rotation r puts options[k] in slot (k + r) % 5."""
    slots = [None] * len(options)
    for k, title in enumerate(options):
        slots[(k + r) % len(options)] = title
    criteria = {LETTERS[i]: t for i, t in enumerate(slots)}
    call = _cached_decide(f"ClinicalTrials.gov trial {nct_id}",
                          {"title": {"type": "choice",
                                     "instructions": f"Which of these is the official title of the "
                                                     f"ClinicalTrials.gov trial {nct_id}?",
                                     "criteria": criteria}})
    ans = (call["payload"].get("answers") or {}).get("title") or {}
    probs = ans.get("probabilities") or {}
    ok = ans.get("choice") in criteria and set(probs) == set(criteria)
    # probability per OPTION index (0 = scored title), undoing the rotation
    p_by_option = [float(probs[LETTERS[(k + r) % len(options)]]) for k in range(len(options))] if ok else None
    return {"call": call, "parsed": ok, "choice_slot": ans.get("choice"), "p_by_option": p_by_option}


def _title_block(nct_id: str, options: list) -> dict:
    rots = [_title_rotation(nct_id, options, r) for r in range(len(options))]
    ok = [r for r in rots if r["parsed"]]
    mean_p = np.mean([r["p_by_option"] for r in ok], axis=0) if ok else None
    return {"rotations": rots, "n_parsed": len(ok),
            "p_true": float(mean_p[0]) if ok else None,
            "top1_hit": bool(int(np.argmax(mean_p)) == 0) if ok else None,
            "mean_p_by_option": [round(float(x), 4) for x in mean_p] if ok else None}


def _usage(call):
    u = call["payload"].get("usage") or {}
    billed = not call.get("cache_hit")
    return (u.get("input_tokens") or 0, u.get("output_tokens") or 0, (u.get("cost") or 0.0) if billed else 0.0)


def _boot_ci(x, rng, stat=np.mean):
    x = np.asarray(x, dtype=float)
    bs = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(N_BOOT)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _per_task_outcome(rows):
    out = {}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task and r["outcome_noul"] is not None]
        y = np.array([r["y"] for r in rs]); s = np.array([r["outcome_noul"] for r in rs])
        pred = (s >= 0.5).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        p = float(fisher_exact(cm, alternative="greater")[1]) if cm.shape == (2, 2) else None
        out[task] = {
            "n": len(rs), "positive_rate": float(y.mean()),
            "majority_class_rate": float(max(y.mean(), 1 - y.mean())),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auroc": float(roc_auc_score(y, s)) if len(set(y)) == 2 else None,
            "fisher_exact_p": p,
            "clears_t28a_family_alpha": p is not None and p < FAMILY_ALPHA_T28A,
            "clears_jev_only_alpha": p is not None and p < JEV_ONLY_ALPHA,
        }
    return out


def run(data_root="data", n_trials=200, seed=42, workers=WORKERS):
    _load_cache()
    rng = np.random.default_rng(seed)
    sample = _pool_trials(data_root, n_trials, seed)
    decoy_pool = _pool_trials(data_root, 2000, seed + 1)
    decoy_titles = (decoy_pool[~decoy_pool["nct_id"].isin(sample["nct_id"])]["brief_title"]
                    .dropna().drop_duplicates().tolist())

    ids = sample["nct_id"].tolist()
    while True:  # derangement: every trial's control id is another trial's
        perm = rng.permutation(len(ids))
        if not np.any(perm == np.arange(len(ids))):
            break
    jobs = []
    for i, (_, row) in enumerate(sample.iterrows()):
        options = [row["brief_title"]] + list(rng.choice(decoy_titles, size=N_DECOYS, replace=False))
        jobs.append((row, options, ids[perm[i]]))

    def work(job):
        row, options, control_id = job
        o = _outcome(row["task"], render_arm(row, "L7", row["nct_id"]).text)
        real = _title_block(row["nct_id"], options)
        ctrl = _title_block(control_id, options)
        return row, options, control_id, o, real, ctrl

    with ThreadPoolExecutor(max_workers=workers) as ex:
        done = list(ex.map(work, jobs))

    rows, calls = [], []
    for row, options, control_id, o, real, ctrl in done:
        calls += [o["call"]] + [r["call"] for r in real["rotations"]] + [r["call"] for r in ctrl["rotations"]]
        rows.append({
            "nct_id": row["nct_id"], "task": row["task"], "y": int(row["y"]),
            "outcome_noul": o["noul"], "outcome_status": o["call"]["status"],
            "title_options": options, "control_nct_id": control_id,
            "real_p_true": real["p_true"], "real_top1_hit": real["top1_hit"],
            "real_mean_p_by_option": real["mean_p_by_option"], "real_n_parsed": real["n_parsed"],
            "ctrl_p_true": ctrl["p_true"], "ctrl_top1_hit": ctrl["top1_hit"],
            "ctrl_mean_p_by_option": ctrl["mean_p_by_option"], "ctrl_n_parsed": ctrl["n_parsed"],
            "real_choice_slots": [r["choice_slot"] for r in real["rotations"]],
            "ctrl_choice_slots": [r["choice_slot"] for r in ctrl["rotations"]],
            "errors": [c["payload"] for c in [o["call"]] + [r["call"] for r in real["rotations"] + ctrl["rotations"]]
                       if c["status"] != 200] or None,
        })

    brng = np.random.default_rng(seed + 7)
    paired = [r for r in rows if r["real_p_true"] is not None and r["ctrl_p_true"] is not None]
    real_p = np.array([r["real_p_true"] for r in paired]); ctrl_p = np.array([r["ctrl_p_true"] for r in paired])
    diff = real_p - ctrl_p
    w_p = float(wilcoxon(diff, alternative="greater").pvalue) if np.any(diff != 0) else 1.0
    real_hits = sum(r["real_top1_hit"] for r in paired); ctrl_hits = sum(r["ctrl_top1_hit"] for r in paired)
    all_slots = [s for r in rows for s in r["real_choice_slots"] + r["ctrl_choice_slots"] if s]
    title = {
        "n_paired": len(paired), "chance": 1 / (N_DECOYS + 1),
        "real_mean_p_true": float(real_p.mean()), "real_ci_95": _boot_ci(real_p, brng),
        "ctrl_mean_p_true": float(ctrl_p.mean()), "ctrl_ci_95": _boot_ci(ctrl_p, brng),
        "mean_diff": float(diff.mean()), "mean_diff_ci_95": _boot_ci(diff, brng),
        "wilcoxon_p_one_sided": w_p, "alpha": TITLE_ALPHA,
        "recognition_demonstrated": bool(w_p < TITLE_ALPHA and diff.mean() > 0),
        "real_top1_hits": int(real_hits), "ctrl_top1_hits": int(ctrl_hits),
        "real_top1_binom_p_vs_chance": float(binomtest(int(real_hits), len(paired), 0.2, alternative="greater").pvalue),
        "fisher_p_real_vs_ctrl_top1": float(fisher_exact([[real_hits, len(paired) - real_hits],
                                                          [ctrl_hits, len(paired) - ctrl_hits]],
                                                         alternative="greater")[1]),
        "slot_choice_share": {L: round(all_slots.count(L) / len(all_slots), 4) for L in LETTERS},
    }
    per_task = _per_task_outcome(rows)
    if title["recognition_demonstrated"]:
        branch = "RECALL_SHOWN"
    elif any(t["clears_jev_only_alpha"] for t in per_task.values()):
        branch = "NOT_CLEARED_NEEDS_T28B"
    else:
        branch = "NO_SIGNAL"

    usage = [_usage(c) for c in calls]
    meter = {
        "calls": len(calls), "cache_hits": sum(bool(c.get("cache_hit")) for c in calls),
        "http_status_counts": {str(k): sum(c["status"] == k for c in calls)
                               for k in sorted({c["status"] for c in calls}, key=str)},
        "input_tokens": int(sum(u[0] for u in usage)), "output_tokens": int(sum(u[1] for u in usage)),
        "dollars_realized": round(sum(u[2] for u in usage), 6),
        "outcome_parse_rate": float(np.mean([r["outcome_noul"] is not None for r in rows])),
        "title_rotation_parse_rate": float(np.mean([r["real_n_parsed"] + r["ctrl_n_parsed"] for r in rows]) / 10),
    }
    artifact = {
        "test_id": "t28a_jev", "model": MODEL, "endpoint": URL, "n_trials": n_trials, "seed": seed,
        "git_sha": git_sha(),
        "claim_at_stake": "Whether Jev recognises TrialBench trials from their NCT id (title recognition vs a "
                          "shuffled-ID floor), and whether it discriminates outcomes from the full trial text, "
                          "on the same 200 trials as Opus 4.5's T28a.",
        "decision_rule": __doc__.split("Decision rule, fixed before the run:")[1].split("Usage:")[0].strip(),
        "title_recognition": title, "per_task_outcome_discrimination": per_task,
        "branch": branch, "meter": meter, "per_trial": rows,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1, default=str)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    art = run(a.data_root, a.n_trials, a.seed, a.workers)
    print(json.dumps({k: art[k] for k in ["title_recognition", "per_task_outcome_discrimination",
                                           "branch", "meter"]}, indent=1, default=str))
