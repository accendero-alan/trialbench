"""T28b for Jev -- is its outcome signal reading or recall?

The arms are exactly those of t28b_jev_preflight.py, rebuilt by the same
calls with the same seed:
- Arm A: TrialBench test trials with results posted before 2023-01-01,
  well before any plausible cutoff. 500 rows, stratified by endpoint and
  label, labelled under P14's rule.
- Arm B: trials registered before Jev's 2026-09-18 OpenRouter release with
  results posted after it, from the 2026-10-15 AACT snapshot. 500 rows of
  356 trials. Jev cannot have seen these outcomes, whatever its cutoff.

Scoring:
- Jev (typesafe/jev-1.13-20260917, /api/alpha/decisions): `state` = the L7
  rendered trial, exactly as Opus's T28b and T28a used it. One `noul`
  question carrying Opus's T28b prediction framing (a risk assessor
  estimating the probability), with no recall framing. A parse failure
  scores 0.5 and is logged per arm, per P13.
- Reference: the frozen TF-IDF + logistic regression on R1_TEXT_COLS,
  fit once on TrialBench train, identical to Opus's T28b. It has no
  memory, so it measures how much of any A->B drop is distribution shift.

Decision rule, fixed before the run (also recorded on newsletter-pipeline
TASK-006 before any call). It is Opus's T28b R1->R2 procedure
(docs/t28b_reanalysis_plan.md), unchanged apart from the arms:
- Metric: AUROC, mortality and SAE pooled, A vs B, independent-arm
  cluster bootstrap on nct_id (1,000 resamples, seed 42).
- R1: does Jev's AUROC drop from A to B (CI lower bound > 0)? No ->
  NO_RECALL_DETECTED: Jev discriminates outcomes it could not have read
  as well as outcomes it could have, so its TrialBench signal is read as
  reading. Yes -> R2.
- R2: diff-in-diff, Jev's A->B drop minus the reference's on the same
  rows. CI lower bound > 0 -> OUTCOME_RECALL_DEMONSTRATED: no Jev
  TrialBench figure is quoted as skill. Otherwise -> INCONCLUSIVE: report
  both drops and the difference, read as neither recall nor cleared.
- The upside is pre-registered too. If Jev holds on Arm B, the finding is
  Jev's AUROC on mortality and SAE for trials whose results postdate it.
  That is the stronger result, and it is reported as such, not as a
  passed safety check.
- Reported, not decisive: PR-AUC and balanced accuracy at 0.5 (context
  only; T28b's R1 showed a 0.5 threshold misleads when base rates
  shift); per-endpoint AUROC per arm; parse-failure rate per arm.
- Known limits: Arm B's phase strings are AACT's (PHASE2, PHASE1/PHASE2),
  the same gap Opus's T28b carried. Jev's answers are not bit-for-bit
  repeatable (+/-0.01 seen on repeats). Arm B's results are at most four
  weeks old, so its trials are recent where Arm A's are pre-2023, which
  the reference arm is there to absorb.

Usage:
    python -m experiments.t28b_jev
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments import t28a_jev as jev
from experiments._common import git_sha
from experiments.t28b_jev_preflight import A_CUTOFF, B_CUTOFF, N_ARM_A, N_ARM_B
from experiments.t28b_opus_recall import (DECISION_METRIC, ENDPOINTS, R1_TEXT_COLS,
                                          align_opus_and_reference, bootstrap_all_metrics,
                                          fit_frozen_reference, sample_arm_a, sample_slice_arm,
                                          score_arm_reference)
from src.data.aact import SNAPSHOT_2026_10_15_DIR, snapshot_info
from src.data.aact_slice import slice_ab_nct_ids
from src.data.serialize import render_arm
from src.eval.pooled_bootstrap import diff_in_diff_bootstrap
from src.methods.llm import TASK_QUESTIONS

OUT_PATH = "results/experiments/t28b_jev.json"
N_RESAMPLES = 1000


def _instructions(endpoint: str) -> str:
    return ("You are a clinical trial risk assessor. Read the trial record and estimate the "
            f"probability that {TASK_QUESTIONS[endpoint]}.")


def _score_row(row) -> dict:
    call = jev._cached_decide(render_arm(row, "L7", str(row["nct_id"])).text,
                              {"risk": {"type": "noul", "instructions": _instructions(row["endpoint"])}})
    val = ((call["payload"].get("answers") or {}).get("risk") or {}).get("noul")
    ok = isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0
    return {"nct_id": row["nct_id"], "endpoint": row["endpoint"], "p14_label": int(row["p14_label"]),
            "proba": float(val) if ok else 0.5, "parse_ok": ok, "status": call["status"],
            "error": None if ok else call["payload"], "_call": call}


def score_arm_jev(arm_df: pd.DataFrame, workers: int) -> tuple:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(_score_row, [row for _, row in arm_df.iterrows()]))
    calls = [r.pop("_call") for r in recs]
    return pd.DataFrame.from_records(recs), calls


def _per_endpoint_auroc(df: pd.DataFrame) -> dict:
    out = {}
    for ep in ENDPOINTS:
        sub = df[df["endpoint"] == ep]
        out[ep] = float(roc_auc_score(sub["p14_label"], sub["proba"])) if sub["p14_label"].nunique() == 2 else None
    return out


def decide(primary_jev: dict, diff_in_diff: dict) -> tuple:
    a = primary_jev[DECISION_METRIC]
    if not a["lo"] > 0:
        return ("NO_RECALL_DETECTED",
                f"R1: Jev's {DECISION_METRIC} does not significantly drop from A to B "
                f"(A {a['mean_x']:.3f}, B {a['mean_y']:.3f}, delta={a['mean_delta']:.3f}, "
                f"CI=[{a['lo']:.3f}, {a['hi']:.3f}]) -- it discriminates outcomes it could not have "
                f"read as well as ones it could have.")
    d = diff_in_diff
    if d["lo"] > 0:
        return ("OUTCOME_RECALL_DEMONSTRATED",
                f"R1: Jev's {DECISION_METRIC} drops (delta={a['mean_delta']:.3f}, CI=[{a['lo']:.3f}, "
                f"{a['hi']:.3f}]). R2: the drop exceeds the reference's on the same rows "
                f"(diff={d['mean_diff']:.3f}, CI=[{d['lo']:.3f}, {d['hi']:.3f}]) -- Jev's TrialBench "
                f"figures are recall-inflated and are not quoted as skill.")
    return ("INCONCLUSIVE",
            f"R1: Jev's {DECISION_METRIC} drops (delta={a['mean_delta']:.3f}, CI=[{a['lo']:.3f}, "
            f"{a['hi']:.3f}]). R2: the drop does not significantly exceed the reference's "
            f"(diff={d['mean_diff']:.3f}, CI=[{d['lo']:.3f}, {d['hi']:.3f}]) -- report both drops; "
            f"neither recall nor cleared.")


def run(data_root="data", seed=42, workers=jev.WORKERS, limit=None):
    jev._load_cache()
    slices = slice_ab_nct_ids(B_CUTOFF, snapshot_dir=SNAPSHOT_2026_10_15_DIR)
    arm_a = sample_arm_a(data_root, A_CUTOFF, N_ARM_A, seed)
    arm_b = sample_slice_arm(slices["a"], N_ARM_B, seed, snapshot_dir=SNAPSHOT_2026_10_15_DIR)
    if limit:  # smoke check only; the artifact says so. Arms are stratum-ordered, so sample, not head()
        arm_a = arm_a.sample(min(limit, len(arm_a)), random_state=seed).reset_index(drop=True)
        arm_b = arm_b.sample(min(limit, len(arm_b)), random_state=seed).reset_index(drop=True)

    jev_a, calls_a = score_arm_jev(arm_a, workers)
    jev_b, calls_b = score_arm_jev(arm_b, workers)
    refs = {ep: fit_frozen_reference(ep, R1_TEXT_COLS, data_root, seed) for ep in ENDPOINTS}
    ref_a, ref_b = score_arm_reference(refs, arm_a), score_arm_reference(refs, arm_b)

    primary_jev = bootstrap_all_metrics(jev_a, jev_b, N_RESAMPLES, seed)
    primary_ref = bootstrap_all_metrics(ref_a, ref_b, N_RESAMPLES, seed)
    nct_a, y_a, jp_a, rp_a = align_opus_and_reference(jev_a, ref_a)
    nct_b, y_b, jp_b, rp_b = align_opus_and_reference(jev_b, ref_b)
    did = diff_in_diff_bootstrap(nct_a, y_a, jp_a, rp_a, nct_b, y_b, jp_b, rp_b,
                                 metric=DECISION_METRIC, n_resamples=N_RESAMPLES, seed=seed)
    branch, reason = decide(primary_jev, did)

    calls = calls_a + calls_b
    usage = [jev._usage(c) for c in calls]
    artifact = {
        "test_id": "t28b_jev", "model": jev.MODEL, "endpoint": jev.URL, "git_sha": git_sha(),
        "smoke_check_limit": limit,
        "claim_at_stake": "Whether Jev's mortality/SAE discrimination survives on trials whose results "
                          "were posted after Jev existed, against a frozen no-memory reference.",
        "decision_rule": __doc__.split("Decision rule, fixed before the run")[1].split("Usage:")[0].strip(),
        "inputs": {"a_cutoff": str(A_CUTOFF.date()), "b_cutoff": str(B_CUTOFF.date()), "seed": seed,
                   "n_resamples": N_RESAMPLES, "decision_metric": DECISION_METRIC,
                   "r1_text_cols": list(R1_TEXT_COLS), "snapshot_b": snapshot_info("2026-10-15"),
                   "snapshot_a_labels": snapshot_info(), "slice_b_trials": len(slices["a"])},
        "n_rows": {"arm_a": len(arm_a), "arm_b": len(arm_b)},
        "parse_fail_rate": {"arm_a": float(1 - jev_a["parse_ok"].mean()),
                            "arm_b": float(1 - jev_b["parse_ok"].mean())},
        "primary_a_vs_b": {"jev": primary_jev, "reference": primary_ref},
        "diff_in_diff": did,
        "per_endpoint_auroc": {"jev_a": _per_endpoint_auroc(jev_a), "jev_b": _per_endpoint_auroc(jev_b),
                               "ref_a": _per_endpoint_auroc(ref_a), "ref_b": _per_endpoint_auroc(ref_b)},
        "branch": branch, "branch_reason": reason,
        "meter": {"calls": len(calls), "cache_hits": sum(bool(c.get("cache_hit")) for c in calls),
                  "http_status_counts": {str(k): sum(c["status"] == k for c in calls)
                                         for k in sorted({c["status"] for c in calls}, key=str)},
                  "input_tokens": int(sum(u[0] for u in usage)), "output_tokens": int(sum(u[1] for u in usage)),
                  "dollars_realized": round(sum(u[2] for u in usage), 6)},
        "per_trial": {"jev_arm_a": jev_a.to_dict(orient="records"), "jev_arm_b": jev_b.to_dict(orient="records"),
                      "reference_arm_a": ref_a.to_dict(orient="records"),
                      "reference_arm_b": ref_b.to_dict(orient="records")},
    }
    out = OUT_PATH if not limit else OUT_PATH.replace(".json", "_smoke.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1, default=str)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=jev.WORKERS)
    ap.add_argument("--limit", type=int, default=None, help="rows per arm, smoke check only")
    a = ap.parse_args()
    art = run(a.data_root, a.seed, a.workers, a.limit)
    print(json.dumps({k: art[k] for k in ["n_rows", "parse_fail_rate", "primary_a_vs_b", "diff_in_diff",
                                           "per_endpoint_auroc", "branch", "branch_reason", "meter"]},
                     indent=1, default=str))
