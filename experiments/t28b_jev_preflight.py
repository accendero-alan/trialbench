"""T28b preflight for Jev -- zero model calls.

T28b separates reading from recall by when a trial's results became
public. Opus's run partitioned at Opus's published training cutoff. Jev
publishes none, so this partitions at a bound that holds whatever its
cutoff is: Jev cannot have trained on results posted after it existed.

- B_CUTOFF = 2026-09-18, the OpenRouter listing of typesafe/jev-1.13
  (created 2026-09-18 00:01 UTC; the model id is dated 20260917; press
  reports a 2026-09-15 launch). Arm B = registered before it, results
  posted after it -- outcome unseen by construction. Needs the 2026-10-15
  AACT snapshot, since the pinned 2026-08-26 one predates the release.
- A_CUTOFF = 2023-01-01. Arm A = TrialBench test trials with results
  posted before it, so they precede any plausible cutoff of a model
  released in late 2026, including one built on an older open-weight base.
- Arm C (registered after release) is reported as a count only.

Checks, all from the existing T28b machinery: slice sizes, n and label
balance per arm, covariates, power, TrialBench-vs-AACT text identity on
the new snapshot, plus a zero-call preview of the frozen TF-IDF reference
on A and B -- how much A->B shift the reference sees before any model is
asked anything.

Usage:
    python -m experiments.t28b_jev_preflight
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments._common import git_sha
from experiments.t28b_opus_recall import (ENDPOINTS, R1_TEXT_COLS, fit_frozen_reference,
                                          preflight_covariates, preflight_n_and_label_balance,
                                          preflight_power, preflight_text_identity, sample_arm_a,
                                          sample_slice_arm, score_arm_reference)
from src.data.aact import SNAPSHOT_2026_10_15_DIR, results_posted_date, snapshot_info
from src.data.aact_slice import slice_ab_nct_ids
from src.eval.pooled_bootstrap import two_sample_cluster_bootstrap

OUT_PATH = "results/experiments/t28b_jev_preflight.json"
B_CUTOFF = pd.Timestamp("2026-09-18")
A_CUTOFF = pd.Timestamp("2023-01-01")
N_ARM_A = 500
N_ARM_B = 500


def _reference_preview(arm_a, arm_b, data_root, seed):
    refs = {ep: fit_frozen_reference(ep, R1_TEXT_COLS, data_root, seed) for ep in ENDPOINTS}
    ra, rb = score_arm_reference(refs, arm_a), score_arm_reference(refs, arm_b)
    out = {}
    for ep in ENDPOINTS:
        a, b = ra[ra["endpoint"] == ep], rb[rb["endpoint"] == ep]
        auc = lambda d: float(roc_auc_score(d["p14_label"], d["proba"])) if d["p14_label"].nunique() == 2 else None
        diff = two_sample_cluster_bootstrap(a["nct_id"].values, a["p14_label"].values, a["proba"].values,
                                            b["nct_id"].values, b["p14_label"].values, b["proba"].values,
                                            metric="auroc", n_resamples=1000, seed=seed) \
            if len(a) and len(b) and auc(a) is not None and auc(b) is not None else None
        out[ep] = {"auroc_a": auc(a), "auroc_b": auc(b), "n_a": len(a), "n_b": len(b), "a_minus_b": diff}
    return out


def run(data_root="data", seed=42):
    snap = SNAPSHOT_2026_10_15_DIR
    res = results_posted_date(snapshot_dir=snap)
    after = res[res > B_CUTOFF]
    slices = slice_ab_nct_ids(B_CUTOFF, snapshot_dir=snap)

    arm_a = sample_arm_a(data_root, A_CUTOFF, N_ARM_A, seed)
    arm_b = sample_slice_arm(slices["a"], N_ARM_B, seed, snapshot_dir=snap)
    empty = arm_b.iloc[0:0]

    artifact = {
        "test_id": "t28b_jev_preflight", "git_sha": git_sha(), "seed": seed,
        "snapshot_b": snapshot_info("2026-10-15"), "snapshot_a_labels": snapshot_info(),
        "b_cutoff": str(B_CUTOFF.date()), "a_cutoff": str(A_CUTOFF.date()),
        "b_cutoff_source": "OpenRouter listing of typesafe/jev-1.13, created 2026-09-18 00:01 UTC "
                           "(model id dated 20260917; press launch 2026-09-15)",
        "snapshot_results_dates": {
            "max_results_posted": str(res.max().date()),
            "n_results_posted_after_b_cutoff": int(len(after)),
            "by_week_after_b_cutoff": {str(k.date()): int(v) for k, v in
                                       after.dt.to_period("W").dt.start_time.value_counts().sort_index().items()},
        },
        "slice_sizes": {"a_registered_pre_results_post": len(slices["a"]),
                        "b_fully_post_release": len(slices["b"])},
        "n_and_label_balance": preflight_n_and_label_balance(arm_a, arm_b, empty),
        "covariates": preflight_covariates(arm_a, arm_b, empty),
        "power": preflight_power(arm_a, arm_b, seed),
        "text_identity_new_snapshot": preflight_text_identity(data_root=data_root, seed=seed, snapshot_dir=snap),
        "reference_preview": _reference_preview(arm_a, arm_b, data_root, seed),
        "arm_b_overlap_with_trialbench_test": int(arm_b["nct_id"].isin(arm_a["nct_id"]).sum()),
        "model_calls": 0,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1, default=str)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(run(a.data_root, a.seed), indent=1, default=str))
