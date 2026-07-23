"""End-to-end smoke test on synthetic TrialBench-shaped data (no download).

Writes tiny train/test CSVs in the real folder layout, then runs the full
loader -> featurize -> fit -> bootstrap -> leaderboard path for a couple of
always-available methods, for one binary and one multiclass task.

Run:  python -m pytest -q         (or)   python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import methods as _m  # noqa: E402,F401  (register)
from src.data.features import TabularFeaturizer  # noqa: E402
from src.data.loader import load_task_phase  # noqa: E402
from src.eval import leaderboard as lb  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.methods.registry import get as get_method  # noqa: E402
from src.run_benchmark import run_cell  # noqa: E402


def _make_x(n, rng):
    return pd.DataFrame({
        "nctid": [f"NCT{i:08d}" for i in range(n)],
        "enrollment": rng.integers(10, 500, n).astype(float),
        "number_of_arms": rng.integers(1, 4, n).astype(float),
        "eligibility/minimum_age": rng.choice(["18 Years", "12 Years", "6 Months"], n),
        "phase": rng.choice(["Phase 1", "Phase 2", "Phase 3"], n),
        "study_type": rng.choice(["Interventional", "Observational"], n),
        "sponsors/lead_sponsor/agency_class": rng.choice(["Industry", "NIH", "Other"], n),
        "MaskingType-DoubleBlind": rng.integers(0, 2, n),
        "brief_title": rng.choice(["cancer drug trial", "vaccine study", "device safety"], n),
        "eligibility/criteria/textblock": rng.choice(
            ["inclusion adults healthy", "exclusion pregnancy cardiac", "prior therapy allowed"], n),
        "smiless": ["['CCO']"] * n,
        "icdcode": ["[['C50']]"] * n,
        "condition_browse/mesh_term": rng.random(n),
    }).set_index("nctid")


def _write_task(root, folder, phase, n_train=160, n_test=60, multiclass=False, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, folder, phase)
    os.makedirs(d, exist_ok=True)
    for split, n in [("train", n_train), ("test", n_test)]:
        X = _make_x(n, rng)
        # signal: bigger enrollment + industry -> positive
        logit = (X["enrollment"].values / 250.0) + (X["sponsors/lead_sponsor/agency_class"].values == "Industry")
        p = 1 / (1 + np.exp(-(logit - 1.2)))
        if multiclass:
            y = rng.integers(0, 3, n)
            ydf = pd.DataFrame({"trial_failure_reason_prediction": y}, index=X.index)
        else:
            y = (rng.random(n) < p).astype(int)
            rate = p
            ydf = pd.DataFrame({"mortality_rate": rate, "Y/N": y}, index=X.index)
        X.to_csv(os.path.join(d, f"{split}_x.csv"))
        ydf.to_csv(os.path.join(d, f"{split}_y.csv"))


def test_metrics_shapes():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.7, 0.6, 0.3])
    m = M.binary_metrics(y, s)
    assert set(M.BINARY_METRICS).issubset(m)
    assert 0 <= m["auroc"] <= 1
    b = M.bootstrap(y, s, "binary", 2, n_resamples=50, seed=1)
    assert "prauc" in b and "mean" in b["prauc"]


def test_end_to_end(tmp_path=None):
    root = tmp_path or tempfile.mkdtemp()
    data_root = os.path.join(str(root), "data")
    results_dir = os.path.join(str(root), "results")
    _write_task(data_root, "mortality-event-prediction", "Phase1", multiclass=False)
    _write_task(data_root, "trial-failure-reason-identification", "Phase1", multiclass=True)

    # loader + featurizer sanity
    td = load_task_phase(data_root, "mortality_rate_yn", "Phase1")
    fz = TabularFeaturizer(task_type=td.task_type)
    Xtr = fz.fit_transform(td.X_train, td.y_train)
    assert Xtr.shape[0] == len(td.y_train) and Xtr.shape[1] > 0
    # no raw multimodal columns leaked into the tabular view
    assert not any(("mesh_term" in n or n in {"smiless", "icdcode"}) for n in fz.feature_names_)

    cfg = {
        "data_root": data_root, "results_dir": results_dir,
        "tasks": ["mortality_rate_yn", "failure_reason"], "phases": ["Phase1"],
        "methods": ["majority", "logreg_l2", "random_forest", "hist_gbm", "tfidf_logreg"],
        "seeds": [42], "bootstrap": {"n_resamples": 100, "ci": 0.95},
        "max_train_rows": None, "max_test_rows": None,
    }
    os.makedirs(os.path.join(results_dir, "runs"), exist_ok=True)
    n_ok = 0
    for task in cfg["tasks"]:
        for method in cfg["methods"]:
            rec = run_cell(cfg, task, "Phase1", method, 42)
            assert rec["status"] == "ok"
            assert 0 <= rec["bootstrap"]["prauc"]["mean"] <= 1
            n_ok += 1
            import json
            with open(os.path.join(results_dir, "runs",
                      f"{task}__Phase1__{method}__seed42.json"), "w") as f:
                json.dump(rec, f)
    assert n_ok == len(cfg["tasks"]) * len(cfg["methods"])

    df = lb.build(results_dir)
    assert os.path.exists(os.path.join(results_dir, "leaderboard.md"))
    assert (df["status"] == "ok").sum() == n_ok
    print(f"OK: {n_ok} cells ran; leaderboard written to {results_dir}/leaderboard.md")


if __name__ == "__main__":
    test_metrics_shapes()
    test_end_to_end()
    print("smoke test passed")
