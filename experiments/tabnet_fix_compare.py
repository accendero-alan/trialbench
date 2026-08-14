"""Compare the original TabNet against a fixed TabNet on a small real-data
subset. Run this ON THE EC2 BOX, where torch/pytorch-tabnet and the TrialBench
data (data/) already exist.

The three fixes vs. src/methods/deep_tabular.py::TabNet:
  1. class weighting     -> pass `weights=1` to TabNetClassifier.fit (inverse-freq)
  2. feature scaling      -> StandardScaler fit on train, applied to valid/test
  3. loss-based early stop -> eval_metric=['logloss'] instead of the accuracy-ish default

Usage (from repo root, in the venv):
    python -m experiments.tabnet_fix_compare --task mortality_rate_yn --phase Phase1
    python -m experiments.tabnet_fix_compare --task serious_adverse_rate_yn --phase Phase1 \
        --max-train-rows 1500 --seeds 42 7 123

It touches nothing in results/ and writes no run JSON -- it just prints a
PR-AUC / AUROC comparison table so you can see whether the fix helps before
deciding to change the method for real.
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.loader import load_task_phase
from src.data.features import TabularFeaturizer


def _scores(model, X):
    p = model.predict_proba(X.astype(np.float32))
    return p[:, 1] if p.shape[1] == 2 else p


def run_original(Xtr, ytr, Xva, yva, Xte, seed):
    from pytorch_tabnet.tab_model import TabNetClassifier
    n = len(Xtr); bs = max(8, min(256, n)); vbs = max(4, min(64, bs))
    clf = TabNetClassifier(n_d=16, n_a=16, n_steps=3, gamma=1.3,
                           seed=seed, verbose=0, device_name="cpu")
    clf.fit(Xtr.astype(np.float32), ytr.astype(np.int64),
            eval_set=[(Xva.astype(np.float32), yva.astype(np.int64))], eval_name=["valid"],
            max_epochs=100, patience=15, batch_size=bs, virtual_batch_size=vbs, drop_last=False)
    return _scores(clf, Xte)


def run_fixed(Xtr, ytr, Xva, yva, Xte, seed):
    from pytorch_tabnet.tab_model import TabNetClassifier
    sc = StandardScaler().fit(Xtr)                       # fix #2: scale (train-only)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    n = len(Xtr); bs = max(8, min(256, n)); vbs = max(4, min(64, bs))
    clf = TabNetClassifier(n_d=16, n_a=16, n_steps=3, gamma=1.3,
                           seed=seed, verbose=0, device_name="cpu")
    clf.fit(Xtr.astype(np.float32), ytr.astype(np.int64),
            eval_set=[(Xva.astype(np.float32), yva.astype(np.int64))], eval_name=["valid"],
            eval_metric=["logloss"],                     # fix #3: loss-based early stop
            weights=1,                                   # fix #1: inverse-freq class weights
            max_epochs=100, patience=15, batch_size=bs, virtual_batch_size=vbs, drop_last=False)
    return _scores(clf, Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mortality_rate_yn")
    ap.add_argument("--phase", default="Phase1")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-test-rows", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    args = ap.parse_args()

    print(f"task={args.task} phase={args.phase}  seeds={args.seeds}")
    rows = {"original": [], "fixed": []}
    prior = None
    for seed in args.seeds:
        td = load_task_phase(args.data_root, args.task, args.phase, seed=seed,
                             max_train_rows=args.max_train_rows, max_test_rows=args.max_test_rows)
        fz = TabularFeaturizer(task_type=td.task_type)
        Xtr = fz.fit_transform(td.X_train, td.y_train)
        Xva = fz.transform(td.X_valid); Xte = fz.transform(td.X_test)
        prior = float(np.mean(td.y_test))
        for name, fn in (("original", run_original), ("fixed", run_fixed)):
            s = fn(Xtr, td.y_train, Xva, td.y_valid, Xte, seed)
            rows[name].append((average_precision_score(td.y_test, s),
                               roc_auc_score(td.y_test, s), float((s >= 0.5).mean())))

    print(f"\nmajority/prior PR-AUC = {prior:.3f}  (test positive prevalence)")
    print(f"{'variant':10s} {'PR-AUC':>16s} {'AUROC':>10s} {'pred-pos@0.5':>14s}")
    for name in ("original", "fixed"):
        pa = np.array([r[0] for r in rows[name]]); au = np.array([r[1] for r in rows[name]])
        pf = np.array([r[2] for r in rows[name]])
        print(f"{name:10s} {pa.mean():8.3f}±{pa.std():.3f} {au.mean():10.3f} {pf.mean():14.2f}")
    dpa = np.mean([r[0] for r in rows['fixed']]) - np.mean([r[0] for r in rows['original']])
    print(f"\ndelta PR-AUC (fixed - original) = {dpa:+.3f}")


if __name__ == "__main__":
    main()
