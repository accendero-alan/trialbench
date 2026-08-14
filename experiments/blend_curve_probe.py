"""Sweep the tabular/text blend weight and record the full PR-AUC curve.

`signal-analysis.md` §4.3 reported only the *optimum* of this blend (approval:
0.70 tabular / 0.30 text; mortality: 0.25 tabular / 0.75 text). This re-runs the
same fusion but keeps the whole curve from w=0 (tabular only) to w=1 (text only),
so the lift, its flatness, and the flip in optimal mix between tasks are all
visible in one frame.

Method (unchanged from signal-analysis.md §3):
  - tabular consensus = mean of the per-method rank-percentiles of the 10 tabular
    Tier A methods (rank fusion; scale-free, so heterogeneous score ranges blend
    sanely);
  - text = `tfidf_logreg`, whose ~50k TF-IDF features are disjoint from the
    38-43 engineered tabular columns;
  - blend(w) = (1-w) * tabular_rank + w * text_rank, ranked within each split.

Protocol: every method is fit on **train** only. The reported optimum w* is
selected on **validation**; the plotted curve is **test**. The test argmax is
recorded separately so the gap between "what validation picked" and "what test
would have preferred" stays visible rather than being quietly conflated.

Usage:
    python -m experiments.blend_curve_probe --out blend_curve.json
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
from sklearn.metrics import average_precision_score

from src.data.features import TabularFeaturizer
from src.data.loader import load_task_phase
from src import methods as _methods_pkg  # noqa: F401  (populates the registry)
from src.methods.registry import get as get_method

warnings.filterwarnings("ignore")

TABULAR_METHODS = ["logreg_l1", "logreg_l2", "svm_linear", "knn", "random_forest",
                   "extra_trees", "hist_gbm", "xgboost", "lightgbm", "catboost"]
TEXT_METHOD = "tfidf_logreg"

# (label, task, phase) — the two cells signal-analysis.md §4.3 reports
CELLS = [("Trial approval", "outcome", "Phase1"),
         ("Mortality", "mortality_rate_yn", "Phase1")]


def _scores(proba):
    return proba if proba.ndim == 1 else proba[:, 1]


def _rank_pct(s):
    """Map scores to [0,1] percentiles within the split (ties get equal rank)."""
    from scipy.stats import rankdata
    r = rankdata(np.asarray(s, dtype=float), method="average")
    return (r - 1.0) / max(len(r) - 1, 1)


def run_cell(data_root, task, phase, seed, n_grid):
    td = load_task_phase(data_root, task, phase, seed=seed)
    fz = TabularFeaturizer(task_type=td.task_type)
    Xtr = fz.fit_transform(td.X_train, td.y_train)
    Xva, Xte = fz.transform(td.X_valid), fz.transform(td.X_test)

    va_ranks, te_ranks, per_method = [], [], {}
    for name in TABULAR_METHODS:
        m = get_method(name)(task_type=td.task_type, num_classes=td.num_classes, seed=seed)
        m.fit(Xtr, td.y_train, Xva, td.y_valid)
        sv, st = _scores(m.predict_proba(Xva)), _scores(m.predict_proba(Xte))
        va_ranks.append(_rank_pct(sv))
        te_ranks.append(_rank_pct(st))
        per_method[name] = float(average_precision_score(td.y_test, st))
        print(f"  {name:16s} test PR-AUC {per_method[name]:.4f}")

    tab_va, tab_te = np.mean(va_ranks, axis=0), np.mean(te_ranks, axis=0)

    t = get_method(TEXT_METHOD)(task_type=td.task_type, num_classes=td.num_classes, seed=seed)
    t.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)
    txt_va = _rank_pct(_scores(t.predict_proba(td.X_valid)))
    txt_te = _rank_pct(_scores(t.predict_proba(td.X_test)))
    per_method[TEXT_METHOD] = float(average_precision_score(td.y_test, txt_te))
    print(f"  {TEXT_METHOD:16s} test PR-AUC {per_method[TEXT_METHOD]:.4f}")

    ws = np.linspace(0.0, 1.0, n_grid)
    curve_va, curve_te = [], []
    for w in ws:
        curve_va.append(float(average_precision_score(td.y_valid, (1 - w) * tab_va + w * txt_va)))
        curve_te.append(float(average_precision_score(td.y_test, (1 - w) * tab_te + w * txt_te)))

    i_val = int(np.argmax(curve_va))       # protocol: selected on validation
    i_te = int(np.argmax(curve_te))        # recorded for honesty, not for selection
    return {
        "task": task, "phase": phase, "seed": seed,
        "test_pos_rate": float(td.y_test.mean()),
        "w": ws.tolist(), "curve_test": curve_te, "curve_valid": curve_va,
        "w_star_valid": float(ws[i_val]), "prauc_test_at_w_star": curve_te[i_val],
        "w_argmax_test": float(ws[i_te]), "prauc_test_at_argmax": curve_te[i_te],
        "prauc_tabular_only": curve_te[0], "prauc_text_only": curve_te[-1],
        "best_single_tabular_method": max(
            ((k, v) for k, v in per_method.items() if k != TEXT_METHOD), key=lambda kv: kv[1]),
        "per_method_test_prauc": per_method,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-grid", type=int, default=101)
    ap.add_argument("--out", default="blend_curve.json")
    args = ap.parse_args()

    out = []
    for label, task, phase in CELLS:
        print(f"\n=== {label}: {task}/{phase} ===")
        rec = run_cell(args.data_root, task, phase, args.seed, args.n_grid)
        rec["label"] = label
        print(f"  tabular-only {rec['prauc_tabular_only']:.4f} | "
              f"text-only {rec['prauc_text_only']:.4f} | "
              f"blend@w*={rec['w_star_valid']:.2f} {rec['prauc_test_at_w_star']:.4f} "
              f"(test argmax w={rec['w_argmax_test']:.2f} -> {rec['prauc_test_at_argmax']:.4f})")
        out.append(rec)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
