"""Compare feature reliance and prediction agreement across several methods
on one real task/phase cell. Generalizes ft_transformer_importance.py to an
arbitrary method list, so cells can be compared head-to-head. Touches nothing
in results/, requires no EC2.

For methods with feature_view == "tabular" (majority of Tier A/B), computes
permutation importance over the shared feature matrix -- so different
methods' reliance on the SAME columns is directly comparable. Raw-view
methods (tfidf_logreg, clinical_embeddings) are fit and scored for the
correlation matrix but skipped for permutation importance (their input isn't
the tabular matrix, so column-shuffling isn't a like-for-like comparison).

Usage:
    python -m experiments.multi_method_signal_compare --task mortality_rate_yn --phase Phase2 \
        --methods ft_transformer tabnet extra_trees random_forest tfidf_logreg
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from src.data.features import TabularFeaturizer
from src.data.loader import load_task_phase
from src import methods as _methods_pkg  # noqa: F401  (populates the registry)
from src.methods.registry import get as get_method


def _as_scores(proba):
    return proba if proba.ndim == 1 else proba[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mortality_rate_yn")
    ap.add_argument("--phase", default="Phase1")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--n-repeats", type=int, default=5, help="permutation repeats per feature")
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    td = load_task_phase(args.data_root, args.task, args.phase, seed=args.seed)
    fz = TabularFeaturizer(task_type=td.task_type)
    Xtr = fz.fit_transform(td.X_train, td.y_train)
    Xva = fz.transform(td.X_valid)
    Xte = fz.transform(td.X_test)
    feature_names = fz.feature_names_

    print(f"task={args.task} phase={args.phase} seed={args.seed}  "
          f"n_train={len(Xtr)} n_test={len(Xte)} n_features={Xtr.shape[1]}\n")

    scores = {}   # name -> test-set proba (1-D, P(y=1))
    fitted = {}   # name -> (method, is_tabular)

    for name in args.methods:
        MethodCls = get_method(name)
        method = MethodCls(task_type=td.task_type, num_classes=td.num_classes, seed=args.seed)
        if method.feature_view == "tabular":
            method.fit(Xtr, td.y_train, Xva, td.y_valid)
            proba = _as_scores(method.predict_proba(Xte))
            fitted[name] = (method, True)
        else:
            method.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)
            proba = _as_scores(method.predict_proba(td.X_test))
            fitted[name] = (method, False)
        scores[name] = proba
        pr = average_precision_score(td.y_test, proba)
        print(f"{name:16s} test PR-AUC = {pr:.4f}")

    # -- Permutation importance for tabular-view methods ----------------------
    rng = np.random.default_rng(args.seed)
    for name, (method, is_tabular) in fitted.items():
        if not is_tabular:
            continue
        base_score = average_precision_score(td.y_test, scores[name])
        importances = np.zeros(Xte.shape[1])
        for j in range(Xte.shape[1]):
            drops = []
            for _ in range(args.n_repeats):
                Xte_perm = Xte.copy()
                rng.shuffle(Xte_perm[:, j])
                proba = _as_scores(method.predict_proba(Xte_perm))
                drops.append(base_score - average_precision_score(td.y_test, proba))
            importances[j] = np.mean(drops)
        order = np.argsort(importances)[::-1]
        print(f"\n{name} -- top {args.top_k} features by permutation importance "
              f"(mean PR-AUC drop / {args.n_repeats} shuffles):")
        for rank, j in enumerate(order[:args.top_k], 1):
            print(f"  {rank:2d}. {feature_names[j]:45s} {importances[j]:+.4f}")

    # -- Full pairwise correlation matrix -------------------------------------
    names = list(scores.keys())
    print("\nSpearman correlation matrix (test-set predicted scores):")
    header = "".join(f"{n[:12]:>13s}" for n in names)
    print(f"{'':16s}{header}")
    for a in names:
        row = "".join(f"{spearmanr(scores[a], scores[b])[0]:13.3f}" for b in names)
        print(f"{a:16s}{row}")


if __name__ == "__main__":
    main()
