"""Diagnose where ft_transformer's PR-AUC is actually coming from, on a real
task/phase subset. Two complementary checks, neither requiring EC2 or
touching results/:

  1. Permutation importance: shuffle one feature column at a time in the test
     set and measure the PR-AUC drop. Names the columns the model relies on.
  2. Cross-method prediction correlation: Spearman correlation between
     ft_transformer's test-set scores and a few other strong methods'
     (extra_trees, random_forest, tfidf_logreg) scores on the same split. High
     correlation means it's recovering the same signal everyone else finds;
     low correlation would mean it's doing something distinctive.

Usage (from repo root, in the venv):
    python -m experiments.ft_transformer_importance --task mortality_rate_yn --phase Phase1
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


def _fit_tabular(name, task_type, num_classes, seed, Xtr, ytr, Xva, yva, Xte):
    method = get_method(name)(task_type=task_type, num_classes=num_classes, seed=seed)
    method.fit(Xtr, ytr, Xva, yva)
    proba = method.predict_proba(Xte)
    return proba if proba.ndim == 1 else proba[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mortality_rate_yn")
    ap.add_argument("--phase", default="Phase1")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=8, help="permutation repeats per feature")
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    td = load_task_phase(args.data_root, args.task, args.phase, seed=args.seed)
    fz = TabularFeaturizer(task_type=td.task_type)
    Xtr = fz.fit_transform(td.X_train, td.y_train)
    Xva = fz.transform(td.X_valid)
    Xte = fz.transform(td.X_test)
    feature_names = fz.feature_names_

    print(f"task={args.task} phase={args.phase} seed={args.seed}  "
          f"n_train={len(Xtr)} n_test={len(Xte)} n_features={Xtr.shape[1]}")

    ft = get_method("ft_transformer")(task_type=td.task_type, num_classes=td.num_classes, seed=args.seed)
    ft.fit(Xtr, td.y_train, Xva, td.y_valid)
    base_proba = ft.predict_proba(Xte)
    base_proba = base_proba if base_proba.ndim == 1 else base_proba[:, 1]
    base_score = average_precision_score(td.y_test, base_proba)
    print(f"\nft_transformer baseline test PR-AUC = {base_score:.4f}")

    # -- 1. Permutation importance -----------------------------------------
    rng = np.random.default_rng(args.seed)
    importances = np.zeros(Xte.shape[1])
    for j in range(Xte.shape[1]):
        drops = []
        for _ in range(args.n_repeats):
            Xte_perm = Xte.copy()
            rng.shuffle(Xte_perm[:, j])
            proba = ft.predict_proba(Xte_perm)
            proba = proba if proba.ndim == 1 else proba[:, 1]
            drops.append(base_score - average_precision_score(td.y_test, proba))
        importances[j] = np.mean(drops)

    order = np.argsort(importances)[::-1]
    print(f"\nTop {args.top_k} features by permutation importance "
          f"(mean PR-AUC drop over {args.n_repeats} shuffles):")
    for rank, j in enumerate(order[:args.top_k], 1):
        print(f"  {rank:2d}. {feature_names[j]:45s} {importances[j]:+.4f}")
    n_negligible = int(np.sum(importances[order] < 0.001))
    print(f"  ... {n_negligible}/{len(feature_names)} features have <0.001 importance")

    # -- 2. Cross-method prediction correlation ------------------------------
    print("\nSpearman correlation with ft_transformer's test-set scores:")
    others_tabular = ["extra_trees", "random_forest"]
    for name in others_tabular:
        proba = _fit_tabular(name, td.task_type, td.num_classes, args.seed, Xtr, td.y_train, Xva, td.y_valid, Xte)
        score = average_precision_score(td.y_test, proba)
        rho, _ = spearmanr(base_proba, proba)
        print(f"  {name:15s} PR-AUC={score:.4f}  spearman(vs ft_transformer)={rho:.3f}")

    tfidf = get_method("tfidf_logreg")(task_type=td.task_type, num_classes=td.num_classes, seed=args.seed)
    tfidf.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)
    tfidf_proba = tfidf.predict_proba(td.X_test)
    tfidf_proba = tfidf_proba if tfidf_proba.ndim == 1 else tfidf_proba[:, 1]
    tfidf_score = average_precision_score(td.y_test, tfidf_proba)
    rho, _ = spearmanr(base_proba, tfidf_proba)
    print(f"  {'tfidf_logreg':15s} PR-AUC={tfidf_score:.4f}  spearman(vs ft_transformer)={rho:.3f}")


if __name__ == "__main__":
    main()
