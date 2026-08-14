"""What is tfidf_logreg actually keying on, and is it the same signal as the
dominant tabular feature (eligibility/healthy_volunteers) or something else?

Two checks, on a real task/phase cell:
  1. Top TF-IDF terms by learned LogisticRegression coefficient (most
     positive = pushes toward class 1; most negative = pushes toward class 0).
  2. Does tfidf_logreg's predicted score differ systematically by the raw
     structured eligibility/healthy_volunteers value? If yes, the text
     channel is (at least partly) redundant with the tabular flag, not an
     independent signal.

Usage:
    python -m experiments.tfidf_signal_probe --task mortality_rate_yn --phase Phase1
"""
from __future__ import annotations

import argparse

import numpy as np

from src.data.loader import load_task_phase
from src.data.features import concat_text
from src import methods as _methods_pkg  # noqa: F401
from src.methods.registry import get as get_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mortality_rate_yn")
    ap.add_argument("--phase", default="Phase1")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    td = load_task_phase(args.data_root, args.task, args.phase, seed=args.seed)

    tfidf = get_method("tfidf_logreg")(task_type=td.task_type, num_classes=td.num_classes, seed=args.seed)
    tfidf.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)

    # -- 1. Top learned terms -------------------------------------------------
    vocab = tfidf.vec_.get_feature_names_out()
    coef = tfidf.clf_.coef_[0]
    order = np.argsort(coef)
    print(f"task={args.task} phase={args.phase}\n")
    print(f"Top {args.top_k} terms pushing toward class 1 (positive):")
    for i in order[::-1][:args.top_k]:
        print(f"  {coef[i]:+.3f}  {vocab[i]!r}")
    print(f"\nTop {args.top_k} terms pushing toward class 0 (negative):")
    for i in order[:args.top_k]:
        print(f"  {coef[i]:+.3f}  {vocab[i]!r}")

    # -- 2. Does the raw eligibility/healthy_volunteers column show up in the
    #       free text, and does tfidf's score track it directly? -----------
    if "eligibility/healthy_volunteers" in td.X_train.columns:
        col = "eligibility/healthy_volunteers"
        texts = concat_text(td.X_test)
        mentions = sum("healthy" in t.lower() for t in texts)
        print(f"\n'{col}' present as a raw column. "
              f"{mentions}/{len(texts)} test rows' concatenated text contains 'healthy'.")

        proba = tfidf.predict_proba(td.X_test)
        proba = proba if proba.ndim == 1 else proba[:, 1]
        raw_vals = td.X_test[col].astype(str)
        print(f"\nMean tfidf_logreg predicted score by raw '{col}' value (test set):")
        for v in raw_vals.unique():
            mask = (raw_vals == v).values
            if mask.sum() == 0:
                continue
            print(f"  {v!r:10s} n={mask.sum():4d}  mean_score={proba[mask].mean():.3f}")
    else:
        print(f"\n(no raw '{'eligibility/healthy_volunteers'}' column in this task's data)")


if __name__ == "__main__":
    main()
