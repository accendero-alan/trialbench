"""T9 -- The near-duplicate audit (highest-risk test in the plan).

newsletter-part2-test-plan.md, T9. TEXT_PLAN.md's own gate: >2-3% train/test
duplication means stop and reinterpret the benchmark. A first-pass cosine
approximation put mortality/Phase1 at 2.86% and failure_reason/Phase3 at
4.29%, at or over the line, on cells TF-IDF wins.

Per cell (single seed=42, using the now-repaired TEXT_COLS default from T7):
TF-IDF-vectorize train (fit) and test (transform), compute each test row's
max cosine similarity to any train row via a blocked sparse dot product
(TfidfVectorizer L2-normalizes rows by default, so the dot product of two
rows *is* their cosine similarity -- no separate normalization needed).
Report the fraction of test rows above 0.85/0.90/0.95/0.99, plus exact
train/test text-string duplicates. Re-score `tfidf_logreg` and the best GBM
(by T1-baseline point PR-AUC among xgboost/lightgbm/catboost) on the
deduplicated test subset at each threshold.

Decision rule is pre-registered and evaluated at threshold=0.95 (the
plan's own reference numbers -- 2.86%/4.29% -- are the closest match to a
0.95 cosine cut; this script reports all four thresholds so that choice is
auditable, not hidden). Per the plan: do not move the decision threshold
after seeing results.

Usage: `python -m experiments.t9_near_duplicates`
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import TabularFeaturizer, concat_text
from src.data.loader import TASKS, load_task_phase
from src.eval import metrics as M
from src.methods.registry import get as get_method

SEED = 42
DATA_ROOT = "data"
OUT_PATH = "results/experiments/t9_near_duplicates.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
T1_BASELINE_RUNS = "results/experiments/_snapshots/t1_baseline/runs"
THRESHOLDS = [0.85, 0.90, 0.95, 0.99]
DECISION_THRESHOLD = 0.95
MIN_RETAINED = 100
GBM_CANDIDATES = ["xgboost", "lightgbm", "catboost"]

ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]


def _max_cosine_sim(test_mat, train_mat, block_size=500):
    """Max cosine similarity of each test row to any train row. Both
    matrices are L2-normalized TF-IDF rows, so a sparse dot product is the
    cosine similarity directly. Blocked over test rows to bound memory
    (dense per-block result is block_size x n_train)."""
    n_test = test_mat.shape[0]
    out = np.zeros(n_test, dtype=float)
    for start in range(0, n_test, block_size):
        end = min(start + block_size, n_test)
        block = test_mat[start:end] @ train_mat.T
        out[start:end] = np.asarray(block.max(axis=1).todense()).ravel()
    return out


def _best_gbm(task, phase):
    best_method, best_score = None, -np.inf
    for method in GBM_CANDIDATES:
        path = f"{T1_BASELINE_RUNS}/{task}__{phase}__{method}__seed{SEED}.json"
        try:
            with open(path) as f:
                rec = json.load(f)
        except FileNotFoundError:
            continue
        if rec.get("status") == "ok" and rec["point"]["prauc"] > best_score:
            best_method, best_score = method, rec["point"]["prauc"]
    return best_method


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}

    per_cell = []
    with Timer() as t:
        for task, phase in ALL_CELLS:
            td = load_task_phase(DATA_ROOT, task, phase, seed=SEED)
            train_texts = concat_text(td.X_train)
            test_texts = concat_text(td.X_test)

            vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            train_mat = vec.fit_transform(train_texts)
            test_mat = vec.transform(test_texts)

            max_sim = _max_cosine_sim(test_mat, train_mat)
            frac_above = {str(th): float(np.mean(max_sim > th)) for th in THRESHOLDS}
            train_text_set = set(train_texts)
            n_exact = sum(1 for txt in test_texts if txt in train_text_set and txt != "")
            frac_exact = n_exact / len(test_texts) if test_texts else float("nan")

            # tfidf_logreg full-test scoring (repaired TEXT_COLS default, landed in T7)
            TfidfCls = get_method("tfidf_logreg")
            tfidf_model = TfidfCls(task_type=td.task_type, num_classes=td.num_classes, seed=SEED)
            tfidf_model.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)
            tfidf_proba = tfidf_model.predict_proba(td.X_test)

            gbm_name = _best_gbm(task, phase)
            gbm_proba = None
            if gbm_name is not None:
                fz = TabularFeaturizer(task_type=td.task_type)
                Xtr = fz.fit_transform(td.X_train, td.y_train)
                Xte = fz.transform(td.X_test)
                GbmCls = get_method(gbm_name)
                gbm_model = GbmCls(task_type=td.task_type, num_classes=td.num_classes, seed=SEED)
                gbm_model.fit(Xtr, td.y_train, fz.transform(td.X_valid), td.y_valid)
                gbm_proba = gbm_model.predict_proba(Xte)

            tfidf_full = M.compute(td.y_test, tfidf_proba, td.task_type, td.num_classes)["prauc"]
            gbm_full = (M.compute(td.y_test, gbm_proba, td.task_type, td.num_classes)["prauc"]
                        if gbm_proba is not None else float("nan"))
            tfidf_wins = bool(tfidf_full > gbm_full) if not np.isnan(gbm_full) else None

            by_threshold = {}
            for th in THRESHOLDS:
                retained = np.flatnonzero(max_sim <= th)
                n_retained = int(len(retained))
                entry = {"n_retained": n_retained, "n_test": len(test_texts),
                         "resolvable": n_retained >= MIN_RETAINED}
                if entry["resolvable"]:
                    yte_r = td.y_test[retained]
                    tfidf_r = (tfidf_proba[retained] if td.task_type == "binary"
                               else tfidf_proba[retained, :])
                    tfidf_dedup = M.compute(yte_r, tfidf_r, td.task_type, td.num_classes)["prauc"]
                    tfidf_boot = M.bootstrap(yte_r, tfidf_r, td.task_type, td.num_classes,
                                              n_resamples=1000, seed=SEED)["prauc"]
                    entry["tfidf_dedup_prauc"] = tfidf_dedup
                    entry["tfidf_dedup_boot"] = tfidf_boot
                    entry["tfidf_drop"] = tfidf_full - tfidf_dedup
                    if gbm_proba is not None:
                        gbm_r = gbm_proba[retained] if td.task_type == "binary" else gbm_proba[retained, :]
                        gbm_dedup = M.compute(yte_r, gbm_r, td.task_type, td.num_classes)["prauc"]
                        entry["gbm_dedup_prauc"] = gbm_dedup
                        entry["gbm_drop"] = gbm_full - gbm_dedup
                per_cell_thresh = entry
                by_threshold[str(th)] = per_cell_thresh

            delta = delta_cells.get((task, phase), float("nan"))
            decision_entry = by_threshold[str(DECISION_THRESHOLD)]
            if not decision_entry["resolvable"]:
                cell_verdict = "UNRESOLVABLE"
            elif tfidf_wins and decision_entry.get("tfidf_drop", 0) > delta:
                cell_verdict = "CUT"
            elif tfidf_wins:
                cell_verdict = "CONFIRMED"
            else:
                cell_verdict = "N/A (tfidf does not win this cell)"

            per_cell.append({
                "task": task, "phase": phase,
                "n_train": len(train_texts), "n_test": len(test_texts),
                "frac_above_threshold": frac_above, "frac_exact_duplicate": frac_exact,
                "tfidf_full_prauc": tfidf_full, "best_gbm": gbm_name, "gbm_full_prauc": gbm_full,
                "tfidf_wins": tfidf_wins, "delta_cell": delta,
                "by_threshold": by_threshold,
                "cell_verdict_at_0.95": cell_verdict,
            })
            print(f"  {task}/{phase}: frac>0.95={frac_above['0.95']:.4f} exact={frac_exact:.4f} "
                  f"tfidf={tfidf_full:.4f} gbm({gbm_name})={gbm_full:.4f} wins={tfidf_wins} "
                  f"-> {cell_verdict}", flush=True)

    n_over_gate = sum(1 for c in per_cell if c["frac_above_threshold"]["0.95"] > 0.03)
    n_cut = sum(1 for c in per_cell if c["cell_verdict_at_0.95"] == "CUT")
    n_confirmed = sum(1 for c in per_cell if c["cell_verdict_at_0.95"] == "CONFIRMED")
    n_unresolvable = sum(1 for c in per_cell if c["cell_verdict_at_0.95"] == "UNRESOLVABLE")

    if n_cut > 0:
        overall_verdict = (f"CUT: {n_cut} cell(s) TF-IDF wins show a PR-AUC drop on the "
                            f"deduplicated (0.95 cosine) subset exceeding delta_cell -- the "
                            f"text headline must be rewritten as a duplication finding on "
                            f"those cells, and Part 1's published text claim needs the same "
                            f"correction.")
    elif n_confirmed > 0:
        overall_verdict = (f"CONFIRMED: TF-IDF's win holds on the deduplicated subset in all "
                            f"{n_confirmed} cell(s) it wins with a resolvable dedup at 0.95 "
                            f"cosine; the audit gets one sentence.")
    else:
        overall_verdict = "N/A: tfidf_logreg does not win any cell at seed 42 against its best GBM."

    artifact = {
        "test_id": "T9",
        "claim_at_stake": "the entire text result -- is TF-IDF winning by reading trials or recognizing them",
        "inputs": {"seed": SEED, "data_root": DATA_ROOT, "thresholds": THRESHOLDS,
                   "decision_threshold": DECISION_THRESHOLD, "min_retained": MIN_RETAINED,
                   "note": "Decision threshold (0.95) was fixed before this script was run, "
                           "per the plan's own instruction not to move it after seeing results."},
        "n_cells": len(per_cell),
        "n_cells_above_3pct_at_0.95": n_over_gate,
        "per_cell": per_cell,
        "n_cells_cut": n_cut, "n_cells_confirmed": n_confirmed, "n_cells_unresolvable": n_unresolvable,
        "decision_rule": "At threshold=0.95: if dedup TF-IDF PR-AUC drops by more than "
                          "delta_cell on a cell TF-IDF wins, that cell's text headline is CUT "
                          "and rewritten as a duplication finding (same correction applied to "
                          "Part 1). If the drop is within delta_cell, CONFIRMED. If dedup "
                          "leaves <100 test rows, UNRESOLVABLE.",
        "verdict": overall_verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}")
    print(overall_verdict)
    return artifact


if __name__ == "__main__":
    main()
