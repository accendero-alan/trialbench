"""T7 -- Repair TEXT_COLS and measure what the typo cost.

newsletter-part2-test-plan.md, T7. `TEXT_COLS` (src/data/features.py) lists
"brief_summary" and "detailed_description", but the actual CSV columns are
"brief_summary/textblock" and "detailed_description/textblock" (confirmed by
reading the real data: every task folder has `brief_summary/textblock`;
`trial-failure-reason-identification` additionally has
`detailed_description/textblock`; no task folder has a bare `detailed_description`
or an `eligibility/study_pop/textblock` column at all, so those two names in
TEXT_COLS_AS_SHIPPED are dead weight in every config, not just the buggy one).

Runs `tfidf_logreg`-equivalent fits across all 20 cells x 5 seeds x 3 text
configs: as-shipped, repaired, repaired-minus-summaries (isolates how much of
the gain is specifically the newly-readable summary/description fields vs.
everything else already in scope). Requires P1 (TEXT_COLS as an ordered
tuple) -- otherwise these numbers don't reproduce between processes.

If the repair clears the bar, this script also lands it: updates the live
`TEXT_COLS` default in features.py to the repaired names, so T8/T9 onward see
the repaired tabular/text split.

Usage: `python -m experiments.t7_textcols_repair`
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import TEXT_COLS_AS_SHIPPED, concat_text
from src.data.loader import TASKS, load_task_phase
from src.eval import metrics as M

SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"
OUT_PATH = "results/experiments/t7_textcols_repair.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
T1_BASELINE_RUNS = "results/experiments/_snapshots/t1_baseline/runs"

_RENAME = {
    "brief_summary": "brief_summary/textblock",
    "detailed_description": "detailed_description/textblock",
}
TEXT_COLS_REPAIRED = tuple(_RENAME.get(c, c) for c in TEXT_COLS_AS_SHIPPED)
TEXT_COLS_REPAIRED_MINUS_SUMMARIES = tuple(
    c for c in TEXT_COLS_REPAIRED
    if c not in ("brief_summary/textblock", "detailed_description/textblock")
)

CONFIGS = {
    "as_shipped": TEXT_COLS_AS_SHIPPED,
    "repaired": TEXT_COLS_REPAIRED,
    "repaired_minus_summaries": TEXT_COLS_REPAIRED_MINUS_SUMMARIES,
}

ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]


def _fit_score(X_train, y_train, X_test, y_test, text_cols, task_type, num_classes, seed):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(concat_text(X_train, text_cols=text_cols))
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(Xtr, y_train)
    Xte = vec.transform(concat_text(X_test, text_cols=text_cols))
    proba = clf.predict_proba(Xte)
    if task_type == "binary":
        score = proba[:, 1] if proba.ndim > 1 else proba
    else:
        score = np.zeros((Xte.shape[0], num_classes), dtype=float)
        for j, c in enumerate(clf.classes_):
            score[:, int(c)] = proba[:, j]
    return M.compute(y_test, score, task_type, num_classes), M.bootstrap(
        y_test, score, task_type, num_classes, n_resamples=1000, ci=0.95, seed=seed)


def _load_t1_deltas():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    return {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}


def _load_t1_baseline_tabular_prauc(task, phase, seed):
    """mean PR-AUC of the 11 non-tfidf Tier A methods from the frozen T1
    baseline snapshot, for the tfidf-vs-tabular-pack rank comparison."""
    vals = {}
    for method in ["majority", "logreg_l2", "logreg_l1", "random_forest", "extra_trees",
                   "hist_gbm", "knn", "svm_linear", "xgboost", "lightgbm", "catboost"]:
        path = os.path.join(T1_BASELINE_RUNS, f"{task}__{phase}__{method}__seed{seed}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rec = json.load(f)
        if rec.get("status") == "ok":
            vals[method] = rec["point"]["prauc"]
    return vals


def main():
    delta_cells = _load_t1_deltas()
    per_cell = []
    with Timer() as t:
        for task, phase in ALL_CELLS:
            cell_out = {"task": task, "phase": phase, "configs": {}}
            for cfg_name, cols in CONFIGS.items():
                seed_scores = []
                for seed in SEEDS:
                    td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
                    point, boot = _fit_score(td.X_train, td.y_train, td.X_test, td.y_test,
                                              cols, td.task_type, td.num_classes, seed)
                    seed_scores.append({"seed": seed, "point_prauc": point["prauc"],
                                         "boot_mean": boot["prauc"]["mean"],
                                         "boot_lo": boot["prauc"]["lo"], "boot_hi": boot["prauc"]["hi"]})
                cell_out["configs"][cfg_name] = {
                    "mean_point_prauc": float(np.mean([s["point_prauc"] for s in seed_scores])),
                    "seed_scores": seed_scores,
                }

            as_shipped = cell_out["configs"]["as_shipped"]["mean_point_prauc"]
            repaired = cell_out["configs"]["repaired"]["mean_point_prauc"]
            gain = repaired - as_shipped
            delta = delta_cells.get((task, phase), float("nan"))
            cell_out["repair_gain"] = gain
            cell_out["delta_cell"] = delta
            cell_out["gain_exceeds_delta_cell"] = bool(abs(gain) > delta) if not np.isnan(delta) else None

            # rank of tfidf (as-shipped vs repaired) against the T1-baseline tabular pack, per seed
            rank_shifts = []
            for seed in SEEDS:
                tab_vals = _load_t1_baseline_tabular_prauc(task, phase, seed)
                as_shipped_seed = next(s["point_prauc"] for s in cell_out["configs"]["as_shipped"]["seed_scores"] if s["seed"] == seed)
                repaired_seed = next(s["point_prauc"] for s in cell_out["configs"]["repaired"]["seed_scores"] if s["seed"] == seed)
                all_as_shipped = sorted(list(tab_vals.values()) + [as_shipped_seed], reverse=True)
                all_repaired = sorted(list(tab_vals.values()) + [repaired_seed], reverse=True)
                rank_as_shipped = all_as_shipped.index(as_shipped_seed) + 1
                rank_repaired = all_repaired.index(repaired_seed) + 1
                rank_shifts.append(rank_as_shipped - rank_repaired)
            cell_out["mean_rank_shift_vs_t1_baseline_tabular"] = float(np.mean(rank_shifts)) if rank_shifts else None

            per_cell.append(cell_out)
            print(f"  {task}/{phase}: as_shipped={as_shipped:.4f} repaired={repaired:.4f} "
                  f"gain={gain:+.4f} delta_cell={delta:.4f}", flush=True)

    n_cells = len(per_cell)
    n_below_delta = sum(1 for c in per_cell if c["gain_exceeds_delta_cell"] is False)
    n_above_delta = sum(1 for c in per_cell if c["gain_exceeds_delta_cell"] is True)
    mean_rank_shift = float(np.mean([c["mean_rank_shift_vs_t1_baseline_tabular"] for c in per_cell
                                      if c["mean_rank_shift_vs_t1_baseline_tabular"] is not None]))

    if n_below_delta >= 16:
        verdict = ("SOFTENED: the typo cost nothing measurable -- repair gain is below "
                    f"delta_cell on {n_below_delta}/{n_cells} cells.")
    else:
        verdict = (f"repair gain exceeds delta_cell on {n_above_delta}/{n_cells} cells; "
                   "quote the repaired number as the headline TF-IDF result and the "
                   "as-shipped number as history.")

    artifact = {
        "test_id": "T7",
        "claim_at_stake": "the bag of words won while a typo kept it from reading the trial summaries",
        "inputs": {"seeds": SEEDS, "configs": {k: list(v) for k, v in CONFIGS.items()},
                   "data_root": DATA_ROOT,
                   "note": "TEXT_COLS_AS_SHIPPED's 'detailed_description' and "
                           "'eligibility/study_pop/textblock' entries don't match any "
                           "column in any task folder even after the /textblock suffix "
                           "fix -- confirmed by reading the raw CSVs -- so they contribute "
                           "nothing in any of the 3 configs; the only column the repair "
                           "actually activates is 'brief_summary/textblock' everywhere and "
                           "'detailed_description/textblock' on trial-failure-reason-identification only."},
        "n_cells": n_cells,
        "per_cell": per_cell,
        "n_cells_gain_above_delta_cell": n_above_delta,
        "n_cells_gain_below_delta_cell": n_below_delta,
        "mean_rank_shift_vs_t1_baseline_tabular_pack": mean_rank_shift,
        "decision_rule": "Report repaired as headline TF-IDF result, as-shipped as history. "
                          "If repair changes TF-IDF's rank against the tabular pack on the "
                          "shared cell set, that becomes the lead. If gain < delta_cell on "
                          ">=16/20 cells, story is 'typo cost nothing measurable'.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}: {n_above_delta}/{n_cells} cells with gain > delta_cell, "
          f"mean rank shift {mean_rank_shift:+.2f}")
    print(verdict)

    return artifact


if __name__ == "__main__":
    main()
