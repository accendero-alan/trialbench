"""T16 -- Blend on 16 cells instead of 2 (fit + persist).

newsletter-part2-test-plan.md, T16. `blend_curve_probe.py`'s "optimal mix
flips by task" claim rests on 2 Phase1 binary cells, one seed -- indistinguishable
from two draws from a distribution with no task structure at all. This
generalizes it to all 16 binary cells x 5 seeds x the full 101-point weight
grid, using the identical rank-fusion method (unchanged from
`blend_curve_probe.py`/signal-analysis.md): tabular consensus = mean of the
10 tabular Tier A methods' per-split rank percentiles; text = tfidf_logreg;
blend(w) = (1-w)*tabular_rank + w*text_rank; w* selected on validation, scored
on test; test argmax recorded separately as an oracle, never used for
selection.

Every method fit is persisted via `src/eval/predictions.py` (P2) to
`results/predictions/<task>/<phase>/<method>_<seed>.parquet` -- resumable
(skips refit if already persisted) and the store T17/T18/T19 read from
without refitting.

Ends with a Kruskal-Wallis test across the 4 tasks for structure in the
per-(cell,seed) selected weight w*_valid.

Usage: `python -m experiments.t16_blend_full`
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kruskal, rankdata
from sklearn.metrics import average_precision_score

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import TabularFeaturizer
from src.data.loader import load_task_phase
from src.eval.predictions import has_predictions, load_predictions, save_predictions
from src.methods.registry import get as get_method

BINARY_TASKS = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn", "patient_dropout_rate_yn"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
SEEDS = [42, 7, 123, 2024, 5]
TABULAR_METHODS = ["logreg_l1", "logreg_l2", "svm_linear", "knn", "random_forest",
                   "extra_trees", "hist_gbm", "xgboost", "lightgbm", "catboost"]
TEXT_METHOD = "tfidf_logreg"
ALL_METHODS = TABULAR_METHODS + [TEXT_METHOD]
DATA_ROOT = "data"
RESULTS_DIR = "results"
N_GRID = 101
OUT_PATH = "results/experiments/t16_blend_full.json"


def _scores(proba):
    return proba if proba.ndim == 1 else proba[:, 1]


def _rank_pct(s):
    r = rankdata(np.asarray(s, dtype=float), method="average")
    return (r - 1.0) / max(len(r) - 1, 1)


def _fit_or_load(method, task, phase, seed, td, fz=None, Xtr=None, Xva=None, Xte=None):
    """Returns (valid_proba, test_proba) for this method, persisting a fresh
    fit or reusing an already-persisted one (P2 resumability)."""
    if has_predictions(RESULTS_DIR, task, phase, method, seed):
        df = load_predictions(RESULTS_DIR, task, phase, method, seed)
        valid_proba = df[df["split"] == "valid"]["y_proba"].to_numpy()
        test_proba = df[df["split"] == "test"]["y_proba"].to_numpy()
        return valid_proba, test_proba

    m = get_method(method)(task_type=td.task_type, num_classes=td.num_classes, seed=seed)
    if method == TEXT_METHOD:
        m.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)
        valid_proba = _scores(m.predict_proba(td.X_valid))
        test_proba = _scores(m.predict_proba(td.X_test))
    else:
        m.fit(Xtr, td.y_train, Xva, td.y_valid)
        valid_proba = _scores(m.predict_proba(Xva))
        test_proba = _scores(m.predict_proba(Xte))

    save_predictions(RESULTS_DIR, task, phase, method, seed,
                      valid_ids=td.X_valid.index, valid_proba=valid_proba, valid_y=td.y_valid,
                      test_ids=td.X_test.index, test_proba=test_proba, test_y=td.y_test)
    return valid_proba, test_proba


def run_cell_seed(task, phase, seed):
    td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
    fz = TabularFeaturizer(task_type=td.task_type)
    Xtr = fz.fit_transform(td.X_train, td.y_train)
    Xva, Xte = fz.transform(td.X_valid), fz.transform(td.X_test)

    va_ranks, te_ranks = [], []
    for method in TABULAR_METHODS:
        vp, tp = _fit_or_load(method, task, phase, seed, td, fz, Xtr, Xva, Xte)
        va_ranks.append(_rank_pct(vp))
        te_ranks.append(_rank_pct(tp))
    tab_va, tab_te = np.mean(va_ranks, axis=0), np.mean(te_ranks, axis=0)

    txt_vp, txt_tp = _fit_or_load(TEXT_METHOD, task, phase, seed, td)
    txt_va, txt_te = _rank_pct(txt_vp), _rank_pct(txt_tp)

    ws = np.linspace(0.0, 1.0, N_GRID)
    curve_va = [float(average_precision_score(td.y_valid, (1 - w) * tab_va + w * txt_va)) for w in ws]
    curve_te = [float(average_precision_score(td.y_test, (1 - w) * tab_te + w * txt_te)) for w in ws]

    i_val = int(np.argmax(curve_va))
    i_te = int(np.argmax(curve_te))
    prauc_tabular_only, prauc_text_only = curve_te[0], curve_te[-1]
    best_single_source = max(prauc_tabular_only, prauc_text_only)

    return {
        "task": task, "phase": phase, "seed": seed,
        "w": ws.tolist(), "curve_test": curve_te, "curve_valid": curve_va,
        "w_star_valid": float(ws[i_val]), "prauc_test_at_w_star": curve_te[i_val],
        "w_argmax_test": float(ws[i_te]), "prauc_test_at_argmax": curve_te[i_te],
        "prauc_tabular_only": prauc_tabular_only, "prauc_text_only": prauc_text_only,
        "best_single_source_prauc": best_single_source,
        "lift_over_best_single_source": curve_te[i_val] - best_single_source,
    }


def main():
    cells = [(task, phase) for task in BINARY_TASKS for phase in PHASES]
    with Timer() as t:
        per_cell_seed = []
        for task, phase in cells:
            for seed in SEEDS:
                rec = run_cell_seed(task, phase, seed)
                per_cell_seed.append(rec)
                print(f"  {task}/{phase} seed{seed}: w*={rec['w_star_valid']:.2f} "
                      f"lift={rec['lift_over_best_single_source']:+.4f}", flush=True)

        # Kruskal-Wallis across the 4 tasks on the per-(phase,seed) selected weight
        groups = {task: [r["w_star_valid"] for r in per_cell_seed if r["task"] == task]
                  for task in BINARY_TASKS}
        kw_stat, kw_p = kruskal(*groups.values())
        task_means = {task: float(np.mean(vs)) for task, vs in groups.items()}
        between_task_spread = float(np.std(list(task_means.values())))
        within_task_seed_spread = float(np.mean([
            np.std([r["w_star_valid"] for r in per_cell_seed if r["task"] == task and r["phase"] == phase])
            for task in BINARY_TASKS for phase in PHASES
        ]))

    task_structure_confirmed = bool(kw_p < 0.05 and between_task_spread > within_task_seed_spread)
    if task_structure_confirmed:
        verdict = (f"CONFIRMED (first time with real evidence): Kruskal-Wallis across tasks "
                    f"p={kw_p:.4g} < 0.05, and between-task spread ({between_task_spread:.4f}) "
                    f"exceeds between-seed spread ({within_task_seed_spread:.4f}).")
    else:
        verdict = (f"CUT: no evidence the mix varies by task (Kruskal-Wallis p={kw_p:.4g}, "
                    f"between-task spread {between_task_spread:.4f} vs. between-seed spread "
                    f"{within_task_seed_spread:.4f}). Replace 'the optimal mix flips by task' "
                    f"with 'we have no evidence the mix varies by task'.")

    artifact = {
        "test_id": "T16",
        "claim_at_stake": "the optimal mix flips by task",
        "inputs": {"tasks": BINARY_TASKS, "phases": PHASES, "seeds": SEEDS,
                   "tabular_methods": TABULAR_METHODS, "text_method": TEXT_METHOD, "n_grid": N_GRID},
        "n_cells": len(cells), "n_cell_seed_records": len(per_cell_seed),
        "per_cell_seed": per_cell_seed,
        "task_means_w_star": task_means,
        "kruskal_wallis_statistic": float(kw_stat), "kruskal_wallis_p": float(kw_p),
        "between_task_spread": between_task_spread,
        "within_task_seed_spread": within_task_seed_spread,
        "decision_rule": "If Kruskal-Wallis across tasks is significant (p<0.05) AND "
                          "between-task spread in w* exceeds between-seed spread, CONFIRMED "
                          "for the first time. Otherwise CUT, replaced with 'no evidence the "
                          "mix varies by task'.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}")
    print(verdict)
    return artifact


if __name__ == "__main__":
    main()
