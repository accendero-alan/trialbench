"""T10 -- Registration vintage as a confound.

newsletter-part2-test-plan.md, T10. On outcome/Phase3, test-set prevalence
varies sharply across NCT-id bins (0.530/0.619/0.624/0.337/0.000/0.000, two
bins with zero positives). Question: can vintage alone predict the label, and
does TF-IDF's advantage survive conditioning on it?

Single seed (42), matching T9's convention -- this is a confound-diagnosis
test, not a multi-seed variance question T1 already owns. All 20 cells.

Three parts per cell:
  (a) vintage-only baseline: NCT id's numeric part as a single ordinal
      feature into LightGBM (the registered `lightgbm` method's config),
      report test PR-AUC.
  (b) TF-IDF -> vintage-bin classifier: train-only quantile bins (5) of NCT
      id; fit tfidf_logreg-style multiclass logreg on text -> bin, report
      macro accuracy on test -- how legible is era in the vocabulary?
  (c) stratified re-scoring: re-score tfidf_logreg and the best GBM within
      each vintage stratum on test, pool by stratum size, compare the pooled
      TF-IDF margin over the best GBM to the unstratified margin.

Usage: `python -m experiments.t10_vintage_confound`
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import TabularFeaturizer
from src.data.loader import TASKS, load_task_phase
from src.eval import metrics as M
from src.methods.registry import get as get_method

SEED = 42
DATA_ROOT = "data"
N_BINS = 5
OUT_PATH = "results/experiments/t10_vintage_confound.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
RUNS_DIR = "results/extracted/trialbench/results/runs"
GBM_CANDIDATES = ["xgboost", "lightgbm", "catboost"]
ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]


def _nct_numeric(index):
    return np.array([int(re.sub(r"\D", "", str(i)) or 0) for i in index], dtype=float)


def _best_gbm(task, phase, seed=SEED):
    best_method, best_score = None, -np.inf
    for method in GBM_CANDIDATES:
        try:
            with open(f"{RUNS_DIR}/{task}__{phase}__{method}__seed{seed}.json") as f:
                rec = json.load(f)
        except FileNotFoundError:
            continue
        if rec.get("status") == "ok" and rec["point"]["prauc"] > best_score:
            best_method, best_score = method, rec["point"]["prauc"]
    return best_method


def _headline(y_true, proba, task_type, num_classes):
    return M.compute(y_true, proba, task_type, num_classes)["prauc"]


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}

    per_cell = []
    with Timer() as t:
        for task, phase in ALL_CELLS:
            td = load_task_phase(DATA_ROOT, task, phase, seed=SEED)
            vin_tr, vin_va, vin_te = (_nct_numeric(td.X_train.index), _nct_numeric(td.X_valid.index),
                                       _nct_numeric(td.X_test.index))

            # (a) vintage-only baseline
            GbmCls = get_method("lightgbm")
            vintage_model = GbmCls(task_type=td.task_type, num_classes=td.num_classes, seed=SEED)
            vintage_model.fit(vin_tr.reshape(-1, 1), td.y_train, vin_va.reshape(-1, 1), td.y_valid)
            vintage_proba = vintage_model.predict_proba(vin_te.reshape(-1, 1))
            vintage_prauc = _headline(td.y_test, vintage_proba, td.task_type, td.num_classes)

            # bin edges from train vintage only
            try:
                bin_edges = np.unique(np.quantile(vin_tr, np.linspace(0, 1, N_BINS + 1)))
                bin_tr = np.clip(np.digitize(vin_tr, bin_edges[1:-1]), 0, len(bin_edges) - 2)
                bin_te = np.clip(np.digitize(vin_te, bin_edges[1:-1]), 0, len(bin_edges) - 2)
                n_bins_actual = len(bin_edges) - 1
            except Exception:
                bin_tr = np.zeros(len(vin_tr), dtype=int)
                bin_te = np.zeros(len(vin_te), dtype=int)
                n_bins_actual = 1

            # (b) TF-IDF -> vintage bin (macro accuracy)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from src.data.features import concat_text

            vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            Xtr_txt = vec.fit_transform(concat_text(td.X_train))
            Xte_txt = vec.transform(concat_text(td.X_test))
            can_classify_vintage = n_bins_actual > 1 and len(np.unique(bin_tr)) > 1
            vintage_macro_acc = float("nan")
            if can_classify_vintage:
                bin_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
                bin_clf.fit(Xtr_txt, bin_tr)
                bin_pred = bin_clf.predict(Xte_txt)
                per_class_acc = []
                for b in np.unique(bin_te):
                    mask = bin_te == b
                    if mask.sum() > 0:
                        per_class_acc.append(accuracy_score(bin_te[mask], bin_pred[mask]))
                vintage_macro_acc = float(np.mean(per_class_acc)) if per_class_acc else float("nan")

            # tfidf_logreg and best-GBM full-test scoring
            TfidfCls = get_method("tfidf_logreg")
            tfidf_model = TfidfCls(task_type=td.task_type, num_classes=td.num_classes, seed=SEED)
            tfidf_model.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)
            tfidf_proba = tfidf_model.predict_proba(td.X_test)
            tfidf_full_prauc = _headline(td.y_test, tfidf_proba, td.task_type, td.num_classes)

            gbm_name = _best_gbm(task, phase)
            gbm_proba = None
            if gbm_name is not None:
                fz = TabularFeaturizer(task_type=td.task_type)
                Xtr = fz.fit_transform(td.X_train, td.y_train)
                GbmCls2 = get_method(gbm_name)
                gbm_model = GbmCls2(task_type=td.task_type, num_classes=td.num_classes, seed=SEED)
                gbm_model.fit(Xtr, td.y_train, fz.transform(td.X_valid), td.y_valid)
                gbm_proba = gbm_model.predict_proba(fz.transform(td.X_test))
            gbm_full_prauc = (_headline(td.y_test, gbm_proba, td.task_type, td.num_classes)
                               if gbm_proba is not None else float("nan"))
            unstratified_margin = tfidf_full_prauc - gbm_full_prauc

            # (c) stratified re-scoring, pooled by stratum size
            strat_tfidf, strat_gbm, strat_n = [], [], []
            for b in np.unique(bin_te):
                mask = bin_te == b
                n_b = int(mask.sum())
                if n_b < 20 or len(np.unique(np.asarray(td.y_test)[mask])) < 2:
                    continue
                y_b = np.asarray(td.y_test)[mask]
                tfidf_b = _headline(y_b, tfidf_proba[mask] if td.task_type == "binary" else tfidf_proba[mask, :],
                                     td.task_type, td.num_classes)
                strat_tfidf.append(tfidf_b * n_b)
                strat_n.append(n_b)
                if gbm_proba is not None:
                    gbm_b = _headline(y_b, gbm_proba[mask] if td.task_type == "binary" else gbm_proba[mask, :],
                                       td.task_type, td.num_classes)
                    strat_gbm.append(gbm_b * n_b)
            total_n = sum(strat_n) if strat_n else 0
            pooled_tfidf = (sum(strat_tfidf) / total_n) if total_n else float("nan")
            pooled_gbm = (sum(strat_gbm) / total_n) if total_n and strat_gbm else float("nan")
            pooled_margin = pooled_tfidf - pooled_gbm if not np.isnan(pooled_gbm) else float("nan")
            margin_shrinkage_frac = (1 - pooled_margin / unstratified_margin
                                      if unstratified_margin and not np.isnan(pooled_margin) else float("nan"))

            delta = delta_cells.get((task, phase), float("nan"))
            vintage_within_delta = bool(abs(vintage_prauc - tfidf_full_prauc) <= delta) if not np.isnan(delta) else None

            if vintage_within_delta:
                cell_verdict = "CUT (vintage-only within delta_cell of TF-IDF)"
            elif not np.isnan(margin_shrinkage_frac) and margin_shrinkage_frac > 0.5:
                cell_verdict = f"SOFTENED (margin shrinks {margin_shrinkage_frac:.0%} under stratification)"
            else:
                cell_verdict = "CONFIRMED"

            per_cell.append({
                "task": task, "phase": phase, "delta_cell": delta,
                "vintage_only_prauc": vintage_prauc, "tfidf_full_prauc": tfidf_full_prauc,
                "vintage_within_delta_of_tfidf": vintage_within_delta,
                "vintage_bin_macro_accuracy_from_text": vintage_macro_acc, "n_bins": n_bins_actual,
                "best_gbm": gbm_name, "gbm_full_prauc": gbm_full_prauc,
                "unstratified_tfidf_minus_gbm": unstratified_margin,
                "pooled_stratified_tfidf": pooled_tfidf, "pooled_stratified_gbm": pooled_gbm,
                "pooled_stratified_margin": pooled_margin,
                "margin_shrinkage_fraction": margin_shrinkage_frac,
                "cell_verdict": cell_verdict,
            })
            print(f"  {task}/{phase}: vintage_only={vintage_prauc:.4f} tfidf={tfidf_full_prauc:.4f} "
                  f"vintage_acc={vintage_macro_acc:.3f} shrink={margin_shrinkage_frac} -> {cell_verdict}", flush=True)

    n_cut = sum(1 for c in per_cell if c["cell_verdict"].startswith("CUT"))
    n_softened = sum(1 for c in per_cell if c["cell_verdict"].startswith("SOFTENED"))
    n_confirmed = sum(1 for c in per_cell if c["cell_verdict"] == "CONFIRMED")
    overall_verdict = (f"{n_cut} cell(s) CUT (vintage-only indistinguishable from TF-IDF), "
                        f"{n_softened} SOFTENED (margin shrinks >50% under vintage stratification), "
                        f"{n_confirmed} CONFIRMED.")

    artifact = {
        "test_id": "T10",
        "claim_at_stake": "the text model learned clinical content, not era",
        "inputs": {"seed": SEED, "n_bins_target": N_BINS, "data_root": DATA_ROOT,
                   "note": "single seed (42), matching T9's convention for a confound-diagnosis "
                           "test rather than a multi-seed variance question."},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "n_cut": n_cut, "n_softened": n_softened, "n_confirmed": n_confirmed,
        "decision_rule": "If vintage-only baseline is within delta_cell of TF-IDF on a cell, "
                          "CUT on that cell. If TF-IDF's margin over the best GBM shrinks by "
                          ">50% under vintage-stratified scoring, SOFTENED with the fraction "
                          "stated. Otherwise CONFIRMED.",
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
