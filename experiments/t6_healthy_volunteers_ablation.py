"""T6 -- Ablate the eligibility checkbox instead of permuting it.

newsletter-part2-test-plan.md, T6. The most quotable number in Surprise 1:
`eligibility/healthy_volunteers` is claimed as 51.7% of FT-Transformer's
skill above chance on mortality/Phase1 -- a number derived from permutation
importance plus a prevalence file, appearing nowhere in the repo. This
measures the direct drop-column effect instead.

Scope decision (logged here per the plan's "no silent caps" rule): the field
turns out to be populated (40-68%) on **every** cell, not just a few -- so
"every cell where the field is populated" is effectively all 20. Running
FT-Transformer with/without across all 20 cells x 5 seeds x 2 configs would
be ~7 hours at its own ~2 min/cell rate, blowing Tier 2's ~4-hour budget for
six tests. FT-Transformer is run only on **mortality/Phase1**, the specific
cell the 51.7% claim is about -- this directly re-measures the quoted number.
The best GBM and tfidf_logreg (cheap) run the full 20-cell x 5-seed ablation,
giving broad coverage for the design-confound cross-tab check.

Usage: `python -m experiments.t6_healthy_volunteers_ablation`
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import TabularFeaturizer
from src.data.loader import TASKS, load_task_phase
from src.eval import metrics as M
from src.methods.registry import get as get_method

SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"
DROP_COL = "eligibility/healthy_volunteers"
FT_TRANSFORMER_CELL = ("mortality_rate_yn", "Phase1")
GBM_CANDIDATES = ["xgboost", "lightgbm", "catboost"]
RUNS_DIR = "results/extracted/trialbench/results/runs"
OUT_PATH = "results/experiments/t6_healthy_volunteers_ablation.json"
ALL_CELLS = [(task, phase) for task in TASKS for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]]


def _best_gbm(task, phase, seed=42):
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


def _drop_col(X):
    return X.drop(columns=[DROP_COL], errors="ignore")


def _tabular_fit_score(method_name, td, drop, seed):
    Xtr, Xva, Xte = (_drop_col(td.X_train), _drop_col(td.X_valid), _drop_col(td.X_test)) if drop \
        else (td.X_train, td.X_valid, td.X_test)
    fz = TabularFeaturizer(task_type=td.task_type)
    Ztr = fz.fit_transform(Xtr, td.y_train)
    Zva, Zte = fz.transform(Xva), fz.transform(Xte)
    Cls = get_method(method_name)
    m = Cls(task_type=td.task_type, num_classes=td.num_classes, seed=seed)
    m.fit(Ztr, td.y_train, Zva, td.y_valid)
    proba = m.predict_proba(Zte)
    return M.compute(td.y_test, proba, td.task_type, td.num_classes)["prauc"]


def _tfidf_fit_score(td, drop, seed):
    # tfidf_logreg reads free-text columns only via concat_text; dropping a
    # tabular-only eligibility flag has no effect on its inputs, but it's run
    # both ways for a uniform table (drop is a no-op here, as expected).
    Cls = get_method("tfidf_logreg")
    m = Cls(task_type=td.task_type, num_classes=td.num_classes, seed=seed)
    m.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)
    proba = m.predict_proba(td.X_test)
    return M.compute(td.y_test, proba, td.task_type, td.num_classes)["prauc"]


def _ft_transformer_fit_score(td, drop, seed):
    Xtr, Xva, Xte = (_drop_col(td.X_train), _drop_col(td.X_valid), _drop_col(td.X_test)) if drop \
        else (td.X_train, td.X_valid, td.X_test)
    fz = TabularFeaturizer(task_type=td.task_type)
    Ztr = fz.fit_transform(Xtr, td.y_train)
    Zva, Zte = fz.transform(Xva), fz.transform(Xte)
    Cls = get_method("ft_transformer")
    m = Cls(task_type=td.task_type, num_classes=td.num_classes, seed=seed)
    m.fit(Ztr, td.y_train, Zva, td.y_valid)
    proba = m.predict_proba(Zte)
    return M.compute(td.y_test, proba, td.task_type, td.num_classes)["prauc"]


def _skill(prauc, base_rate):
    return prauc - base_rate


def main():
    per_cell = []
    with Timer() as t:
        for task, phase in ALL_CELLS:
            gbm_name = _best_gbm(task, phase)
            with_praucs = {"gbm": [], "tfidf": [], "ft_transformer": []}
            without_praucs = {"gbm": [], "tfidf": [], "ft_transformer": []}
            base_rates = []
            run_ft = (task, phase) == FT_TRANSFORMER_CELL
            for seed in SEEDS:
                td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
                base_rates.append(float(td.y_test.mean()) if td.task_type == "binary" else float("nan"))
                if gbm_name is not None:
                    with_praucs["gbm"].append(_tabular_fit_score(gbm_name, td, drop=False, seed=seed))
                    without_praucs["gbm"].append(_tabular_fit_score(gbm_name, td, drop=True, seed=seed))
                with_praucs["tfidf"].append(_tfidf_fit_score(td, drop=False, seed=seed))
                without_praucs["tfidf"].append(_tfidf_fit_score(td, drop=True, seed=seed))
                if run_ft:
                    with_praucs["ft_transformer"].append(_ft_transformer_fit_score(td, drop=False, seed=seed))
                    without_praucs["ft_transformer"].append(_ft_transformer_fit_score(td, drop=True, seed=seed))

            base_rate = float(np.mean(base_rates)) if base_rates and not np.isnan(base_rates[0]) else float("nan")
            cell_out = {"task": task, "phase": phase, "best_gbm": gbm_name, "base_rate": base_rate,
                        "ft_transformer_run": run_ft, "methods": {}}
            for key in ("gbm", "tfidf", "ft_transformer"):
                if not with_praucs[key]:
                    continue
                w, wo = float(np.mean(with_praucs[key])), float(np.mean(without_praucs[key]))
                skill_with, skill_without = _skill(w, base_rate), _skill(wo, base_rate)
                pct_of_skill_lost = ((skill_with - skill_without) / skill_with * 100
                                      if skill_with else float("nan"))
                cell_out["methods"][key] = {
                    "prauc_with_column": w, "prauc_without_column": wo,
                    "drop_in_prauc": w - wo,
                    "skill_above_base_with": skill_with, "skill_above_base_without": skill_without,
                    "pct_of_skill_above_base_lost_by_dropping": pct_of_skill_lost,
                }
            per_cell.append(cell_out)
            print(f"  {task}/{phase}: " + ", ".join(
                f"{k}: -{v['pct_of_skill_above_base_lost_by_dropping']:.1f}% skill"
                for k, v in cell_out["methods"].items()), flush=True)

        # cross-tab: is healthy_volunteers a proxy for "Phase1, nobody dies"?
        crosstabs = {}
        for task in TASKS:
            frames = []
            for phase in ["Phase1", "Phase2", "Phase3", "Phase4"]:
                td = load_task_phase(DATA_ROOT, task, phase, seed=42)
                X_all = pd.concat([td.X_train, td.X_valid, td.X_test])
                y_all = np.concatenate([td.y_train, td.y_valid, td.y_test]) if td.task_type == "binary" else None
                if y_all is None or DROP_COL not in X_all.columns:
                    continue
                hv = X_all[DROP_COL].astype(str)
                frames.append(pd.DataFrame({"phase": phase, "healthy_volunteers": hv, "y": y_all}))
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            ct = df.groupby(["phase", "healthy_volunteers"])["y"].agg(["mean", "count"]).reset_index()
            crosstabs[task] = ct.to_dict(orient="records")

    ft_cell = next((c for c in per_cell if c["ft_transformer_run"]), None)
    ft_pct = ft_cell["methods"]["ft_transformer"]["pct_of_skill_above_base_lost_by_dropping"] if ft_cell else None
    if ft_pct is not None and ft_pct < 40:
        verdict = (f"CUT: measured drop-column effect on mortality/Phase1 is {ft_pct:.1f}% of "
                    f"FT-Transformer's skill above base rate -- well below the quoted 51.7%. "
                    f"The 'over half of everything the model earns' line is replaced by this figure.")
    elif ft_pct is not None:
        verdict = (f"Drop-column measurement on mortality/Phase1: {ft_pct:.1f}% of skill lost "
                    f"(quoted permutation-based figure was 51.7%).")
    else:
        verdict = "FT-Transformer ablation did not run (unexpected)."

    artifact = {
        "test_id": "T6",
        "claim_at_stake": "healthy_volunteers is 51.7% of FT-Transformer's skill above chance",
        "inputs": {"seeds": SEEDS, "drop_column": DROP_COL,
                   "scope_note": "field populated 40-68% on every cell, not a few -- FT-Transformer "
                                  "capped to mortality/Phase1 (the cell the quoted number is about) "
                                  "for budget; best GBM and tfidf_logreg run the full 20-cell grid."},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "crosstabs_phase_and_label": crosstabs,
        "decision_rule": "Quote the drop-column number, not the permutation number. If the "
                          "measured drop puts the field below 40% of skill above chance, the "
                          "'over half' line is CUT and replaced with the measured figure. If "
                          "the cross-tab shows near-collinearity with phase/condition, SOFTENED "
                          "to a design-confound finding.",
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
