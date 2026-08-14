"""T2 -- Scaling ablation (reduced-scope directional subset).

newsletter-part2-test-plan.md, T2. TabNet and FT-Transformer were the only
scale-sensitive methods denied a StandardScaler, while logreg/knn/svm/
clinical_embeddings all get one. Is deep tabular's mediocre showing an
architecture result or a missing preprocessing step?

**Scope reduction (logged per the plan's "no silent caps" rule).** The full
test is 20 cells x 3 seeds x 2 configs, ~4 hours (FT-Transformer-dominated).
By user request this runs a directional subset instead: 6 cells (one small +
one large per data pool, covering all 4 binary task families -- mortality
and serious_adverse share an identical row pool across phases, so only one
of them contributes both a small and large cell) x 2 seeds. This CANNOT
trigger the plan's own decision rule ("exceeds delta_cell in >=5 of 20
cells"), which needs the full 20-cell grid -- the verdict below is
directional (sign and rough magnitude of the scaling effect), not a
CUT/closed determination. TabNet's scaling question is largely already
answered by T3 (which bundled scaling with class-weighting and a loss-based
early-stop metric and found a large gain) -- this isolates scaling alone,
holding the other two T3 fixes constant, as a secondary check; it's cheap
(~16s/cell) so it runs on the same 6 cells for a complete picture.

Usage: `python -m experiments.t2_deep_scaling`
"""
from __future__ import annotations

import json

import numpy as np

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import TabularFeaturizer
from src.data.loader import load_task_phase
from src.eval import metrics as M
from src.methods.registry import get as get_method

CELLS = [
    ("outcome", "Phase1"), ("outcome", "Phase2"),
    ("mortality_rate_yn", "Phase1"), ("serious_adverse_rate_yn", "Phase2"),
    ("patient_dropout_rate_yn", "Phase1"), ("patient_dropout_rate_yn", "Phase2"),
]
SEEDS = [42, 7]
DATA_ROOT = "data"
OUT_PATH = "results/experiments/t2_deep_scaling.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"


def _ft_transformer_score(Xtr, ytr, Xva, yva, Xte, yte, task_type, num_classes, seed, scaled):
    if scaled:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    m = get_method("ft_transformer")(task_type=task_type, num_classes=num_classes, seed=seed)
    m.fit(Xtr, ytr, Xva, yva)
    proba = m.predict_proba(Xte)
    return M.compute(yte, proba, task_type, num_classes)["prauc"]


def _tabnet_score(Xtr, ytr, Xva, yva, Xte, yte, task_type, num_classes, seed, scaled):
    from pytorch_tabnet.tab_model import TabNetClassifier

    Xtr, ytr = np.asarray(Xtr, dtype=float), np.asarray(ytr, dtype=np.int64)
    Xva, yva = np.asarray(Xva, dtype=float), np.asarray(yva, dtype=np.int64)
    Xte = np.asarray(Xte, dtype=float)
    if scaled:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    Xtr, Xva, Xte = Xtr.astype(np.float32), Xva.astype(np.float32), Xte.astype(np.float32)

    n = len(Xtr)
    batch_size = max(8, min(256, n))
    virtual_batch_size = max(4, min(64, batch_size))
    clf = TabNetClassifier(n_d=16, n_a=16, n_steps=3, gamma=1.3, seed=seed, verbose=0, device_name="cpu")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_name=["valid"], eval_metric=["logloss"],
            weights=1, max_epochs=100, patience=15, batch_size=batch_size,
            virtual_batch_size=virtual_batch_size, drop_last=False)
    proba = clf.predict_proba(Xte)
    classes = clf.classes_
    full = np.zeros((len(Xte), max(2, num_classes)), dtype=float)
    for j, c in enumerate(classes):
        full[:, int(c)] = proba[:, j]
    score = full[:, 1] if task_type == "binary" else full
    return M.compute(yte, score, task_type, num_classes)["prauc"]


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}

    per_cell = []
    with Timer() as t:
        for task, phase in CELLS:
            ft_scaled, ft_unscaled, tn_scaled, tn_unscaled = [], [], [], []
            for seed in SEEDS:
                td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
                fz = TabularFeaturizer(task_type=td.task_type)
                Xtr = fz.fit_transform(td.X_train, td.y_train)
                Xva, Xte = fz.transform(td.X_valid), fz.transform(td.X_test)

                ft_unscaled.append(_ft_transformer_score(Xtr, td.y_train, Xva, td.y_valid, Xte, td.y_test,
                                                          td.task_type, td.num_classes, seed, scaled=False))
                ft_scaled.append(_ft_transformer_score(Xtr, td.y_train, Xva, td.y_valid, Xte, td.y_test,
                                                        td.task_type, td.num_classes, seed, scaled=True))
                tn_unscaled.append(_tabnet_score(Xtr, td.y_train, Xva, td.y_valid, Xte, td.y_test,
                                                  td.task_type, td.num_classes, seed, scaled=False))
                tn_scaled.append(_tabnet_score(Xtr, td.y_train, Xva, td.y_valid, Xte, td.y_test,
                                                td.task_type, td.num_classes, seed, scaled=True))

            delta = delta_cells.get((task, phase), float("nan"))
            ft_gain = float(np.mean(ft_scaled)) - float(np.mean(ft_unscaled))
            tn_gain = float(np.mean(tn_scaled)) - float(np.mean(tn_unscaled))
            per_cell.append({
                "task": task, "phase": phase, "delta_cell": delta,
                "ft_transformer_unscaled_mean": float(np.mean(ft_unscaled)),
                "ft_transformer_scaled_mean": float(np.mean(ft_scaled)),
                "ft_transformer_scaling_gain": ft_gain,
                "ft_gain_exceeds_delta_cell": bool(abs(ft_gain) > delta) if not np.isnan(delta) else None,
                "tabnet_unscaled_mean": float(np.mean(tn_unscaled)),
                "tabnet_scaled_mean": float(np.mean(tn_scaled)),
                "tabnet_scaling_gain": tn_gain,
                "tabnet_gain_exceeds_delta_cell": bool(abs(tn_gain) > delta) if not np.isnan(delta) else None,
            })
            print(f"  {task}/{phase}: FT gain={ft_gain:+.4f} TabNet gain={tn_gain:+.4f} "
                  f"delta_cell={delta:.4f}", flush=True)

    n_cells = len(per_cell)
    n_ft_above = sum(1 for c in per_cell if c["ft_gain_exceeds_delta_cell"])
    n_tn_above = sum(1 for c in per_cell if c["tabnet_gain_exceeds_delta_cell"])
    mean_ft_gain = float(np.mean([c["ft_transformer_scaling_gain"] for c in per_cell]))
    mean_tn_gain = float(np.mean([c["tabnet_scaling_gain"] for c in per_cell]))

    verdict = (f"DIRECTIONAL ONLY (6/20 cells, 2 seeds -- cannot trigger the plan's own "
                f">=5-of-20 / >=16-of-20 decision thresholds). FT-Transformer: mean scaling "
                f"gain {mean_ft_gain:+.4f}, exceeds delta_cell on {n_ft_above}/{n_cells} "
                f"subset cells. TabNet (scaling isolated from T3's other two fixes): mean gain "
                f"{mean_tn_gain:+.4f}, exceeds delta_cell on {n_tn_above}/{n_cells}. "
                + ("Sign is positive and non-trivial on this subset -- consistent with scaling "
                   "mattering for FT-Transformer; a full 20-cell run would be needed to confirm "
                   "or cut the 'mediocre deep tabular' framing." if mean_ft_gain > 0 else
                   "Sign is at or below zero on this subset -- scaling does not appear to help "
                   "FT-Transformer here, tentatively supporting closing the scaling objection, "
                   "but the full grid was not run."))

    artifact = {
        "test_id": "T2",
        "claim_at_stake": "deep tabular is mediocre -- architecture result or missing preprocessing step?",
        "inputs": {"cells": CELLS, "seeds": SEEDS,
                   "scope_note": "reduced from the full 20 cells x 3 seeds (~4h) to 6 cells x "
                                  "2 seeds (~45 min) by explicit user request for a directional "
                                  "read; this cannot resolve the plan's own CUT/closed decision "
                                  "rule, which needs the full 20-cell grid."},
        "n_cells": n_cells,
        "per_cell": per_cell,
        "mean_ft_transformer_scaling_gain": mean_ft_gain, "mean_tabnet_scaling_gain": mean_tn_gain,
        "n_cells_ft_gain_above_delta": n_ft_above, "n_cells_tabnet_gain_above_delta": n_tn_above,
        "decision_rule": "(Full-scope rule, not resolvable by this subset.) If mean "
                          "scaled-minus-unscaled gain exceeds delta_cell in >=5/20 cells, CUT "
                          "and replaced with the preprocessing finding. If below delta_cell in "
                          ">=16/20 cells, the scaling objection is closed.",
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
