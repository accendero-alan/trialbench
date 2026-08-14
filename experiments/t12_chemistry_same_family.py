"""T12 -- Re-run every contrast within one model family.

newsletter-part2-test-plan.md, T12. `smiles_chemical_signal.md` asserts
"every contrast that carries a conclusion is within one model family." It is
not: presence and descriptors use LightGBM, fingerprints and drug-id use
logistic regression (see `experiments/smiles_signal_probe.py`'s block/fit
pairing). The headline +0.086 and novel-scaffold +0.167/+0.151 both cross
families.

Runs all four representations (presence control, descriptors, Morgan
fingerprints, exact-SMILES lookup/drug-id) through **both** LightGBM and
logistic regression, all 16 binary cells, 5 seeds. Every contrast is reported
twice -- once per family -- so a same-family headline can replace the
cross-family one, and any sign flip between families is caught rather than
hidden.

Reuses `src/data/mol_features.py` for featurization (identical to
smiles_signal_probe.py's blocks) and the same two fit functions
(_fit_gbm / _fit_sparse_logreg) so a same-family LightGBM contrast reproduces
the original probe's LightGBM numbers exactly, and only the previously
cross-family half (fingerprints/drug-id under LightGBM, descriptors under
logistic regression) is new.

Predictions are persisted via P2 (results/predictions/) under method names
`chem_<block>_<family>`, keyed like everything else, so T13 can reuse the
LightGBM-family fingerprint/drug-id fits without refitting.

Usage: `python -m experiments.t12_chemistry_same_family`
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score

from experiments._common import Timer, git_sha, write_artifact
from src.data import mol_features as mf
from src.data.loader import load_task_phase
from src.eval.predictions import save_predictions

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

BINARY_TASKS = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn", "patient_dropout_rate_yn"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"
RESULTS_DIR = "results"
OUT_PATH = "results/experiments/t12_chemistry_same_family.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
CELLS = [(task, phase) for task in BINARY_TASKS for phase in PHASES]
BLOCKS = ["presence", "descriptors", "morgan_fp", "drug_id"]


def _fit_gbm(Xtr, ytr, seed):
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                         n_jobs=-1, random_state=seed, verbose=-1)
    clf.fit(Xtr, ytr)
    return clf


def _fit_logreg(Xtr, ytr, Xva, yva, seed):
    from sklearn.linear_model import LogisticRegression
    best, best_ap = None, -np.inf
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, penalty="l2", solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(Xtr, ytr)
        ap = average_precision_score(yva, clf.predict_proba(Xva)[:, 1])
        if ap > best_ap:
            best, best_ap = clf, ap
    return best


def _impute_scale(desc_tr, others):
    med = np.nanmedian(desc_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def _fill(M):
        return np.nan_to_num(np.where(np.isfinite(M), M, med), nan=0.0, posinf=0.0, neginf=0.0)

    tr = _fill(desc_tr)
    mu, sd = tr.mean(axis=0), tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (tr - mu) / sd, [((_fill(M)) - mu) / sd for M in others]


def _build_blocks(task, phase, seed):
    td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
    lists = {s: mf.smiles_lists(X) for s, X in
             (("train", td.X_train), ("valid", td.X_valid), ("test", td.X_test))}
    all_smiles = {m for ls in lists.values() for l in ls for m in l}
    feats = mf.featurize_molecules(sorted(all_smiles))
    agg = {s: mf.aggregate(lists[s], feats) for s in lists}
    pres = {s: agg[s][0] for s in agg}
    desc_raw = {s: agg[s][1] for s in agg}
    fp = {s: agg[s][2].astype(float) for s in agg}

    desc_tr, (desc_va, desc_te) = _impute_scale(desc_raw["train"], [desc_raw["valid"], desc_raw["test"]])
    desc = {"train": desc_tr, "valid": desc_va, "test": desc_te}

    drug_vocab = sorted({m for l in lists["train"] for m in l if m in feats})
    drug = {s: mf.vocab_matrix([[m for m in l if m in feats] for l in lists[s]], drug_vocab) for s in lists}

    def cat(*mats_by_split):
        return {s: np.hstack([m[s] for m in mats_by_split]) for s in ("train", "valid", "test")}

    blocks = {
        "presence": pres, "descriptors": cat(pres, desc),
        "morgan_fp": cat(pres, fp), "drug_id": cat(pres, drug),
    }
    return td, blocks


def main():
    per_cell = []
    with Timer() as t:
        for task, phase in CELLS:
            per_seed = []
            for seed in SEEDS:
                td, blocks = _build_blocks(task, phase, seed)
                fam_scores = {"lightgbm": {}, "logreg": {}}
                for block_name in BLOCKS:
                    M = blocks[block_name]
                    gbm = _fit_gbm(M["train"], td.y_train, seed)
                    p_va_g, p_te_g = gbm.predict_proba(M["valid"])[:, 1], gbm.predict_proba(M["test"])[:, 1]
                    fam_scores["lightgbm"][block_name] = float(average_precision_score(td.y_test, p_te_g))
                    save_predictions(RESULTS_DIR, task, phase, f"chem_{block_name}_lightgbm", seed,
                                      valid_ids=td.X_valid.index, valid_proba=p_va_g, valid_y=td.y_valid,
                                      test_ids=td.X_test.index, test_proba=p_te_g, test_y=td.y_test)

                    lr = _fit_logreg(M["train"], td.y_train, M["valid"], td.y_valid, seed)
                    p_va_l, p_te_l = lr.predict_proba(M["valid"])[:, 1], lr.predict_proba(M["test"])[:, 1]
                    fam_scores["logreg"][block_name] = float(average_precision_score(td.y_test, p_te_l))
                    save_predictions(RESULTS_DIR, task, phase, f"chem_{block_name}_logreg", seed,
                                      valid_ids=td.X_valid.index, valid_proba=p_va_l, valid_y=td.y_valid,
                                      test_ids=td.X_test.index, test_proba=p_te_l, test_y=td.y_test)
                per_seed.append(fam_scores)

            mean_scores = {fam: {b: float(np.mean([ps[fam][b] for ps in per_seed])) for b in BLOCKS}
                           for fam in ("lightgbm", "logreg")}
            contrasts = {fam: {b: mean_scores[fam][b] - mean_scores[fam]["presence"] for b in BLOCKS if b != "presence"}
                         for fam in ("lightgbm", "logreg")}
            sign_flips = [b for b in contrasts["lightgbm"]
                          if np.sign(contrasts["lightgbm"][b]) != np.sign(contrasts["logreg"][b])
                          and abs(contrasts["lightgbm"][b]) > 1e-4 and abs(contrasts["logreg"][b]) > 1e-4]

            per_cell.append({
                "task": task, "phase": phase, "mean_scores_by_family": mean_scores,
                "same_family_contrasts_vs_presence": contrasts, "sign_flips_between_families": sign_flips,
            })
            print(f"  {task}/{phase}: lgbm(fp-presence)={contrasts['lightgbm']['morgan_fp']:+.4f} "
                  f"logreg(fp-presence)={contrasts['logreg']['morgan_fp']:+.4f} "
                  f"flips={sign_flips}", flush=True)

    all_flips = [c for c in per_cell if c["sign_flips_between_families"]]
    mean_fp_lgbm = float(np.mean([c["same_family_contrasts_vs_presence"]["lightgbm"]["morgan_fp"] for c in per_cell]))
    mean_fp_logreg = float(np.mean([c["same_family_contrasts_vs_presence"]["logreg"]["morgan_fp"] for c in per_cell]))

    verdict = (f"Same-family headline (morgan_fp - presence): LightGBM {mean_fp_lgbm:+.4f}, "
                f"logreg {mean_fp_logreg:+.4f} (previously estimated same-family ~+0.074, "
                f"cross-family headline was +0.086). ")
    if all_flips:
        verdict += (f"{len(all_flips)} cell(s) show a sign flip between families on at least one "
                    f"contrast -- those specific contrasts are CUT: "
                    + "; ".join(f"{c['task']}/{c['phase']}: {c['sign_flips_between_families']}" for c in all_flips))
    else:
        verdict += "No sign flips between families on any contrast."

    artifact = {
        "test_id": "T12",
        "claim_at_stake": "every contrast that carries a conclusion is within one model family",
        "inputs": {"seeds": SEEDS, "blocks": BLOCKS, "families": ["lightgbm", "logreg"]},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "mean_fp_minus_presence_lightgbm": mean_fp_lgbm, "mean_fp_minus_presence_logreg": mean_fp_logreg,
        "n_cells_with_sign_flip": len(all_flips),
        "decision_rule": "Quote only same-family contrasts. Headline becomes the same-family "
                          "figure. If any contrast flips sign between families, that contrast "
                          "is CUT. Report the measured family-artifact magnitude.",
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
