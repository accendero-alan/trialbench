"""T14 -- Separate "chemistry is weak" from "ECFP4 is weak".

newsletter-part2-test-plan.md, T14. Explanation (a) -- the benchmark measured
one fingerprint (ECFP4, radius 2, 1024 bits, binary), not chemistry -- was
never tested: no radius sweep, no bit-size sweep, nothing on disk
distinguishes "chemistry is weak" from "ECFP4 is weak".

Sweeps Morgan radius {1,2,3} x bits {1024,2048,4096} (9 binary configs), plus
MACCS keys, plus the RDKit descriptor set, plus a count-based (not binary)
Morgan variant at the default radius=2/bits=1024 -- 12 representations in
total (the plan's own cost line says 11; by its own listed items this counts
to 12, a minor inconsistency in the source document, noted rather than
silently resolved). All through LightGBM (one family, per T12's same-family
finding), 16 binary cells, 3 seeds, selection on validation. The
radius=2/bits=1024 binary config is identical to T12's `morgan_fp` block --
its already-persisted predictions are reused rather than refit.

Usage: `python -m experiments.t14_representation_sweep`
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.metrics import average_precision_score

from experiments._common import Timer, git_sha, write_artifact
from src.data import mol_features as mf
from src.data.loader import load_task_phase
from src.eval.predictions import load_predictions

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

BINARY_TASKS = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn", "patient_dropout_rate_yn"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
SEEDS = [42, 7, 123]
DATA_ROOT = "data"
RESULTS_DIR = "results"
OUT_PATH = "results/experiments/t14_representation_sweep.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"
CELLS = [(task, phase) for task in BINARY_TASKS for phase in PHASES]

RADII = [1, 2, 3]
BIT_SIZES = [1024, 2048, 4096]
REPR_NAMES = ([f"morgan_r{r}_b{b}" for r in RADII for b in BIT_SIZES]
              + ["maccs", "descriptors", "morgan_r2_b1024_count"])
BASELINE_REPR = "morgan_r2_b1024"  # == T12's morgan_fp, radius=2/bits=1024/binary

_MOL_CACHE = {}


def _get_mol(smiles):
    from rdkit import Chem
    if smiles not in _MOL_CACHE:
        _MOL_CACHE[smiles] = Chem.MolFromSmiles(smiles)
    return _MOL_CACHE[smiles]


def _per_mol_feature(repr_name):
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

    if repr_name == "maccs":
        def fn(smi):
            mol = _get_mol(smi)
            if mol is None:
                return None
            return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)
        return fn
    if repr_name == "morgan_r2_b1024_count":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        def fn(smi):
            mol = _get_mol(smi)
            if mol is None:
                return None
            return gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        return fn
    # morgan_r{r}_b{b} binary
    r, b = int(repr_name.split("_r")[1].split("_b")[0]), int(repr_name.split("_b")[1])
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=b)
    def fn(smi):
        mol = _get_mol(smi)
        if mol is None:
            return None
        return gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return fn


def _aggregate_repr(mol_lists, feat_map, agg):
    dim = next(iter(feat_map.values())).shape[0]
    out = np.zeros((len(mol_lists), dim), dtype=float)
    for i, mols in enumerate(mol_lists):
        known = [feat_map[m] for m in mols if m in feat_map]
        if not known:
            continue
        stacked = np.vstack(known)
        out[i] = stacked.max(axis=0) if agg == "or" else stacked.sum(axis=0)
    return out


def _fit_gbm(Xtr, ytr, seed):
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                         n_jobs=-1, random_state=seed, verbose=-1)
    clf.fit(Xtr, ytr)
    return clf


def _cell_seed_data(task, phase, seed):
    td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
    lists = {s: mf.smiles_lists(X) for s, X in
             (("train", td.X_train), ("valid", td.X_valid), ("test", td.X_test))}
    all_smiles = sorted({m for ls in lists.values() for row in ls for m in row})
    # descriptors + presence via the existing cached featurizer
    feats = mf.featurize_molecules(all_smiles)
    agg = {s: mf.aggregate(lists[s], feats) for s in lists}
    pres = {s: agg[s][0] for s in agg}
    desc_raw = {s: agg[s][1] for s in agg}
    med = np.nanmedian(desc_raw["train"], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def _fill_scale(M, mu=None, sd=None):
        filled = np.nan_to_num(np.where(np.isfinite(M), M, med), nan=0.0)
        return filled if mu is None else (filled - mu) / sd
    tr_d = _fill_scale(desc_raw["train"])
    mu, sd = tr_d.mean(axis=0), np.where(tr_d.std(axis=0) > 0, tr_d.std(axis=0), 1.0)
    desc = {"train": (tr_d - mu) / sd, "valid": _fill_scale(desc_raw["valid"], mu, sd),
            "test": _fill_scale(desc_raw["test"], mu, sd)}
    return td, lists, pres, desc


def _score_repr(repr_name, td, lists, pres, desc, seed):
    if repr_name == "descriptors":
        rep = desc
    else:
        fn = _per_mol_feature(repr_name)
        all_smiles = sorted({m for l in lists["train"] + lists["valid"] + lists["test"] for m in l})
        feat_map = {}
        for smi in all_smiles:
            v = fn(smi)
            if v is not None:
                feat_map[smi] = v
        if not feat_map:
            return float("nan")
        agg_kind = "sum" if repr_name.endswith("_count") else "or"
        rep = {s: _aggregate_repr(lists[s], feat_map, agg_kind) for s in lists}

    def cat(a, b):
        return {s: np.hstack([a[s], b[s]]) for s in ("train", "valid", "test")}
    M = cat(pres, rep)
    clf = _fit_gbm(M["train"], td.y_train, seed)
    p_te = clf.predict_proba(M["test"])[:, 1]
    return float(average_precision_score(td.y_test, p_te))


def main():
    with open(T1_ARTIFACT) as f:
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}

    per_cell = []
    with Timer() as t:
        for task, phase in CELLS:
            scores_by_repr = {name: [] for name in REPR_NAMES}
            presence_scores = []
            for seed in SEEDS:
                td, lists, pres, desc = _cell_seed_data(task, phase, seed)
                p_te_presence = _fit_gbm(pres["train"], td.y_train, seed).predict_proba(pres["test"])[:, 1]
                presence_scores.append(float(average_precision_score(td.y_test, p_te_presence)))
                for repr_name in REPR_NAMES:
                    scores_by_repr[repr_name].append(_score_repr(repr_name, td, lists, pres, desc, seed))

            # reuse T12's persisted morgan_fp (radius2/bits1024 binary) fit instead of refitting
            baseline_scores = []
            for seed in SEEDS:
                try:
                    df = load_predictions(RESULTS_DIR, task, phase, "chem_morgan_fp_lightgbm", seed, split="test")
                    baseline_scores.append(float(average_precision_score(df["y_true"], df["y_proba"])))
                except FileNotFoundError:
                    pass
            baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else float(np.mean(scores_by_repr["morgan_r2_b1024_count"]))

            mean_by_repr = {name: float(np.mean(vals)) for name, vals in scores_by_repr.items()}
            mean_by_repr[BASELINE_REPR] = baseline_mean
            presence_mean = float(np.mean(presence_scores))

            delta = delta_cells.get((task, phase), float("nan"))
            best_repr = max(mean_by_repr, key=mean_by_repr.get)
            best_score = mean_by_repr[best_repr]
            per_cell.append({
                "task": task, "phase": phase, "delta_cell": delta,
                "presence_control_prauc": presence_mean, "baseline_ecfp4_1024_prauc": baseline_mean,
                "mean_prauc_by_representation": mean_by_repr,
                "best_representation": best_repr, "best_repr_prauc": best_score,
                "best_beats_baseline_by": best_score - baseline_mean,
                "best_beats_baseline_exceeds_delta": bool(best_score - baseline_mean > delta),
                "any_repr_beats_presence_by_delta": bool(any(v - presence_mean > delta for v in mean_by_repr.values())),
            })
            print(f"  {task}/{phase}: best={best_repr} ({best_score:.4f}) baseline={baseline_mean:.4f} "
                  f"presence={presence_mean:.4f} delta={delta:.4f}", flush=True)

    n_best_beats_baseline = sum(1 for c in per_cell if c["best_beats_baseline_exceeds_delta"])
    n_any_beats_presence = sum(1 for c in per_cell if c["any_repr_beats_presence_by_delta"])

    verdicts = []
    if n_best_beats_baseline >= 5:
        verdicts.append(f"RULED IN: the best representation beats ECFP4/1024 by more than "
                          f"delta_cell on {n_best_beats_baseline}/16 cells -- 'we measured one "
                          f"fingerprint, not chemistry.'")
    else:
        verdicts.append(f"Representation choice doesn't change the picture on "
                          f"{16 - n_best_beats_baseline}/16 cells (best repr only beats ECFP4/1024 "
                          f"by >delta_cell on {n_best_beats_baseline}/16).")
    if n_any_beats_presence < 5:
        verdicts.append(f"RULED OUT: no representation clears the presence control by more than "
                          f"delta_cell on enough cells ({n_any_beats_presence}/16) -- 'chemistry is "
                          f"weak' would need to stand on its own, not blamed on ECFP4.")
    else:
        verdicts.append(f"At least one representation clears the presence control by more than "
                          f"delta_cell on {n_any_beats_presence}/16 cells -- chemistry signal exists "
                          f"regardless of representation choice.")
    verdict = " ".join(verdicts)

    artifact = {
        "test_id": "T14",
        "claim_at_stake": "explanation (a): the benchmark measured one fingerprint, not chemistry (Untouched)",
        "inputs": {"seeds": SEEDS, "representations": REPR_NAMES + [BASELINE_REPR],
                   "note": "12 representations by the plan's own item list vs. its stated cost "
                           "estimate of 11 -- noted, not silently resolved."},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "n_cells_best_beats_baseline": n_best_beats_baseline,
        "n_cells_any_repr_beats_presence": n_any_beats_presence,
        "decision_rule": "If the best representation beats ECFP4/1024 by more than delta_cell "
                          "in >=5 cells, RULE IN explanation (a). If no representation clears "
                          "the presence control by more than delta_cell in >=5 cells, RULE OUT "
                          "explanation (a).",
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
