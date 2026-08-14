"""Does the ``smiless`` column carry *chemical* predictive signal?

A local, CPU-only probe (RDKit only — no torch, no EC2 round trip) that answers
three separate questions the benchmark cannot currently distinguish:

  1. **Is it chemistry, or just "this is a small-molecule drug trial"?**
     Only ~44-57% of trials have any SMILES at all, and that presence is
     correlated with trial type. So every molecule block is compared against a
     `presence` control that sees *only* [has_molecule, n_molecules].
  2. **Is it chemistry, or drug memorization?** The same drug recurs across many
     trials. A `drug_id` control multi-hots exact molecule identity (train
     vocabulary only). If fingerprints don't beat drug identity, the model is
     recognising drugs, not reading structure.
  3. **Does it add anything over the tabular view?** Fusion blocks concatenate
     the repo's own `TabularFeaturizer` output with the molecule blocks.

Plus a **scaffold-novelty diagnostic**: PR-AUC restricted to test trials whose
Bemis-Murcko scaffolds are *all* unseen in train — the sharpest available test of
chemically generalizing signal (small n; diagnostic only, not a leaderboard row).

Leakage rules follow CLAUDE.md: every vocabulary / imputer / scaler is fit on
**train** only. Model selection uses validation; test is scored once per block.

Usage:
    python -m experiments.smiles_signal_probe --task serious_adverse_rate_yn --phase Phase3
    python -m experiments.smiles_signal_probe --all-binary --phases Phase2 Phase3
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np

# LightGBM's sklearn wrapper invents feature names, then sklearn complains that
# the numpy arrays we predict on don't have them. Harmless, and very loud.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

from src.data import mol_features as mf
from src.data.features import TabularFeaturizer
from src.data.loader import load_task_phase
from src.eval.metrics import binary_metrics, bootstrap

BINARY_TASKS = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn",
                "patient_dropout_rate_yn"]


# --------------------------------------------------------------------------
# models — one family per block shape, fit on train, tuned on validation
# --------------------------------------------------------------------------
def _fit_gbm(Xtr, ytr, Xva, yva, seed):
    """EXACTLY the benchmark's registered ``lightgbm`` config (src/methods/gbm.py).

    Deliberately identical — including no early stopping — so the
    ``tabular (reference)`` block reproduces the leaderboard's own LightGBM number
    and the fusion deltas measure the molecule blocks rather than a hyperparameter
    difference. An earlier version of this probe early-stopped on validation
    PR-AUC, which collapsed the tabular baseline in 4 of 16 cells and made
    fusion look far better than it is.
    """
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                         n_jobs=-1, random_state=seed, verbose=-1)
    clf.fit(Xtr, ytr)
    return clf


def _fit_sparse_logreg(Xtr, ytr, Xva, yva, seed):
    """L2 logistic regression over binary/sparse blocks (fingerprints, vocabs).

    C is selected on validation PR-AUC — the standard ECFP baseline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    best, best_ap = None, -np.inf
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, penalty="l2", solver="liblinear",
                                 class_weight="balanced", max_iter=2000,
                                 random_state=seed)
        clf.fit(Xtr, ytr)
        ap = average_precision_score(yva, clf.predict_proba(Xva)[:, 1])
        if ap > best_ap:
            best, best_ap = clf, ap
    return best


def _impute_scale(desc_tr, desc_other_list):
    """Train-median impute + train standardize; returns (tr, [others...])."""
    med = np.nanmedian(desc_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def _fill(M):
        out = np.where(np.isfinite(M), M, med)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    tr = _fill(desc_tr)
    mu, sd = tr.mean(axis=0), tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (tr - mu) / sd, [((_fill(M)) - mu) / sd for M in desc_other_list]


# --------------------------------------------------------------------------
def run_cell(data_root: str, task: str, phase: str, seed: int,
             n_boot: int, top_k: int) -> dict:
    td = load_task_phase(data_root, task, phase, seed=seed)
    if td.task_type != "binary":
        raise SystemExit(f"{task} is {td.task_type}; this probe covers binary tasks only.")

    # ---- molecule featurization (label-free, cached across cells) ----------
    lists = {s: mf.smiles_lists(X) for s, X in
             (("train", td.X_train), ("valid", td.X_valid), ("test", td.X_test))}
    all_smiles = {m for ls in lists.values() for l in ls for m in l}
    t0 = time.time()
    feats = mf.featurize_molecules(sorted(all_smiles))
    mol_secs = time.time() - t0

    agg = {s: mf.aggregate(lists[s], feats) for s in lists}
    pres = {s: agg[s][0] for s in agg}
    desc_raw = {s: agg[s][1] for s in agg}
    fp = {s: agg[s][2].astype(float) for s in agg}
    scaf = {s: agg[s][3] for s in agg}

    desc_tr, (desc_va, desc_te) = _impute_scale(
        desc_raw["train"], [desc_raw["valid"], desc_raw["test"]])
    desc = {"train": desc_tr, "valid": desc_va, "test": desc_te}

    # train-only vocabularies
    drug_vocab = sorted({m for l in lists["train"] for m in l if m in feats})
    scaf_vocab = sorted({s for st in scaf["train"] for s in st})
    drug = {s: mf.vocab_matrix([[m for m in l if m in feats] for l in lists[s]], drug_vocab)
            for s in lists}
    scafM = {s: mf.vocab_matrix(scaf[s], scaf_vocab) for s in scaf}

    # ---- tabular reference view (repo featurizer, fit on train) -----------
    tf = TabularFeaturizer(task_type=td.task_type).fit(td.X_train, td.y_train)
    tab = {"train": tf.transform(td.X_train), "valid": tf.transform(td.X_valid),
           "test": tf.transform(td.X_test)}

    # ---- context: is molecule presence itself informative? ----------------
    has_te = pres["test"][:, 0] > 0
    base = {
        "n_train": int(len(td.y_train)), "n_test": int(len(td.y_test)),
        "test_pos_rate": float(td.y_test.mean()),
        "test_has_mol_frac": float(has_te.mean()),
        "pos_rate_with_mol": float(td.y_test[has_te].mean()) if has_te.any() else float("nan"),
        "pos_rate_without_mol": float(td.y_test[~has_te].mean()) if (~has_te).any() else float("nan"),
        "unique_smiles_parsed": len(feats),
        "unique_smiles_seen": len(all_smiles),
        "unique_scaffolds_train": len(scaf_vocab),
        "mol_featurize_secs": round(mol_secs, 2),
    }

    def block(name, mats, kind):
        return {"name": name, "mats": mats, "kind": kind}

    def cat(*mats_by_split):
        return {s: np.hstack([m[s] for m in mats_by_split]) for s in ("train", "valid", "test")}

    blocks = [
        block("presence (control)", pres, "gbm"),
        block("descriptors", cat(pres, desc), "gbm"),
        block("morgan_fp", cat(pres, fp), "sparse"),
        block("drug_id (control)", cat(pres, drug), "sparse"),
        block("scaffold", cat(pres, scafM), "sparse"),
        block("tabular (reference)", tab, "gbm"),
        block("tabular+descriptors", cat(tab, desc), "gbm"),
        block("tabular+morgan_fp", cat(tab, fp), "gbm"),
    ]

    # scaffold-novelty subset: test trials with molecules whose scaffolds are
    # ALL unseen in train (drug memorization cannot help here)
    train_scafs = set(scaf_vocab)
    novel = np.array([bool(st) and not (st & train_scafs) for st in scaf["test"]])

    from sklearn.metrics import average_precision_score
    results = []
    for b in blocks:
        M = b["mats"]
        fit = _fit_gbm if b["kind"] == "gbm" else _fit_sparse_logreg
        t0 = time.time()
        clf = fit(M["train"], td.y_train, M["valid"], td.y_valid, seed)
        secs = time.time() - t0
        p_va = clf.predict_proba(M["valid"])[:, 1]
        p_te = clf.predict_proba(M["test"])[:, 1]

        m = binary_metrics(td.y_test, p_te)
        bs = bootstrap(td.y_test, p_te, "binary", 2, n_resamples=n_boot, seed=seed)
        rec = {
            "block": b["name"], "n_features": int(M["train"].shape[1]),
            "valid_prauc": float(average_precision_score(td.y_valid, p_va)),
            "test_prauc": m["prauc"],
            "test_prauc_lo": bs["prauc"]["lo"], "test_prauc_hi": bs["prauc"]["hi"],
            "test_auroc": m["auroc"], "fit_secs": round(secs, 2),
        }
        # molecule-bearing test rows only (apples-to-apples where chemistry exists)
        if has_te.sum() > 20 and len(np.unique(td.y_test[has_te])) > 1:
            rec["test_prauc_hasmol"] = float(
                average_precision_score(td.y_test[has_te], p_te[has_te]))
        if novel.sum() > 20 and len(np.unique(td.y_test[novel])) > 1:
            rec["test_prauc_novel_scaffold"] = float(
                average_precision_score(td.y_test[novel], p_te[novel]))
        results.append(rec)

        if b["name"] == "descriptors" and top_k:
            imp = getattr(clf, "feature_importances_", None)
            if imp is not None:
                names = ["has_molecule", "n_molecules"] + mf.DESCRIPTOR_NAMES
                order = np.argsort(imp)[::-1][:top_k]
                rec["top_features"] = [(names[i], int(imp[i])) for i in order]

    return {
        "task": task, "phase": phase, "seed": seed,
        "base": base,
        "novel_scaffold_test_rows": int(novel.sum()),
        "results": results,
    }


# --------------------------------------------------------------------------
def _print_cell(out: dict) -> None:
    b = out["base"]
    print(f"\n{'='*82}\n{out['task']} / {out['phase']}  (seed {out['seed']})\n{'='*82}")
    print(f"  train={b['n_train']}  test={b['n_test']}  test pos-rate={b['test_pos_rate']:.3f}")
    print(f"  test rows with a parseable molecule: {b['test_has_mol_frac']:.1%}")
    print(f"  pos-rate  with molecule {b['pos_rate_with_mol']:.3f}  |  "
          f"without {b['pos_rate_without_mol']:.3f}   <- the presence confound")
    print(f"  unique SMILES {b['unique_smiles_parsed']}/{b['unique_smiles_seen']} parsed, "
          f"{b['unique_scaffolds_train']} train scaffolds, "
          f"featurized in {b['mol_featurize_secs']}s")
    print(f"  novel-scaffold test rows: {out['novel_scaffold_test_rows']}")

    rows = out["results"]
    ref = next((r for r in rows if r["block"].startswith("tabular (")), None)
    ctl = next((r for r in rows if r["block"].startswith("presence")), None)
    hdr = (f"\n  {'block':22s} {'nfeat':>6s} {'val PR':>7s} {'test PR-AUC (95% CI)':>24s} "
           f"{'Δ vs ctl':>9s} {'Δ vs tab':>9s} {'PR hasmol':>10s} {'PR novel':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for r in rows:
        d_ctl = r["test_prauc"] - ctl["test_prauc"] if ctl else float("nan")
        d_tab = r["test_prauc"] - ref["test_prauc"] if ref else float("nan")
        hm = r.get("test_prauc_hasmol", float("nan"))
        nv = r.get("test_prauc_novel_scaffold", float("nan"))
        print(f"  {r['block']:22s} {r['n_features']:6d} {r['valid_prauc']:7.3f} "
              f"{r['test_prauc']:8.3f} [{r['test_prauc_lo']:.3f},{r['test_prauc_hi']:.3f}] "
              f"{d_ctl:+9.3f} {d_tab:+9.3f} {hm:10.3f} {nv:9.3f}")
    tf = next((r for r in rows if r.get("top_features")), None)
    if tf:
        print("\n  top descriptor-block features (LightGBM split gain):")
        for name, imp in tf["top_features"]:
            print(f"    {imp:6d}  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="serious_adverse_rate_yn")
    ap.add_argument("--phase", default="Phase3")
    ap.add_argument("--all-binary", action="store_true",
                    help="run every binary task at the given --phases")
    ap.add_argument("--phases", nargs="*", default=None)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=None, help="write all cell records to this JSON")
    args = ap.parse_args()

    tasks = BINARY_TASKS if args.all_binary else [args.task]
    phases = args.phases or [args.phase]

    all_out = []
    for task in tasks:
        for phase in phases:
            out = run_cell(args.data_root, task, phase, args.seed, args.n_boot, args.top_k)
            _print_cell(out)
            all_out.append(out)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(all_out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
