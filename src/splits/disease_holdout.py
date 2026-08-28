"""P12 (disease-representation-test-plan.md, T26): disease-family holdout folds.

Within one (task, phase) cell's train+valid pool (TrialBench's own seed-42
split, recombined -- T26 re-splits this pool itself, so which half a trial
originally landed in doesn't matter), K rotating folds hold out whole ICD-10
3-character families: every trial whose *primary* family (the char3 rollup of
its first parsed ICD code, in the order ``src.data.features._recursive_parse_terms``
returns) falls in a fold's held-out family set is that fold's eval trial.
Multi-hot/ancestor/SapBERT representations of an eval trial's disease are
therefore built from families the fit half of that fold never saw, which is
the point (T26: "which representation still predicts when the disease family
was never in training?").

Families are assigned to folds by longest-processing-time bin-packing (sort
families by trial count descending, always add the next family to the
currently-smallest fold) -- a simple heuristic that keeps every fold near the
plan's 5-15% band without solving a real partition problem; it is not a hard
guarantee (one very common family can pin a fold above 15% on its own -- the
artifact records actual ``fold_fracs`` so that's visible, not hidden). Trials
with no ICD code at all have no family to hold out, so they never become an
eval trial (excluded from the numerator, not the denominator: every fold's
*train* side still includes them) -- reported as ``n_no_code``, not silently
dropped.

Fold assignment is written to disk once per (task, phase) and every T26 arm
reads the same file, so representations are compared on identical splits.
TrialBench's own test set is never touched by this module.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..data.features import _icd_char3, _recursive_parse_terms
from ..data.loader import load_task_phase

SPLITS_DIR = os.path.join("results", "splits", "disease_holdout")
DEFAULT_K = 8


def _primary_family(icdcode_value) -> str | None:
    codes = [c for c in _recursive_parse_terms(icdcode_value) if c.strip()]
    return _icd_char3(codes[0]) if codes else None


def _assign_families_to_folds(family_counts: dict, K: int, seed: int) -> dict:
    """Longest-processing-time bin packing: families sorted by trial count
    descending (ties broken by a seeded shuffle, not insertion/alphabetic
    order -- an alphabetic tiebreak would make every K-way split of this data
    correlated with ICD chapter order), each assigned in turn to the fold
    with the fewest trials so far. Returns ``{family: fold_idx}``."""
    rng = np.random.default_rng(seed)
    families = list(family_counts.keys())
    rng.shuffle(families)
    families.sort(key=lambda f: family_counts[f], reverse=True)
    fold_totals = [0] * K
    family_to_fold = {}
    for fam in families:
        fold = int(np.argmin(fold_totals))
        family_to_fold[fam] = fold
        fold_totals[fold] += family_counts[fam]
    return family_to_fold


def build_folds(nct_ids, families, K: int = DEFAULT_K, seed: int = 42) -> dict:
    """``nct_ids``: array-like of trial ids. ``families``: same-length
    array-like of that trial's primary family (``None``/``""`` if the trial
    has no ICD code). Returns a JSON-serializable dict recording the
    family->fold assignment and, per fold, the held-out eval trial ids."""
    nct_ids = list(nct_ids)
    families = list(families)
    if len(nct_ids) != len(families):
        raise ValueError(f"length mismatch: {len(nct_ids)} ids vs {len(families)} families")

    has_family = [f not in (None, "") for f in families]
    family_counts: dict = {}
    for f, has in zip(families, has_family):
        if has:
            family_counts[f] = family_counts.get(f, 0) + 1

    family_to_fold = _assign_families_to_folds(family_counts, K, seed)

    fold_eval_ids = [[] for _ in range(K)]
    for nid, f, has in zip(nct_ids, families, has_family):
        if has:
            fold_eval_ids[family_to_fold[f]].append(nid)

    n_total = len(nct_ids)
    n_no_code = n_total - sum(has_family)
    fold_sizes = [len(ids) for ids in fold_eval_ids]
    fold_fracs = [round(n / n_total, 4) if n_total else 0.0 for n in fold_sizes]

    return {
        "K": K, "seed": seed, "n_total_trials": n_total, "n_no_code": n_no_code,
        "n_families": len(family_counts),
        "family_to_fold": family_to_fold,
        "fold_eval_ids": fold_eval_ids,
        "fold_sizes": fold_sizes, "fold_fracs": fold_fracs,
    }


def fold_path(task: str, phase: str, splits_dir: str = SPLITS_DIR) -> str:
    return os.path.join(splits_dir, f"{task}__{phase}.json")


def build_and_save(task: str, phase: str, data_root: str = "data", K: int = DEFAULT_K,
                    seed: int = 42, splits_dir: str = SPLITS_DIR, force: bool = False) -> dict:
    """Build (or load, if already on disk and ``force`` is false) the fold
    assignment for one (task, phase) cell's train+valid pool -- TrialBench's
    own seed-42 split, recombined (which half a trial started in is
    irrelevant once T26 re-splits the whole pool into folds). Written once;
    every T26 arm calls this and gets the identical assignment back."""
    path = fold_path(task, phase, splits_dir)
    if os.path.exists(path) and not force:
        return load_folds(task, phase, splits_dir)

    td = load_task_phase(data_root, task, phase, seed=seed)
    X_pool = pd.concat([td.X_train, td.X_valid])
    icd_col = (X_pool["icdcode"] if "icdcode" in X_pool.columns
               else pd.Series([None] * len(X_pool), index=X_pool.index))
    families = [_primary_family(v) for v in icd_col.values]
    nct_ids = [str(i) for i in X_pool.index]

    artifact = build_folds(nct_ids, families, K=K, seed=seed)
    artifact["task"] = task
    artifact["phase"] = phase

    os.makedirs(splits_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(artifact, f, indent=2)
    os.replace(tmp_path, path)
    return artifact


def load_folds(task: str, phase: str, splits_dir: str = SPLITS_DIR) -> dict:
    path = fold_path(task, phase, splits_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- call build_and_save({task!r}, {phase!r}) first (P12)."
        )
    with open(path) as f:
        return json.load(f)
