"""T13 -- Is the novel-scaffold result real?

newsletter-part2-test-plan.md, T13. On novel-scaffold rows (drug-id matrix
verifiably all zeros), fingerprints reportedly beat exact-SMILES lookup by
+0.220 (mortality) / +0.123 (serious AE), resting on 780 rows across 16
cells, median 40/cell. Is that separable from noise at n=40?

Requires P2 (persisted predictions) -- reads T12's persisted
`chem_morgan_fp_lightgbm` / `chem_drug_id_lightgbm` test predictions (the
same-family LightGBM fits T12 established as the valid contrast) rather than
refitting. The novel-scaffold mask itself is recomputed here (cheap,
RDKit-only, no model fitting) per (task, phase, seed) using that seed's own
train split, matching what T12 used internally.

Pools novel-scaffold (row, seed) pairs across all 4 phases x 5 seeds within
each of the 4 binary tasks (note: this counts (row, seed) pairs, not unique
physical rows -- the same physical test row contributes once per seed, each
time with that seed's own model fit's predictions, which is a legitimate way
to stabilize an n=40-per-cell estimate, not double-counting independent
evidence -- both counts are reported). Per task: paired bootstrap
(10,000 resamples) on fingerprints-minus-lookup PR-AUC, plus a label-permutation
null (1,000 permutations) for an exact p-value.

Per-cell numbers are computed for completeness but the decision rule
explicitly forbids using them as evidence (n too small per cell).

Usage: `python -m experiments.t13_novel_scaffold_significance` (after T12)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from experiments._common import Timer, git_sha, write_artifact
from src.data import mol_features as mf
from src.data.loader import load_task_phase
from src.eval.predictions import load_predictions

BINARY_TASKS = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn", "patient_dropout_rate_yn"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"
RESULTS_DIR = "results"
FAMILY = "lightgbm"
N_BOOT = 10000
N_PERM = 1000
RNG_SEED = 2
OUT_PATH = "results/experiments/t13_novel_scaffold_significance.json"


def _novel_mask(task, phase, seed):
    td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
    train_smiles = mf.smiles_lists(td.X_train)
    test_smiles = mf.smiles_lists(td.X_test)
    all_smiles = {m for l in (train_smiles + test_smiles) for m in l}
    feats = mf.featurize_molecules(sorted(all_smiles))
    _, _, _, train_scaf_sets = mf.aggregate(train_smiles, feats)
    _, _, _, test_scaf_sets = mf.aggregate(test_smiles, feats)
    train_scafs = {s for st in train_scaf_sets for s in st}
    novel = np.array([bool(st) and not (st & train_scafs) for st in test_scaf_sets])
    return novel, td.X_test.index


def _pooled_bootstrap_and_perm(y, fp_score, lookup_score, rng_seed):
    rng = np.random.default_rng(rng_seed)
    n = len(y)
    boot_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        fp_prauc = average_precision_score(y[idx], fp_score[idx])
        lk_prauc = average_precision_score(y[idx], lookup_score[idx])
        boot_diffs[i] = fp_prauc - lk_prauc
    observed = average_precision_score(y, fp_score) - average_precision_score(y, lookup_score)

    perm_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        y_perm = rng.permutation(y)
        perm_diffs[i] = (average_precision_score(y_perm, fp_score)
                          - average_precision_score(y_perm, lookup_score))
    p_value = float((np.sum(np.abs(perm_diffs) >= abs(observed)) + 1) / (N_PERM + 1))
    return observed, boot_diffs, p_value


def main():
    per_cell = []
    with Timer() as t:
        pooled_by_task = {task: {"y": [], "fp": [], "lookup": [], "n_unique_rows": set()} for task in BINARY_TASKS}
        for task in BINARY_TASKS:
            for phase in PHASES:
                for seed in SEEDS:
                    novel, test_ids = _novel_mask(task, phase, seed)
                    n_novel = int(novel.sum())
                    if n_novel == 0:
                        continue
                    fp_df = load_predictions(RESULTS_DIR, task, phase, f"chem_morgan_fp_{FAMILY}", seed, split="test")
                    lk_df = load_predictions(RESULTS_DIR, task, phase, f"chem_drug_id_{FAMILY}", seed, split="test")
                    y_cell = fp_df["y_true"].to_numpy()[novel]
                    fp_cell = fp_df["y_proba"].to_numpy()[novel]
                    lk_cell = lk_df["y_proba"].to_numpy()[novel]

                    cell_diff = (average_precision_score(y_cell, fp_cell) - average_precision_score(y_cell, lk_cell)
                                 if len(np.unique(y_cell)) > 1 else float("nan"))
                    per_cell.append({"task": task, "phase": phase, "seed": seed,
                                      "n_novel_rows": n_novel, "fp_minus_lookup_prauc": cell_diff})

                    pooled_by_task[task]["y"].append(y_cell)
                    pooled_by_task[task]["fp"].append(fp_cell)
                    pooled_by_task[task]["lookup"].append(lk_cell)
                    pooled_by_task[task]["n_unique_rows"].update(str(i) for i in np.array(test_ids)[novel])

        per_task = {}
        rng_seed = RNG_SEED
        for task in BINARY_TASKS:
            y = np.concatenate(pooled_by_task[task]["y"]) if pooled_by_task[task]["y"] else np.array([])
            fp = np.concatenate(pooled_by_task[task]["fp"]) if pooled_by_task[task]["fp"] else np.array([])
            lk = np.concatenate(pooled_by_task[task]["lookup"]) if pooled_by_task[task]["lookup"] else np.array([])
            n_pooled = len(y)
            n_unique = len(pooled_by_task[task]["n_unique_rows"])
            if n_pooled < 20 or len(np.unique(y)) < 2:
                per_task[task] = {"n_pooled_row_seed_pairs": n_pooled, "n_unique_physical_rows": n_unique,
                                   "resolvable": False}
                print(f"  {task}: n_pooled={n_pooled} -- unresolvable", flush=True)
                continue
            observed, boot_diffs, p_value = _pooled_bootstrap_and_perm(y, fp, lk, rng_seed)
            rng_seed += 1
            ci_lo, ci_hi = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
            per_task[task] = {
                "n_pooled_row_seed_pairs": n_pooled, "n_unique_physical_rows": n_unique,
                "resolvable": True, "observed_fp_minus_lookup": float(observed),
                "bootstrap_ci_lo": ci_lo, "bootstrap_ci_hi": ci_hi,
                "clears_zero": bool(ci_lo > 0), "permutation_p_value": p_value,
            }
            print(f"  {task}: n_pooled={n_pooled} (unique={n_unique}) observed={observed:+.4f} "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}] p={p_value:.4f}", flush=True)

    n_confirmed = sum(1 for v in per_task.values() if v.get("clears_zero"))
    n_resolvable = sum(1 for v in per_task.values() if v.get("resolvable"))
    if n_confirmed > 0:
        verdict = (f"CONFIRMED for {n_confirmed}/{n_resolvable} resolvable task(s) -- pooled CI "
                    f"clears zero, becomes the lead of Surprise 3 with its n stated. Per-cell "
                    f"numbers are NOT reported as evidence, per protocol.")
    else:
        verdict = (f"SOFTENED: no task's pooled CI clears zero -- 'suggestive, 780 rows, would "
                    f"need a targeted study,' not written as a result.")

    artifact = {
        "test_id": "T13",
        "claim_at_stake": "fingerprints beat exact-SMILES lookup by +0.220/+0.123 on novel-scaffold rows",
        "inputs": {"seeds": SEEDS, "family": FAMILY, "n_boot": N_BOOT, "n_perm": N_PERM,
                   "note": "pools (row, seed) pairs across all 4 phases x 5 seeds within each "
                           "task; n_pooled_row_seed_pairs and n_unique_physical_rows both "
                           "reported -- see module docstring."},
        "per_task": per_task,
        "per_cell_not_evidence": per_cell,
        "decision_rule": "If the pooled per-task CI clears zero, CONFIRMED and becomes the "
                          "lead of Surprise 3, stated with its n. If it crosses zero, SOFTENED "
                          "to 'suggestive, 780 rows, would need a targeted study' -- must not be "
                          "written as a result. Per-cell numbers are never reported as evidence.",
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
