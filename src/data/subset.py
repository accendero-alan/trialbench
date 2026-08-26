"""P13.7 (wave2-start-plan.md): generate the fixed test-set sample T28 needs
-- "min(full test, 1,000 trials) per cell, stratified by label, identical
rows across all arms and all models" -- once per (task, phase), with a
recorded seed, as a plain NCT-id list ``load_task_phase``'s
``test_subset_file`` consumes.

Usage:
    python -m src.data.subset --data-root data --task mortality_rate_yn \\
        --phase Phase1 --seed 42 --out results/subsets/mortality_rate_yn_Phase1.txt
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

from .loader import TASKS, _pick_label, _read_xy

DEFAULT_MAX_ROWS = 1000


def generate_test_subset(data_root: str, task: str, phase: str, seed: int = 42,
                         max_rows: int = DEFAULT_MAX_ROWS) -> list:
    """The full test split for (task, phase), stratified-sampled down to
    ``max_rows`` by label (or kept whole if already smaller), in a
    deterministic order given by ``seed``. Returns the NCT id list; callers
    write it to disk (see ``main`` below)."""
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}'. Known: {sorted(TASKS)}")
    folder, label_candidates, task_type = TASKS[task]
    folder_path = os.path.join(data_root, folder, phase)
    Xte, yte_df = _read_xy(folder_path, "test")
    y_raw = _pick_label(yte_df, label_candidates)

    if len(Xte) <= max_rows:
        ids = list(Xte.index.astype(str))
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)  # deterministic order even when nothing is dropped
        return ids

    rng = np.random.default_rng(seed)
    labels = y_raw.astype(str).values
    ids = np.asarray(Xte.index.astype(str))
    classes, counts = np.unique(labels, return_counts=True)
    # Proportional-to-class-size allocation, largest remainder to hit max_rows exactly.
    raw_alloc = counts / counts.sum() * max_rows
    alloc = np.floor(raw_alloc).astype(int)
    shortfall = max_rows - alloc.sum()
    remainders = raw_alloc - alloc
    for i in np.argsort(-remainders)[:shortfall]:
        alloc[i] += 1

    chosen = []
    for cls, n in zip(classes, alloc):
        cls_ids = ids[labels == cls]
        n = min(n, len(cls_ids))
        chosen.extend(rng.choice(cls_ids, size=n, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def write_subset(ids: list, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write("\n".join(ids) + "\n")
    os.replace(tmp_path, out_path)
    with open(out_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--phase", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = generate_test_subset(args.data_root, args.task, args.phase, args.seed, args.max_rows)
    sha = write_subset(ids, args.out)
    print(f"wrote {len(ids)} NCT ids to {args.out} (sha256={sha})")


if __name__ == "__main__":
    main()
