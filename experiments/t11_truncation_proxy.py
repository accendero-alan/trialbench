"""T11 -- Truncation, measured with the cheap model.

newsletter-part2-test-plan.md, T11. "BERT lost fair and square" vs "BERT was
crippled by MAX_LENGTH=256". A real BERT re-run is out of budget (51 min/cell);
this uses TF-IDF as a cheap probe for how much signal lives beyond the
truncation point, on the 9 cells `clinical_embeddings` actually ran (confirmed
from the run files: mortality_rate_yn all 4 phases, outcome all 4 phases,
serious_adverse_rate_yn Phase1 only).

Per cell, 5 seeds: fit tfidf_logreg on the full document and on the document
truncated to the first 256 Bio_ClinicalBERT-tokenizer tokens (tokenize with
the actual model tokenizer, truncate, decode back to text -- not a naive
word/character cut, so the TF-IDF vocabulary sees exactly what BERT would
have seen). Text truncation is computed once per cell (keyed by NCT id, from
the union of all rows) and reused across seeds -- it doesn't depend on the
train/valid split, only tokenization, so this avoids re-tokenizing 5x per cell.

Usage: `python -m experiments.t11_truncation_proxy`
"""
from __future__ import annotations

import numpy as np

from experiments._common import Timer, git_sha, write_artifact
from src.data.features import concat_text
from src.data.loader import load_task_phase
from src.eval import metrics as M

SEEDS = [42, 7, 123, 2024, 5]
DATA_ROOT = "data"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 256
OUT_PATH = "results/experiments/t11_truncation_proxy.json"
T1_ARTIFACT = "results/experiments/t1_noise_floor.json"

# confirmed from results/extracted/trialbench/results/runs -- the 9 cells
# clinical_embeddings actually completed
BERT_CELLS = [
    ("mortality_rate_yn", "Phase1"), ("mortality_rate_yn", "Phase2"),
    ("mortality_rate_yn", "Phase3"), ("mortality_rate_yn", "Phase4"),
    ("outcome", "Phase1"), ("outcome", "Phase2"), ("outcome", "Phase3"), ("outcome", "Phase4"),
    ("serious_adverse_rate_yn", "Phase1"),
]


def _build_truncation_map(task, phase):
    """{nct_id: (full_text, truncated_text, frac_retained)} for every row in
    the cell (train_full union valid union test), tokenized once."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    td = load_task_phase(DATA_ROOT, task, phase, seed=42)
    import pandas as pd
    X_all = pd.concat([td.X_train, td.X_valid, td.X_test])
    texts = concat_text(X_all)
    ids = list(X_all.index)

    out = {}
    for nct_id, text in zip(ids, texts):
        ids_full = tok.encode(text, add_special_tokens=False)
        n_full = max(len(ids_full), 1)
        ids_trunc = ids_full[:MAX_LENGTH]
        trunc_text = tok.decode(ids_trunc, skip_special_tokens=True) if ids_trunc else ""
        out[str(nct_id)] = (text, trunc_text, len(ids_trunc) / n_full)
    return out


def _fit_score(train_texts, train_y, test_texts, test_y, task_type, num_classes, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(Xtr, train_y)
    Xte = vec.transform(test_texts)
    proba = clf.predict_proba(Xte)
    score = proba[:, 1] if task_type == "binary" else proba
    return M.compute(test_y, score, task_type, num_classes)["prauc"]


def main():
    with open(T1_ARTIFACT) as f:
        import json
        t1 = json.load(f)
    delta_cells = {(r["task"], r["phase"]): r["delta_cell"] for r in t1["delta_cell_table"]}

    per_cell = []
    with Timer() as t:
        for task, phase in BERT_CELLS:
            trunc_map = _build_truncation_map(task, phase)
            frac_retained_mean = float(np.mean([v[2] for v in trunc_map.values()]))

            full_praucs, trunc_praucs = [], []
            for seed in SEEDS:
                td = load_task_phase(DATA_ROOT, task, phase, seed=seed)
                tr_full = [trunc_map[str(i)][0] for i in td.X_train.index]
                te_full = [trunc_map[str(i)][0] for i in td.X_test.index]
                tr_trunc = [trunc_map[str(i)][1] for i in td.X_train.index]
                te_trunc = [trunc_map[str(i)][1] for i in td.X_test.index]

                full_prauc = _fit_score(tr_full, td.y_train, te_full, td.y_test,
                                         td.task_type, td.num_classes, seed)
                trunc_prauc = _fit_score(tr_trunc, td.y_train, te_trunc, td.y_test,
                                          td.task_type, td.num_classes, seed)
                full_praucs.append(full_prauc)
                trunc_praucs.append(trunc_prauc)

            mean_full, mean_trunc = float(np.mean(full_praucs)), float(np.mean(trunc_praucs))
            gap = mean_full - mean_trunc
            delta = delta_cells.get((task, phase), float("nan"))
            within_delta = bool(gap <= delta) if not np.isnan(delta) else None
            per_cell.append({
                "task": task, "phase": phase, "frac_doc_retained_at_256_tokens": frac_retained_mean,
                "full_text_prauc": mean_full, "truncated_256_prauc": mean_trunc,
                "gap": gap, "delta_cell": delta, "gap_within_delta_cell": within_delta,
            })
            print(f"  {task}/{phase}: full={mean_full:.4f} trunc={mean_trunc:.4f} gap={gap:+.4f} "
                  f"frac_retained={frac_retained_mean:.3f} delta={delta:.4f}", flush=True)

    n_within = sum(1 for c in per_cell if c["gap_within_delta_cell"])
    if n_within == len(per_cell):
        verdict = (f"CONFIRMED: truncated TF-IDF is within delta_cell of full TF-IDF on all "
                    f"{len(per_cell)}/9 cells BERT ran -- truncation cannot explain BERT's loss, "
                    f"the 9-of-9 claim is CONFIRMED with the objection pre-empted.")
    else:
        verdict = (f"SOFTENED: truncated TF-IDF loses materially on {len(per_cell) - n_within}/"
                    f"{len(per_cell)} cells -- 'BERT lost, and part of the gap is a sequence-length "
                    f"budget nobody set deliberately'.")

    artifact = {
        "test_id": "T11",
        "claim_at_stake": "BERT lost fair and square, 9 of 9 -- vs. BERT was crippled by MAX_LENGTH=256",
        "inputs": {"seeds": SEEDS, "model_name": MODEL_NAME, "max_length": MAX_LENGTH,
                   "cells": BERT_CELLS,
                   "note": "TF-IDF is a proxy that bounds where signal is located in the "
                            "document, not a measurement of BERT's own ability to use it."},
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "n_cells_gap_within_delta": n_within,
        "decision_rule": "If truncated TF-IDF is within delta_cell of full TF-IDF on all 9 "
                          "cells, truncation cannot explain BERT's loss -- CONFIRMED. If it "
                          "loses materially, SOFTENED.",
        "verdict": verdict,
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(OUT_PATH, artifact)
    print(f"\nwrote {OUT_PATH}: {n_within}/{len(per_cell)} cells within delta_cell")
    print(verdict)
    return artifact


if __name__ == "__main__":
    main()
