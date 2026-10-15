"""Jev vs BERT vs LR -- the comparison, written before the grid runs.

Reads the test-split predictions that `src.run_benchmark` saves for four
methods on the 16 binary TrialBench cells (outcome, mortality, SAE,
dropout x Phase1-4), seed 42, from RESULTS_DIR:

- tfidf_logreg            TF-IDF + logistic regression (Part 3's winner)
- clinical_embeddings     frozen Bio_ClinicalBERT, mean-pooled, linear head (Part 3's arm)
- clinical_bert_finetuned Bio_ClinicalBERT fine-tuned per cell (new)
- jev_zeroshot            Jev, zero-shot, no training data (new)

Decision rule, fixed before the run (also on newsletter-pipeline TASK-006):
- Metric: PR-AUC (the benchmark's headline), AUROC secondary. Test split,
  rows inner-joined on nct_id so every method is scored on the same rows.
- Primary statistic: the macro-average over the 16 cells of the per-cell
  difference (Jev - arm), with a 95% CI from a stratified bootstrap
  (each cell's trials resampled independently, 1,000 draws, seed 42).
  Computed against each of the three trained arms.
- Readings, per arm:
    CI entirely below 0          -> JEV_BELOW
    CI inside [-0.03, +0.03]     -> EQUIVALENT (margin fixed in advance)
    CI entirely above 0          -> JEV_ABOVE
    otherwise                    -> UNRESOLVED
- Alan's question, "is Jev just BERT with more data?", is read off the
  fine-tuned arm: EQUIVALENT there means behaviourally indistinguishable
  from a fine-tuned clinical BERT on this benchmark. That says nothing
  about Jev's architecture, which is undisclosed.
- Part 3's prediction, "the ceiling is the data", is read off TF-IDF,
  chosen in advance as the reference because it won every Part 3 cell.
  JEV_ABOVE against TF-IDF refutes it for Jev. Otherwise it stands.
- Reported, not decisive: per-cell deltas with CIs and win/loss counts;
  AUROC versions of everything; Jev parse-failure rate and spend.
- Reproduction check before reading anything: frozen BERT and TF-IDF on
  the 9 cells bert_vs_tfidf.md published (PR-AUC). A gap above 0.02 in
  any cell is flagged, and the run is explained before it is interpreted.
- Known asymmetries, stated rather than corrected: Jev is zero-shot while
  the other three train on each cell's training split; Jev reads the L7
  render (structured fields + text) while the text arms read concat_text
  over TEXT_COLS.

Usage:
    python -m experiments.jev_benchmark_compare [--results-dir results_jev]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.predictions import load_predictions

TASKS = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn", "patient_dropout_rate_yn"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
JEV = "jev_zeroshot"
ARMS = ["tfidf_logreg", "clinical_embeddings", "clinical_bert_finetuned"]
SEED = 42
N_BOOT = 1000
MARGIN = 0.03
REPRO_TOL = 0.02
OUT_PATH = "results/experiments/jev_benchmark_compare.json"
METRICS = {"prauc": average_precision_score, "auroc": roc_auc_score}

# bert_vs_tfidf.md, PR-AUC, (tfidf_logreg, clinical_embeddings)
PART3 = {
    ("serious_adverse_rate_yn", "Phase1"): (0.909, 0.824), ("mortality_rate_yn", "Phase1"): (0.857, 0.752),
    ("mortality_rate_yn", "Phase2"): (0.853, 0.773), ("mortality_rate_yn", "Phase3"): (0.852, 0.751),
    ("outcome", "Phase1"): (0.662, 0.520), ("outcome", "Phase2"): (0.598, 0.497),
    ("outcome", "Phase3"): (0.824, 0.712), ("outcome", "Phase4"): (0.621, 0.504),
    ("mortality_rate_yn", "Phase4"): (0.504, 0.294),
}


def _cell(results_dir, task, phase):
    frames = {}
    for m in [JEV] + ARMS:
        df = load_predictions(results_dir, task, phase, m, SEED, split="test")
        frames[m] = df.set_index("nct_id")[["y_true", "y_proba"]]
    ids = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    y = frames[JEV].loc[ids, "y_true"].to_numpy().astype(int)
    return np.array(ids), y, {m: f.loc[ids, "y_proba"].to_numpy(dtype=float) for m, f in frames.items()}


def _reading(lo, hi):
    if hi < 0:
        return "JEV_BELOW"
    if lo > 0:
        return "JEV_ABOVE"
    if -MARGIN <= lo and hi <= MARGIN:
        return "EQUIVALENT"
    return "UNRESOLVED"


def run(results_dir="results_jev"):
    rng = np.random.default_rng(SEED)
    cells, per_cell = [], {}
    for task in TASKS:
        for phase in PHASES:
            ids, y, p = _cell(results_dir, task, phase)
            cells.append((task, phase, ids, y, p))
            per_cell[f"{task}/{phase}"] = {
                "n": int(len(y)), "positive_rate": float(y.mean()),
                **{metric: {m: float(fn(y, p[m])) for m in p} for metric, fn in METRICS.items()},
            }

    # one set of draws per cell, shared by every metric and arm, so all contrasts see the same resamples
    draws = []
    for _, _, ids, y, _ in cells:
        cell_draws = []
        while len(cell_draws) < N_BOOT:
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                cell_draws.append(idx)
        draws.append(cell_draws)

    summary = {}
    for metric, fn in METRICS.items():
        summary[metric] = {}
        for arm in ARMS:
            point_deltas = [fn(y, p[JEV]) - fn(y, p[arm]) for _, _, _, y, p in cells]
            boot = np.mean([[fn(y[d], p[JEV][d]) - fn(y[d], p[arm][d]) for d in cell_draws[:N_BOOT]]
                            for (_, _, _, y, p), cell_draws in zip(cells, draws)], axis=0)
            lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            per = []
            for (task, phase, _, y, p), cell_draws in zip(cells, draws):
                b = [fn(y[d], p[JEV][d]) - fn(y[d], p[arm][d]) for d in cell_draws]
                per.append({"cell": f"{task}/{phase}", "delta": float(fn(y, p[JEV]) - fn(y, p[arm])),
                            "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))})
            summary[metric][arm] = {
                "macro_mean_delta": float(np.mean(point_deltas)), "ci_95": [lo, hi],
                "reading": _reading(lo, hi),
                "cells_jev_above": sum(c["lo"] > 0 for c in per), "cells_jev_below": sum(c["hi"] < 0 for c in per),
                "per_cell": per,
            }

    repro = {f"{t}/{ph}": {"part3_tfidf": tf, "now_tfidf": per_cell[f"{t}/{ph}"]["prauc"]["tfidf_logreg"],
                           "part3_frozen": fr, "now_frozen": per_cell[f"{t}/{ph}"]["prauc"]["clinical_embeddings"]}
             for (t, ph), (tf, fr) in PART3.items() if f"{t}/{ph}" in per_cell}
    for r in repro.values():
        r["flag"] = abs(r["now_tfidf"] - r["part3_tfidf"]) > REPRO_TOL or abs(r["now_frozen"] - r["part3_frozen"]) > REPRO_TOL

    metas = [json.loads(line) for f in glob.glob(os.path.join(results_dir, "jev_meta", "*.jsonl"))
             for line in open(f, encoding="utf-8")]
    jev_meter = {"predict_calls": len(metas), "rows": sum(m["n_rows"] for m in metas),
                 "parse_failures": sum(len(m["parse_failures"]) for m in metas),
                 "retries": sum(m["retries"] for m in metas),
                 "input_tokens": sum(m["input_tokens"] for m in metas),
                 "dollars_realized": round(sum(m["dollars_realized"] for m in metas), 4)}

    artifact = {
        "test_id": "jev_benchmark_compare", "results_dir": results_dir, "seed": SEED, "n_boot": N_BOOT,
        "margin": MARGIN, "decision_rule": __doc__.split("Decision rule, fixed before the run")[1].split("Usage:")[0].strip(),
        "answers": {
            "bert_with_more_data": summary["prauc"]["clinical_bert_finetuned"]["reading"],
            "data_is_the_ceiling_refuted": summary["prauc"]["tfidf_logreg"]["reading"] == "JEV_ABOVE",
        },
        "summary": summary, "per_cell": per_cell, "reproduction_check": repro,
        "reproduction_flags": [k for k, r in repro.items() if r["flag"]], "jev_meter": jev_meter,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_jev")
    art = run(ap.parse_args().results_dir)
    print(json.dumps({k: art[k] for k in ["answers", "reproduction_flags", "jev_meter"]}, indent=1))
    for metric in METRICS:
        for arm, s in art["summary"][metric].items():
            print(f"{metric:5s} Jev-{arm:24s} {s['macro_mean_delta']:+.3f} "
                  f"[{s['ci_95'][0]:+.3f}, {s['ci_95'][1]:+.3f}] {s['reading']:11s} "
                  f"cells above/below {s['cells_jev_above']}/{s['cells_jev_below']}")
