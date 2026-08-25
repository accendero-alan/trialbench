"""Benchmark runner.

Iterates tasks × phases × methods × seeds. For each cell it loads the data,
builds the appropriate feature view, fits the method, scores the test set, and
writes a JSON record with bootstrapped metrics. Resumable (skips completed
cells) and robust (a missing optional dependency or an unimplemented stub is
recorded as skipped, never crashes the run).

Usage:
    python -m src.run_benchmark                       # use configs/benchmark.yaml
    python -m src.run_benchmark --methods logreg_l2 xgboost --phases Phase1
    python -m src.run_benchmark --max-test-rows 500   # quick dry run
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import yaml

from . import methods as _methods_pkg  # noqa: F401  (populates the registry)
from .data.features import CodeFeaturizer, TabularFeaturizer
from .data.loader import load_task_phase
from .eval import leaderboard as lb
from .eval import metrics as M
from .eval.predictions import save_predictions
from .methods.registry import get as get_method

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "benchmark.yaml")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_cell(cfg, task, phase, method_name, seed):
    data_root = cfg["data_root"]
    td = load_task_phase(
        data_root, task, phase, seed=seed,
        max_train_rows=cfg.get("max_train_rows"),
        max_test_rows=cfg.get("max_test_rows"),
    )
    MethodCls = get_method(method_name)
    method = MethodCls(task_type=td.task_type, num_classes=td.num_classes, seed=seed)

    # T21 (t21-code-channel-plan.md, P7): a config-level override that swaps
    # the view for any method *declaring* "tabular" -- "raw" methods (e.g.
    # tfidf_logreg) are always left untouched, and no method is renamed or
    # duplicated. Absent the override, behavior is exactly as before.
    override = cfg.get("feature_view_override")
    effective_view = override if (override and method.feature_view == "tabular") else method.feature_view

    t0 = time.time()
    valid_proba = None
    if effective_view == "tabular":
        # Only reached when `override` is unset (see effective_view above) --
        # unchanged from before P7, no extra valid-set inference call added.
        fz = TabularFeaturizer(task_type=td.task_type)
        Xtr = fz.fit_transform(td.X_train, td.y_train)
        Xva = fz.transform(td.X_valid)
        Xte = fz.transform(td.X_test)
        method.fit(Xtr, td.y_train, Xva, td.y_valid)
        proba = method.predict_proba(Xte)
        n_features = Xtr.shape[1]
    elif effective_view in ("tabular+codes", "codes"):
        cz = CodeFeaturizer(min_df=cfg.get("code_min_df", 10)).fit(td.X_train)
        Ctr, Cva, Cte = cz.transform(td.X_train), cz.transform(td.X_valid), cz.transform(td.X_test)
        if effective_view == "tabular+codes":
            fz = TabularFeaturizer(task_type=td.task_type)
            Ttr = fz.fit_transform(td.X_train, td.y_train)
            Tva, Tte = fz.transform(td.X_valid), fz.transform(td.X_test)
            Xtr, Xva, Xte = np.hstack([Ttr, Ctr]), np.hstack([Tva, Cva]), np.hstack([Tte, Cte])
        else:  # "codes" alone -- no administrative columns
            Xtr, Xva, Xte = Ctr, Cva, Cte
        method.fit(Xtr, td.y_train, Xva, td.y_valid)
        valid_proba = method.predict_proba(Xva)
        proba = method.predict_proba(Xte)
        n_features = Xtr.shape[1]
    else:  # raw
        method.fit(td.X_train, td.y_train, td.X_valid, td.y_valid)
        proba = method.predict_proba(td.X_test)
        n_features = None
    fit_secs = time.time() - t0

    # save_predictions (P2) only handles a scalar P(y=1) column; multiclass
    # cells (failure_reason) are skipped here -- their metrics still come
    # through in the JSON record's "point"/"bootstrap" fields as usual.
    if override and valid_proba is not None and td.task_type == "binary":
        save_predictions(
            cfg["results_dir"], task, phase, method_name, seed,
            valid_ids=td.X_valid.index, valid_proba=valid_proba, valid_y=td.y_valid,
            test_ids=td.X_test.index, test_proba=proba, test_y=td.y_test,
        )

    bmcfg = cfg.get("bootstrap", {})
    boot = M.bootstrap(
        td.y_test, proba, td.task_type, td.num_classes,
        n_resamples=bmcfg.get("n_resamples", 1000), ci=bmcfg.get("ci", 0.95), seed=seed,
    )
    point = M.compute(td.y_test, proba, td.task_type, td.num_classes)

    return {
        "task": task, "phase": phase, "method": method_name, "seed": seed,
        "task_type": td.task_type, "num_classes": td.num_classes,
        "n_train": int(len(td.y_train)), "n_test": int(len(td.y_test)),
        "n_features": n_features, "fit_secs": round(fit_secs, 2),
        "feature_view": effective_view,
        "headline": M.HEADLINE, "point": point, "bootstrap": boot,
        "status": "ok",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--phases", nargs="*")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--data-root")
    ap.add_argument("--results-dir")
    ap.add_argument("--max-train-rows", type=int)
    ap.add_argument("--max-test-rows", type=int)
    ap.add_argument("--n-resamples", type=int)
    ap.add_argument("--force", action="store_true", help="recompute existing cells")
    ap.add_argument("--feature-view-override", choices=["tabular+codes", "codes"],
                    help="T21: swap the view for methods declaring feature_view='tabular' "
                         "(raw methods, e.g. tfidf_logreg, are untouched); also persists "
                         "per-fit predictions (P2) since this always means a real experiment "
                         "arm, not the default leaderboard build")
    ap.add_argument("--code-min-df", type=int, help="T21: CodeFeaturizer min_df (default 10)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.tasks: cfg["tasks"] = args.tasks
    if args.phases: cfg["phases"] = args.phases
    if args.methods: cfg["methods"] = args.methods
    if args.seeds: cfg["seeds"] = args.seeds
    if args.data_root: cfg["data_root"] = args.data_root
    if args.results_dir: cfg["results_dir"] = args.results_dir
    if args.max_train_rows is not None: cfg["max_train_rows"] = args.max_train_rows
    if args.max_test_rows is not None: cfg["max_test_rows"] = args.max_test_rows
    if args.n_resamples is not None:
        cfg.setdefault("bootstrap", {})["n_resamples"] = args.n_resamples
    if args.feature_view_override: cfg["feature_view_override"] = args.feature_view_override
    if args.code_min_df is not None: cfg["code_min_df"] = args.code_min_df

    runs_dir = os.path.join(cfg["results_dir"], "runs")
    os.makedirs(runs_dir, exist_ok=True)

    grid = [(t, p, m, s) for t in cfg["tasks"] for p in cfg["phases"]
            for m in cfg["methods"] for s in cfg["seeds"]]
    print(f"{len(grid)} cells: {len(cfg['tasks'])} tasks × {len(cfg['phases'])} phases "
          f"× {len(cfg['methods'])} methods × {len(cfg['seeds'])} seeds", flush=True)

    # Rebuilding the leaderboard after every completed cell (plus once more in
    # `finally`) means results/leaderboard.md is never more than one cell stale
    # while the run is in progress -- important now that a single Tier C cell
    # (clinical_embeddings) can take several minutes, so waiting for a batch
    # of cells to finish before refreshing would leave it looking stalled. The
    # rebuild itself is cheap (~0.3s measured at the full 320-cell grid size),
    # negligible next to actual cell compute time. A crash/kill mid-grid --
    # expected on preemptible EC2 capacity -- still leaves an up-to-date
    # leaderboard, and a restarted process finds all prior cells' JSON on
    # disk and skips them (see the os.path.exists check below).
    try:
        for i, (task, phase, method_name, seed) in enumerate(grid, 1):
            stem = f"{task}__{phase}__{method_name}__seed{seed}"
            out_path = os.path.join(runs_dir, stem + ".json")
            if os.path.exists(out_path) and not args.force:
                print(f"  [{i}/{len(grid)}] skip (done): {stem}", flush=True)
                continue
            try:
                rec = run_cell(cfg, task, phase, method_name, seed)
                h = rec["bootstrap"].get(rec["headline"], {})
                print(f"  [{i}/{len(grid)}] ok   {stem}: {rec['headline']}="
                      f"{h.get('mean', float('nan')):.4f} [{h.get('lo', float('nan')):.4f},"
                      f"{h.get('hi', float('nan')):.4f}] ({rec['fit_secs']}s)", flush=True)
            except (NotImplementedError, ImportError) as e:
                rec = {"task": task, "phase": phase, "method": method_name, "seed": seed,
                       "status": "skipped", "reason": f"{type(e).__name__}: {e}"}
                print(f"  [{i}/{len(grid)}] SKIP {stem}: {rec['reason']}", flush=True)
            except FileNotFoundError as e:
                rec = {"task": task, "phase": phase, "method": method_name, "seed": seed,
                       "status": "no_data", "reason": str(e)}
                print(f"  [{i}/{len(grid)}] DATA {stem}: {e}", flush=True)
            except Exception as e:  # noqa: BLE001
                rec = {"task": task, "phase": phase, "method": method_name, "seed": seed,
                       "status": "error", "reason": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()}
                print(f"  [{i}/{len(grid)}] ERR  {stem}: {rec['reason']}", flush=True)

            # Atomic write: a kill/OOM mid-write leaves at most a stray .tmp file, never
            # a truncated .json that would look "done" on resume or crash the leaderboard.
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(rec, f, indent=2)
            os.replace(tmp_path, out_path)

            lb.build(cfg["results_dir"])
    finally:
        lb.build(cfg["results_dir"])
        print(f"\nLeaderboard written to {os.path.join(cfg['results_dir'], 'leaderboard.md')}",
              flush=True)


if __name__ == "__main__":
    main()
