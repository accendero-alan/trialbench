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
import sys
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


def _expected_view_and_granularity(cfg, method_name):
    """What ``run_cell`` would write to a fresh record's ``feature_view`` /
    ``icd_granularity`` for this method under the current config -- computed
    without actually running the cell, so the resume check below can compare
    against it before deciding to skip."""
    MethodCls = get_method(method_name)
    override = cfg.get("feature_view_override")
    effective_view = override if (override and MethodCls.feature_view == "tabular") else MethodCls.feature_view
    granularity = cfg.get("icd_granularity", "char3") if effective_view in ("tabular+codes", "codes") else None
    return effective_view, granularity


def _assert_amendment_resolves(cfg, task, phase, method_name):
    """P13.10 (wave2-start-plan.md): refuse to instantiate an LLM method
    unless the pre-registration amendment resolves and matches the
    requested run. This check sits *here*, in the runner, and runs *before*
    any billable call -- the analysis scripts (``experiments/t28_*.py``)
    run after the money is already spent, which is where revision 2 put an
    equivalent guard and why revision 3 moved it.

    Checks that (``task``, ``phase``) is one of the amendment's pre-
    registered cells, that ``--llm-model`` is one of its pre-registered
    models, and that the configured seed count matches its repeats spec for
    this cell. Does not touch the network or construct a
    :class:`~src.bedrock.client.BedrockClient` -- a mismatch here must cost
    nothing.
    """
    if method_name != "llm_probability":
        return
    path = cfg.get("wave2_amendment_file", os.path.join(ROOT, "configs", "wave2_amendment.yaml"))
    if not os.path.exists(path):
        sys.exit(
            f"ABORT: llm_probability requires a pre-registration amendment at {path} "
            f"(P13.10, wave2-start-plan.md §6) -- none found. No call was made."
        )
    with open(path) as f:
        amendment = yaml.safe_load(f)

    cells = [(c["task"], c["phase"]) for c in amendment.get("cells", [])]
    if (task, phase) not in cells:
        sys.exit(
            f"ABORT: {task}/{phase} is not a pre-registered cell in {path} (P13.10). "
            f"Pre-registered cells: {cells}. No call was made."
        )
    models = amendment.get("models", [])
    llm_model = cfg.get("llm_model")
    if llm_model not in models:
        sys.exit(
            f"ABORT: --llm-model {llm_model!r} is not a pre-registered model in {path} "
            f"(P13.10). Pre-registered models: {models}. No call was made."
        )
    default_repeats = amendment.get("default_repeats", 1)
    cell_repeats = next(
        (c.get("repeats", default_repeats) for c in amendment.get("cells", [])
         if c["task"] == task and c["phase"] == phase),
        default_repeats,
    )
    n_seeds = len(cfg.get("seeds", []))
    if n_seeds != cell_repeats:
        sys.exit(
            f"ABORT: {task}/{phase} is pre-registered for {cell_repeats} repeat(s) in {path} "
            f"(P13.10 / §6.5), but the current config resolves {n_seeds} seed(s) "
            f"({cfg.get('seeds')}). Fix --seeds to match, or amend {path} before this run, "
            f"not after seeing results. No call was made."
        )


def _assert_resume_matches(out_path, stem, expected_view, expected_granularity,
                           expected_llm_arm=None, expected_llm_model=None):
    """B3 (wave1-preflight-review.md): the skip-if-exists resume path used to
    compare nothing but the file's existence, so one mistyped
    ``--icd-granularity`` on a resumed sweep silently mixed two rungs into
    one results directory -- a corruption ``leaderboard.md`` cannot reveal
    and the T22-T24 analysis scripts never check for. Only ``status: "ok"``
    records carry ``feature_view``/``icd_granularity`` at all (skipped/
    no_data/error records don't reach the code that writes them), so those
    statuses have nothing to compare and are left alone -- resuming past a
    genuinely skipped or errored cell is unaffected.

    ``icd_granularity`` is checked only when the record actually has the key.
    ``results_codes/`` (char3, T21's original sweep) predates this field --
    its early records have no ``icd_granularity`` key at all, not a null
    value -- and that directory has only ever held char3 by construction, so
    an absent key isn't evidence of a mismatch the way a present-but-wrong
    value would be. Treating "missing" as "confirmed mismatch" would abort
    every resume of that directory, including a legitimate backfill.
    ``feature_view`` has no such legacy gap (always written) and is still
    checked unconditionally.

    P13.1 (wave2-start-plan.md): the same failure mode, one mistyped
    ``--llm-arm``/``--llm-model`` on a resumed T28 sweep, mixes two arms or
    models in one results directory -- with money attached this time.
    ``llm_arm``/``llm_model`` are checked the same way ``icd_granularity``
    is: only when the record actually carries the key (absent on every
    non-LLM run, and on LLM records predating this check).
    """
    with open(out_path) as f:
        rec = json.load(f)
    if rec.get("status") != "ok":
        return
    granularity_mismatch = "icd_granularity" in rec and rec["icd_granularity"] != expected_granularity
    llm_arm_mismatch = "llm_arm" in rec and rec["llm_arm"] != expected_llm_arm
    llm_model_mismatch = "llm_model" in rec and rec["llm_model"] != expected_llm_model
    if rec.get("feature_view") != expected_view or granularity_mismatch or llm_arm_mismatch or llm_model_mismatch:
        sys.exit(
            f"ABORT: {stem} on disk was recorded with feature_view="
            f"{rec.get('feature_view')!r}, icd_granularity={rec.get('icd_granularity')!r}, "
            f"llm_arm={rec.get('llm_arm')!r}, llm_model={rec.get('llm_model')!r}, "
            f"but the current config resolves to feature_view={expected_view!r}, "
            f"icd_granularity={expected_granularity!r}, llm_arm={expected_llm_arm!r}, "
            f"llm_model={expected_llm_model!r}. Resuming would mix granularities/views/LLM "
            f"arms/models in one results directory. Fix the config/flags, or point "
            f"--results-dir at a fresh directory, or --force this cell if the mismatch is "
            f"intentional."
        )


def run_cell(cfg, task, phase, method_name, seed):
    data_root = cfg["data_root"]
    td = load_task_phase(
        data_root, task, phase, seed=seed,
        max_train_rows=cfg.get("max_train_rows"),
        max_test_rows=cfg.get("max_test_rows"),
        test_subset_file=cfg.get("test_subset_file"),
    )
    # P13.10: refuses (no call made) unless the pre-registration amendment
    # resolves and matches this cell -- checked before MethodCls is even
    # constructed, since constructing llm_probability with a bad config is
    # still zero-cost, but the point of this guard is to never get one
    # config-typo away from a billable call it wasn't pre-registered for.
    _assert_amendment_resolves(cfg, task, phase, method_name)

    MethodCls = get_method(method_name)
    # H2 (wave1-preflight-review.md): every method that parallelizes its own
    # fit (random_forest/extra_trees/knn's n_jobs, xgboost/lightgbm's n_jobs,
    # catboost's thread_count) defaulted to -1 -- fine for one solo process,
    # but five concurrent rung sweeps each claiming the whole machine is
    # usually slower than running them in sequence. cfg["n_jobs"] (default
    # -1, unchanged solo behavior) lets a concurrent launch cap each
    # process's share; see deploy/run_codes_sweep.sh.
    # P13.1: task/llm_* land in every method's self.params (BaseMethod's
    # generic **params __init__) -- harmless for non-LLM methods, same as
    # n_jobs already was before llm_probability existed.
    method = MethodCls(task_type=td.task_type, num_classes=td.num_classes, seed=seed,
                       n_jobs=cfg.get("n_jobs", -1), task=task,
                       llm_arm=cfg.get("llm_arm"), llm_model=cfg.get("llm_model"),
                       llm_temperature=cfg.get("llm_temperature", 0.0),
                       primary_elicitation=cfg.get("primary_elicitation", "verbalized"),
                       llm_max_calls=cfg.get("llm_max_calls"),
                       results_dir=cfg["results_dir"],
                       llm_service_tier=cfg.get("llm_service_tier", "sync"),
                       llm_router_arn=cfg.get("llm_router_arn"),
                       llm_region=cfg.get("llm_region", "us-east-1"),
                       llm_boto_client=cfg.get("llm_boto_client"))

    # T21 (t21-code-channel-plan.md, P7): a config-level override that swaps
    # the view for any method *declaring* "tabular" -- "raw" methods (e.g.
    # tfidf_logreg) are always left untouched, and no method is renamed or
    # duplicated. Absent the override, behavior is exactly as before.
    override = cfg.get("feature_view_override")
    effective_view = override if (override and method.feature_view == "tabular") else method.feature_view

    t0 = time.time()
    valid_proba = None
    if effective_view == "tabular":
        fz = TabularFeaturizer(task_type=td.task_type)
        Xtr = fz.fit_transform(td.X_train, td.y_train)
        Xva = fz.transform(td.X_valid)
        Xte = fz.transform(td.X_test)
        method.fit(Xtr, td.y_train, Xva, td.y_valid)
        valid_proba = method.predict_proba(Xva)
        proba = method.predict_proba(Xte)
        n_features = Xtr.shape[1]
    elif effective_view in ("tabular+codes", "codes"):
        cz = CodeFeaturizer(min_df=cfg.get("code_min_df", 10),
                            granularity=cfg.get("icd_granularity", "char3")).fit(td.X_train)
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
        valid_proba = method.predict_proba(td.X_valid)
        proba = method.predict_proba(td.X_test)
        n_features = None
    fit_secs = time.time() - t0

    # save_predictions (P2) only handles a scalar P(y=1) column; multiclass
    # cells (failure_reason) are skipped here -- their metrics still come
    # through in the JSON record's "point"/"bootstrap" fields as usual.
    # Unconditional as of the disease-representation campaign (T22-T24's
    # pooled paired bootstrap needs predictions from the T1/tfidf_logreg
    # baseline too, not just override runs) -- was gated behind `if
    # override`, which is why T1's original runs have no predictions/ and
    # needed a backfill re-run.
    if valid_proba is not None and td.task_type == "binary":
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

    rec = {
        "task": task, "phase": phase, "method": method_name, "seed": seed,
        "task_type": td.task_type, "num_classes": td.num_classes,
        "n_train": int(len(td.y_train)), "n_test": int(len(td.y_test)),
        "n_features": n_features, "fit_secs": round(fit_secs, 2),
        "feature_view": effective_view,
        "icd_granularity": cfg.get("icd_granularity", "char3") if effective_view in ("tabular+codes", "codes") else None,
        "headline": M.HEADLINE, "point": point, "bootstrap": boot,
        "status": "ok",
    }
    if method_name == "llm_probability":
        # P13.1/P13.5: written unconditionally for this method (None for
        # everyone else) so the B3 guard above has something to compare on
        # resume, and so a run record is self-describing about which arm/
        # model/sample produced it without needing the sweep's own config.
        rec["llm_arm"] = cfg.get("llm_arm")
        rec["llm_model"] = cfg.get("llm_model")
        rec["llm_service_tier"] = cfg.get("llm_service_tier", "sync")
        rec["llm_router_arn"] = cfg.get("llm_router_arn")
        rec["primary_elicitation"] = cfg.get("primary_elicitation", "verbalized")
        rec["test_subset_file"] = cfg.get("test_subset_file")
        rec["llm_meter"] = method.llm_meter_summary()
        rec["llm_refusal_rate"] = round(method.llm_refusal_rate, 4)
        rec["llm_parse_failure_rate"] = round(method.llm_parse_failure_rate, 4)
        if method.llm_parse_failure_rate > 0.02:
            print(f"  WARNING: {task}/{phase}/seed{seed} LLM parse failure rate "
                  f"{method.llm_parse_failure_rate:.1%} exceeds 2% (P13.3) -- likely a harness bug, "
                  f"not model behavior.", flush=True)
    return rec


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
    ap.add_argument("--icd-granularity",
                    choices=["char3", "full", "chapter", "block", "ccsr", "ancestors", "stack"],
                    help="P9/T23/T24: ICD granularity rung for the code views (default 'char3', "
                         "i.e. T21's original encoding).")
    ap.add_argument("--n-jobs", type=int,
                    help="H2: per-fit parallelism cap passed to every method that claims "
                         "multiple cores (random_forest/extra_trees/knn's n_jobs, "
                         "xgboost/lightgbm's n_jobs, catboost's thread_count). Default -1 "
                         "(claim the whole machine) -- fine for one solo sweep, but cap this "
                         "when running several sweeps concurrently so they don't oversubscribe "
                         "the same cores; see deploy/run_codes_sweep.sh.")
    ap.add_argument("--llm-arm", choices=["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"],
                    help="P13.1: disease-representation arm for llm_probability (src/data/serialize.py).")
    ap.add_argument("--llm-model",
                    help="P13.1: Bedrock model id or inference-profile ARN, passed through to "
                         "the Converse client and recorded in the run for the B3/P13.10 guards "
                         "and provenance -- not validated against Bedrock here (llm_probability's "
                         "predict_proba() fails loudly on the first call if the id is wrong).")
    ap.add_argument("--llm-temperature", type=float, help="P13.3: default 0.0 (deterministic).")
    ap.add_argument("--llm-primary-elicitation", choices=["verbalized"],
                    help="§6.3 amendment: which probability elicitation path is primary. "
                         "'verbalized' is the only implemented path (Bedrock exposes no logprob "
                         "path for Anthropic/Nova; DeepSeek/Llama 4 are unchecked pending W1.8) -- "
                         "pre-register before the first T28a call regardless.")
    ap.add_argument("--llm-max-calls", type=int,
                    help="P13.1: hard cap: predict_proba raises rather than silently truncating "
                         "if the sample exceeds this. Use --test-subset-file to size the sample "
                         "instead of relying on this to cut it down.")
    ap.add_argument("--llm-service-tier", choices=["sync", "batch"],
                    help="P13.4/P13.5/P13.8: which Bedrock service tier serves this cell's calls -- "
                         "part of the response-cache key and the meter's realized-cost basis. "
                         "Default 'sync'; batch is 50%% off and the default for ladder arms once "
                         "P13.8's batch runner is wired into a real submit/poll/reassemble loop.")
    ap.add_argument("--llm-router-arn",
                    help="P13.9/T31: route calls through this prompt-router ARN instead of calling "
                         "--llm-model directly. Synchronous only; forces --llm-service-tier sync.")
    ap.add_argument("--llm-region", help="P13.1: bedrock-runtime region. Default us-east-1.")
    ap.add_argument("--test-subset-file",
                    help="P13.7: fixed NCT-id sample (src/data/subset.py) applied to the test "
                         "split in file order, instead of --max-test-rows's head-n truncation. "
                         "Required for T28/T29 cells so arms/models pair on identical rows.")
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
    if args.n_jobs is not None: cfg["n_jobs"] = args.n_jobs
    if args.n_resamples is not None:
        cfg.setdefault("bootstrap", {})["n_resamples"] = args.n_resamples
    if args.feature_view_override: cfg["feature_view_override"] = args.feature_view_override
    if args.code_min_df is not None: cfg["code_min_df"] = args.code_min_df
    if args.icd_granularity: cfg["icd_granularity"] = args.icd_granularity
    if args.llm_arm: cfg["llm_arm"] = args.llm_arm
    if args.llm_model: cfg["llm_model"] = args.llm_model
    if args.llm_temperature is not None: cfg["llm_temperature"] = args.llm_temperature
    if args.llm_primary_elicitation: cfg["primary_elicitation"] = args.llm_primary_elicitation
    if args.llm_max_calls is not None: cfg["llm_max_calls"] = args.llm_max_calls
    if args.llm_service_tier: cfg["llm_service_tier"] = args.llm_service_tier
    if args.llm_router_arn:
        cfg["llm_router_arn"] = args.llm_router_arn
        cfg["llm_service_tier"] = "sync"  # P13.9: routers cannot be batched
    if args.llm_region: cfg["llm_region"] = args.llm_region
    if args.test_subset_file: cfg["test_subset_file"] = args.test_subset_file

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
                expected_view, expected_granularity = _expected_view_and_granularity(cfg, method_name)
                _assert_resume_matches(out_path, stem, expected_view, expected_granularity,
                                       expected_llm_arm=cfg.get("llm_arm"), expected_llm_model=cfg.get("llm_model"))
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
