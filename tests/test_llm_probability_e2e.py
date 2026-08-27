"""P13.1/P13.5/P13.10 acceptance test (wave2-start-plan.md), run against a
fake injected Bedrock client rather than a real AWS account. ``boto3`` isn't
installed in this environment (the Bedrock harness's only dependency,
imported lazily -- see ``src/bedrock/client.py``, CLAUDE.md golden rule 5),
and even where it is, this test shouldn't need live credentials or spend
money to check the CLI/config/resume/pre-registration wiring, which is what
it's actually verifying -- not model quality.

Runs ``run_cell``/``run_benchmark.main`` in-process (unlike the predecessor
llama-server version of this test, which shelled out to a real subprocess
against a fake local HTTP server) so a fake object can be injected via
``cfg["llm_boto_client"]`` -- ``BedrockClient`` never imports ``boto3`` at
all when a client is injected.

Covers:
  - a normal run against a pre-registered cell/model produces predictions
    and a run record with a real (non-zero, priced-off-configs/
    bedrock_prices.yaml) dollar figure, not ``llm_backend.py``'s old
    hardcoded ``0.00``;
  - P13.10: an unregistered model or cell aborts with no call made;
  - B3 (carried over from Wave 1, now covering llm_service_tier too):
    resuming the same results dir with a different --llm-arm aborts on the
    first cell rather than silently mixing arms.

Run:  python tests/test_llm_probability_e2e.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import methods as _m  # noqa: E402,F401  (populates the registry)
from src.run_benchmark import _assert_amendment_resolves, run_cell  # noqa: E402
from tests.test_smoke import _write_task  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# amazon.nova-lite is both a pre-registered model (configs/wave2_amendment.yaml)
# and a priced entry (configs/bedrock_prices.yaml) -- picked so this test
# exercises the real files P13.10/P15 check against, not a test-only fixture
# that could drift from them unnoticed.
FIXTURE_MODEL = "amazon.nova-lite"
TASK, PHASE, FOLDER = "serious_adverse_rate_yn", "Phase2", "serious-adverse-event-forecasting"


class _FakeConverseClient:
    """Stands in for ``boto3.client("bedrock-runtime")``: only the one
    method ``BedrockClient.converse`` actually calls."""

    def __init__(self, probability: int = 55):
        self.probability = probability
        self.calls = 0
        self.model_ids_seen = []

    def converse(self, modelId, messages, inferenceConfig):
        self.calls += 1
        self.model_ids_seen.append(modelId)
        text = json.dumps({"probability": self.probability})
        return {
            "output": {"message": {"content": [{"text": text}]}},
            "usage": {"inputTokens": 50, "outputTokens": 6},
        }


def _base_cfg(data_root, results_dir, llm_arm, llm_model, boto_client, seeds=(42,)):
    return {
        "data_root": data_root, "results_dir": results_dir,
        "max_train_rows": None, "max_test_rows": 20, "n_jobs": -1,
        "llm_arm": llm_arm, "llm_model": llm_model, "llm_temperature": 0.0,
        "primary_elicitation": "verbalized", "llm_service_tier": "sync",
        "llm_boto_client": boto_client, "seeds": list(seeds),
        "bootstrap": {"n_resamples": 20, "ci": 0.95},
    }


def test_llm_probability_smoke_and_meter():
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, FOLDER, PHASE, n_train=40, n_test=20)
        fake_client = _FakeConverseClient(probability=55)
        cfg = _base_cfg(data_root, results_dir, "L1", FIXTURE_MODEL, fake_client)

        rec = run_cell(cfg, TASK, PHASE, "llm_probability", seed=42)

        assert rec["status"] == "ok", rec
        assert rec["llm_arm"] == "L1", rec
        assert rec["llm_model"] == FIXTURE_MODEL, rec
        # run_cell scores both the valid split (~20% of train, carved by the
        # loader) and the capped 20-row test split, so the call count is
        # 20 plus however many valid rows there were -- not exactly 20.
        assert fake_client.calls >= 20, f"expected at least 20 fake Converse calls, got {fake_client.calls}"

        meter = rec["llm_meter"]
        assert meter["calls"] == fake_client.calls, meter
        # The P13.0 landmine this test would have caught: llm_backend.py's
        # meter hardcoded "dollars": 0.00 under a comment citing this plan
        # as its authority. dollars_realized/normalized here are computed
        # from configs/bedrock_prices.yaml's real amazon.nova-lite rate
        # ($0.06/$1M in, $0.24/$1M out) against 20 calls x (50 in, 6 out)
        # tokens, so they must be a small positive number, not 0 and not a
        # placeholder.
        assert meter["dollars_realized"] > 0, meter
        assert meter["dollars_normalized"] > 0, meter

        pred_files = glob.glob(os.path.join(results_dir, "predictions", TASK, PHASE, "*.parquet"))
        assert pred_files, f"no predictions written under {results_dir}"
        import pandas as pd
        df = pd.read_parquet(pred_files[0])
        assert (df["split"] == "test").sum() == 20

        # run_cell returns the record; writing it to runs/*.json is main()'s
        # job (see test_b3_resume_guard_on_arm_mismatch below for that path),
        # not run_cell's -- calling run_cell directly here is what lets this
        # test inject a fake Converse client with no CLI plumbing for it.
        print("smoke + meter OK: dollars_realized =", meter["dollars_realized"])


def test_converse_receives_resolved_model_id_not_table_key():
    """The bug this guards: --llm-model/configs/wave2_amendment.yaml use
    price-table keys (e.g. "anthropic.claude-opus-4-5"), but several ladder
    models reject that bare/key form at the Bedrock API and need the
    us./global.-prefixed inference-profile id instead (confirmed live,
    2026-08-27). llm.py must resolve before calling Converse -- if it
    doesn't, the fake client below would see the raw table key, which is
    exactly what a real Converse call would reject."""
    import yaml
    from src.bedrock.prices import DEFAULT_PRICE_TABLE_PATH
    with open(DEFAULT_PRICE_TABLE_PATH) as f:
        price_table = yaml.safe_load(f)
    table_key = "anthropic.claude-opus-4-5"
    expected_id = price_table["models"][table_key]["model_id"]
    assert expected_id != table_key, "fixture assumption broken -- table key now equals model_id?"

    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, FOLDER, PHASE, n_train=10, n_test=5)
        fake_client = _FakeConverseClient()
        cfg = _base_cfg(data_root, results_dir, "L1", table_key, fake_client)

        rec = run_cell(cfg, TASK, PHASE, "llm_probability", seed=42)

        assert rec["status"] == "ok", rec
        assert fake_client.model_ids_seen, "no calls were made"
        assert set(fake_client.model_ids_seen) == {expected_id}, (
            f"expected every Converse call to use the resolved id {expected_id!r}, "
            f"got {set(fake_client.model_ids_seen)}"
        )
        print("resolved model id OK:", table_key, "->", expected_id)


def test_p13_10_guard_unregistered_cell_and_model():
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, FOLDER, "Phase4", n_train=10, n_test=10)  # not a pre-registered phase
        fake_client = _FakeConverseClient()

        # Unregistered cell: serious_adverse_rate_yn/Phase4 isn't in
        # configs/wave2_amendment.yaml -- must abort with zero calls made.
        cfg = _base_cfg(data_root, results_dir, "L1", FIXTURE_MODEL, fake_client)
        try:
            _assert_amendment_resolves(cfg, TASK, "Phase4", "llm_probability")
            raise AssertionError("expected SystemExit for an unregistered cell")
        except SystemExit as e:
            assert "not a pre-registered cell" in str(e), e
        assert fake_client.calls == 0

        # Unregistered model: registered cell, but a model not in the amendment.
        cfg2 = _base_cfg(data_root, results_dir, "L1", "not-a-real-model", fake_client)
        try:
            _assert_amendment_resolves(cfg2, TASK, PHASE, "llm_probability")
            raise AssertionError("expected SystemExit for an unregistered model")
        except SystemExit as e:
            assert "not a pre-registered model" in str(e), e
        assert fake_client.calls == 0

        # Non-LLM methods are untouched by the guard.
        _assert_amendment_resolves(cfg, TASK, "Phase4", "logreg_l2")
        print("P13.10 guard OK: unregistered cell/model both abort with zero calls")


def _run_cli(*extra_args):
    return subprocess.run(
        [sys.executable, "-m", "src.run_benchmark", *extra_args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )


def test_b3_resume_guard_on_arm_mismatch():
    """The resume guard itself (src/run_benchmark.py's _assert_resume_matches)
    only inspects an existing on-disk run record -- it never constructs a
    method or a BedrockClient -- so this still runs as a real CLI subprocess
    against a hand-written run record, with no AWS/boto3 involvement."""
    with tempfile.TemporaryDirectory() as results_dir:
        runs_dir = os.path.join(results_dir, "runs")
        os.makedirs(runs_dir)
        stem = f"{TASK}__{PHASE}__llm_probability__seed42"
        with open(os.path.join(runs_dir, stem + ".json"), "w") as f:
            json.dump({
                "task": TASK, "phase": PHASE, "method": "llm_probability", "seed": 42,
                "status": "ok", "feature_view": "raw", "llm_arm": "L1", "llm_model": FIXTURE_MODEL,
            }, f)

        proc = _run_cli(
            "--methods", "llm_probability", "--llm-arm", "L2", "--llm-model", FIXTURE_MODEL,
            "--tasks", TASK, "--phases", PHASE, "--results-dir", results_dir,
        )
        assert proc.returncode != 0, f"expected non-zero exit on llm_arm mismatch, got 0:\n{proc.stdout}"
        assert "ABORT" in (proc.stdout + proc.stderr), proc.stdout + proc.stderr
        print("B3 guard OK: resume with a different --llm-arm aborts before any call")


if __name__ == "__main__":
    test_llm_probability_smoke_and_meter()
    test_converse_receives_resolved_model_id_not_table_key()
    test_p13_10_guard_unregistered_cell_and_model()
    test_b3_resume_guard_on_arm_mismatch()
    print("llm_probability e2e tests passed")
