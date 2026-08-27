"""P13.8 acceptance test (wave2-start-plan.md): batch submission is actually
wired to the elicitation path, and the cost meter reports the tier each call
was *actually* billed at -- the bug found 2026-08-27 (P13.8's status note):
`--llm-service-tier batch` was a label nothing enforced, every call ran
real-time synchronous, and the meter applied the batch discount anyway.

Runs against fake injected S3/Bedrock-control clients, same posture as
tests/test_llm_probability_e2e.py's fake Converse client -- no AWS account,
no boto3 install required, verifies the orchestration (submit/poll/collect,
cache reuse, the per-model-minimum fallback, honest per-call tier tagging),
not live AWS behavior (src/bedrock/batch_formats.py's module docstring
names exactly what still needs a live smoke test before real spend).

Run:  python tests/test_llm_batch.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import methods as _m  # noqa: E402,F401  (populates the registry)
from src.bedrock import batch_formats  # noqa: E402
from src.run_benchmark import run_cell  # noqa: E402
from tests.test_smoke import _write_task  # noqa: E402

FIXTURE_MODEL = "amazon.nova-lite"   # provider "amazon_nova" per batch_formats.detect_provider
TASK, PHASE, FOLDER = "serious_adverse_rate_yn", "Phase2", "serious-adverse-event-forecasting"


# ------------------------------------------------------------- batch_formats

def test_detect_provider():
    assert batch_formats.detect_provider("us.anthropic.claude-opus-4-5-20251101-v1:0") == "anthropic"
    assert batch_formats.detect_provider("amazon.nova-lite-v1:0") == "amazon_nova"
    assert batch_formats.detect_provider("us.meta.llama4-maverick-17b-instruct-v1:0") == "meta_llama"
    assert batch_formats.detect_provider("deepseek.v3.2") == "deepseek"
    try:
        batch_formats.detect_provider("some.unknown.model")
        raise AssertionError("expected ValueError for an unregistered provider")
    except ValueError:
        pass
    print("detect_provider OK")


def test_build_and_extract_round_trip_all_providers():
    """Each provider's build_model_input -> (simulated response) ->
    extract_text/extract_usage must round-trip the text and token counts --
    this is what a fake batch job output below simulates, and what a real
    one must match to be usable at all."""
    cases = {
        "us.anthropic.claude-opus-4-5-20251101-v1:0": {"content": [{"type": "text", "text": '{"probability": 42}'}],
                                                        "usage": {"input_tokens": 10, "output_tokens": 5}},
        "amazon.nova-lite-v1:0": {"output": {"message": {"content": [{"text": '{"probability": 42}'}]}},
                                  "usage": {"inputTokens": 10, "outputTokens": 5}},
        "us.meta.llama4-maverick-17b-instruct-v1:0": {"generation": '{"probability": 42}',
                                                       "prompt_token_count": 10, "generation_token_count": 5},
        "deepseek.v3.2": {"choices": [{"message": {"content": '{"probability": 42}'}}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    }
    for model_id, fake_output in cases.items():
        body = batch_formats.build_model_input(model_id, "test prompt", 0.0, 32)
        assert isinstance(body, dict) and body, (model_id, body)
        text = batch_formats.extract_text(model_id, fake_output)
        assert text == '{"probability": 42}', (model_id, text)
        in_tok, out_tok = batch_formats.extract_usage(model_id, fake_output)
        assert (in_tok, out_tok) == (10, 5), (model_id, in_tok, out_tok)
    print("build/extract round trip OK for all 4 providers")


# ------------------------------------------------------- fake AWS injection

class _FakeS3Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body if isinstance(Body, (bytes, bytearray)) else Body.encode("utf-8")

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        client = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                keys = [k for k in client.objects if k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in keys]}
        return _Paginator()


class _FakeBedrockControlClient:
    """Simulates CreateModelInvocationJob by reading the fake S3 input,
    producing a provider-shaped fake response per record (via
    batch_formats' own build/extract shapes, so this fake and the real
    parsing code stay honest about what "correct" looks like), and writing
    a fake output object -- job status is 'Completed' immediately, no
    actual polling delay."""

    def __init__(self, s3_client, probability: int = 55):
        self.s3 = s3_client
        self.probability = probability
        self.jobs = 0
        self.submitted_model_ids = []

    def _fake_model_output(self, model_id: str) -> dict:
        text = json.dumps({"probability": self.probability})
        provider = batch_formats.detect_provider(model_id)
        if provider == "anthropic":
            return {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 50, "output_tokens": 6}}
        if provider == "amazon_nova":
            return {"output": {"message": {"content": [{"text": text}]}},
                    "usage": {"inputTokens": 50, "outputTokens": 6}}
        if provider == "meta_llama":
            return {"generation": text, "prompt_token_count": 50, "generation_token_count": 6}
        return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 50, "completion_tokens": 6}}

    def create_model_invocation_job(self, jobName, roleArn, modelId, inputDataConfig, outputDataConfig):
        self.jobs += 1
        self.submitted_model_ids.append(modelId)
        input_uri = inputDataConfig["s3InputDataConfig"]["s3Uri"]
        output_uri = outputDataConfig["s3OutputDataConfig"]["s3Uri"]
        input_prefix = input_uri.split("s3://", 1)[1].split("/", 1)[1]
        input_key = input_prefix.rstrip("/") + "/input.jsonl"
        lines_out = []
        for line in self.s3.objects[input_key].decode("utf-8").splitlines():
            rec = json.loads(line)
            lines_out.append(json.dumps({"recordId": rec["recordId"],
                                        "modelOutput": self._fake_model_output(modelId)}))
        output_prefix = output_uri.split("s3://", 1)[1].split("/", 1)[1]
        self.s3.objects[output_prefix.rstrip("/") + "/output.jsonl.out"] = "\n".join(lines_out).encode("utf-8")
        job_id = f"job-{self.jobs}"
        return {"jobArn": job_id}

    def get_model_invocation_job(self, jobIdentifier):
        return {"status": "Completed", "jobArn": jobIdentifier}


class _FakeConverseClient:
    """For the below-minimum sync-fallback path (elicit_probability still
    goes through BedrockClient.converse)."""

    def __init__(self, probability: int = 55):
        self.probability = probability
        self.calls = 0

    def converse(self, modelId, messages, inferenceConfig):
        self.calls += 1
        text = json.dumps({"probability": self.probability})
        return {"output": {"message": {"content": [{"text": text}]}},
                "usage": {"inputTokens": 50, "outputTokens": 6}}


def _batch_cfg(data_root, results_dir, boto_client, s3_client, control_client, batch_min_records):
    return {
        "data_root": data_root, "results_dir": results_dir,
        "max_train_rows": None, "max_test_rows": 8, "n_jobs": -1,
        "llm_arm": "L1", "llm_model": FIXTURE_MODEL, "llm_temperature": 0.0,
        "primary_elicitation": "verbalized", "llm_service_tier": "batch",
        "llm_s3_bucket": "fake-wave2-bucket", "llm_batch_role_arn": "arn:aws:iam::123456789012:role/FakeRole",
        "llm_batch_min_records": batch_min_records,
        "llm_boto_client": boto_client, "llm_boto_s3_client": s3_client,
        "llm_boto_bedrock_control_client": control_client,
        "seeds": [42], "bootstrap": {"n_resamples": 20, "ci": 0.95},
    }


def test_batch_path_submits_a_real_job_and_meter_tags_it_batch():
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, FOLDER, PHASE, n_train=40, n_test=8)
        s3 = _FakeS3Client()
        control = _FakeBedrockControlClient(s3, probability=55)
        converse = _FakeConverseClient()
        # batch_min_records=1: any non-empty uncached batch submits for real,
        # exercising the actual submit/poll/collect path rather than the
        # fallback.
        cfg = _batch_cfg(data_root, results_dir, converse, s3, control, batch_min_records=1)

        rec = run_cell(cfg, TASK, PHASE, "llm_probability", seed=42)

        assert rec["status"] == "ok", rec
        assert control.jobs >= 1, "expected at least one batch job to be submitted"
        assert converse.calls == 0, (
            f"expected zero synchronous Converse calls (nothing below the minimum), got {converse.calls}"
        )

        meter = rec["llm_meter"]
        assert meter["tokens_by_tier"].get("sync", {}).get("input_tokens", 0) == 0, meter
        assert "batch" in meter["tokens_by_tier"], meter
        assert meter["mixed_tier"] is False, meter
        # The regression this test exists for: dollars_realized must reflect
        # the batch_multiplier (0.5 for nova-lite), not the sync rate.
        # Tolerance is loose (not 1e-6) because Meter.summary() rounds each
        # dollar figure to 6 decimal places independently -- at this test's
        # toy token counts (order 1e-4 to 1e-5 dollars), that rounding alone
        # visibly perturbs the ratio (measured ~0.507, not exactly 0.5); at
        # real-grid token volumes this rounding is negligible. 2% comfortably
        # absorbs the rounding artifact while still catching a real bug
        # (e.g. the discount not applying at all would give ratio ~= 1.0).
        assert meter["dollars_realized"] < meter["dollars_normalized"], meter
        ratio = meter["dollars_realized"] / meter["dollars_normalized"]
        assert abs(ratio - 0.5) < 0.02, f"expected realized/normalized ~= batch_multiplier (0.5), got {ratio}"
        print("batch submission OK: jobs=", control.jobs, "dollars_realized/normalized ratio =", round(ratio, 4))


def test_below_minimum_falls_back_to_honest_sync():
    """The direct regression test for the found bug: service_tier='batch'
    configured, but every uncached row is below --llm-batch-min-records, so
    every call is genuinely synchronous -- the meter must say so
    (tokens_by_tier has only 'sync'), not silently apply the batch discount
    to calls that were actually billed at full price."""
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, FOLDER, PHASE, n_train=40, n_test=8)
        s3 = _FakeS3Client()
        control = _FakeBedrockControlClient(s3, probability=55)
        converse = _FakeConverseClient()
        # batch_min_records huge: every cell's uncached row count (well
        # under 1000) falls back to sync every time.
        cfg = _batch_cfg(data_root, results_dir, converse, s3, control, batch_min_records=1000)

        rec = run_cell(cfg, TASK, PHASE, "llm_probability", seed=42)

        assert rec["status"] == "ok", rec
        assert control.jobs == 0, f"expected no batch job submitted, got {control.jobs}"
        assert converse.calls > 0, "expected the fallback to make real synchronous calls"

        meter = rec["llm_meter"]
        assert "batch" not in meter["tokens_by_tier"], (
            f"BUG REGRESSION: cell configured for batch but ran entirely synchronous, and "
            f"tokens_by_tier still shows a 'batch' bucket: {meter['tokens_by_tier']}"
        )
        assert meter["tokens_by_tier"].get("sync", {}).get("input_tokens", 0) > 0, meter
        # dollars_realized must equal the SYNC rate here, not the 50%-off
        # batch rate -- this is the exact number that was wrong before the
        # fix (previously this cell would have reported dollars_realized at
        # half of what these real synchronous calls actually cost).
        assert meter["dollars_realized"] == meter["dollars_normalized"], (
            f"expected realized == normalized (everything ran sync at on-demand rate), got "
            f"{meter['dollars_realized']} vs {meter['dollars_normalized']}"
        )
        # requested_service_tier (labeling only) still correctly says "batch"
        # -- the cell WAS configured for batch, it just couldn't submit.
        assert meter["service_tier"] == "batch", meter
        print("fallback-to-sync OK: dollars_realized == dollars_normalized "
             f"(={meter['dollars_realized']}), converse.calls={converse.calls}, jobs=0")


def test_fit_refuses_batch_without_bucket_or_role():
    """run_cell itself doesn't catch method exceptions -- only main()'s sweep
    loop does (converting NotImplementedError to a "skipped" run record) --
    so calling run_cell directly here, this must raise, not return a
    "failed" status dict."""
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as results_dir:
        _write_task(data_root, FOLDER, PHASE, n_train=10, n_test=5)
        converse = _FakeConverseClient()
        cfg = {
            "data_root": data_root, "results_dir": results_dir, "max_test_rows": 5,
            "llm_arm": "L1", "llm_model": FIXTURE_MODEL, "llm_service_tier": "batch",
            "llm_boto_client": converse, "seeds": [42],
            "bootstrap": {"n_resamples": 20, "ci": 0.95},
            # llm_s3_bucket / llm_batch_role_arn deliberately absent
        }
        try:
            run_cell(cfg, TASK, PHASE, "llm_probability", seed=42)
            raise AssertionError("expected NotImplementedError for batch tier with no bucket/role")
        except NotImplementedError as e:
            assert "s3-bucket" in str(e).lower() or "batch-role-arn" in str(e).lower(), e
        assert converse.calls == 0, "no call should have been made before the guard fired"
        print("fail-closed OK: batch tier without bucket/role refuses before any call")


if __name__ == "__main__":
    test_detect_provider()
    test_build_and_extract_round_trip_all_providers()
    test_batch_path_submits_a_real_job_and_meter_tags_it_batch()
    test_below_minimum_falls_back_to_honest_sync()
    test_fit_refuses_batch_without_bucket_or_role()
    print("llm batch tests passed")
