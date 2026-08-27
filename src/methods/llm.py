"""P13.1 (wave2-start-plan.md): ``llm_probability``, driven entirely by
config the way ``--icd-granularity`` drives the code rungs -- T28 runs
*through* ``run_benchmark.py``, not around it, so resume, per-fit prediction
persistence, the leaderboard, the pairing convention and the B3 guard all come
for free, and T28's analysis script reads the same ``results/predictions/``
layout T22-T24 already read.

Substrate is Amazon Bedrock (revision 3 supersedes revision 1's local-GPU
design; see P13.0's removal of ``llm_backend.py``'s local-server-backed
client).
Zero-shot only, per the plan's design: ``fit`` asserts it wasn't handed any
few-shot exemplars rather than quietly ignoring a ``llm_fewshot_k`` param
that would otherwise do nothing.
"""
from __future__ import annotations

import uuid

import numpy as np

from ..bedrock.client import BedrockClient, elicit_probability
from ..bedrock.router import route_call
from ..data.serialize import ARMS, render_arm
from .base import BaseMethod
from .registry import register

DEFAULT_PRIMARY_ELICITATION = "verbalized"  # revision 3 §6.3 -- reverses revision 1/2's "logprob" default
DEFAULT_REGION = "us-west-2"  # W1.2, 2026-08-27: confirmed live region for the five-rung ladder + router pair
DEFAULT_SERVICE_TIER = "sync"
DEFAULT_MAX_TOKENS = 32  # matches BedrockClient.converse's own default -- the verbalized JSON reply is short
# W1.4 confirmed live, 2026-08-27 (deploy/w1_bedrock_inventory.py --run-batch-probe): a 10-record
# DeepSeek V3.2 batch job failed with "contains less records (10) than the required minimum of: 100"
# -- this constant's placeholder value happened to already match. Confirmed for DeepSeek specifically;
# assumed uniform across providers (the error is a Bedrock-service-level record-count gate, not
# provider-specific) but not independently confirmed for every ladder model -- override with
# --llm-batch-min-records if a different model's batch job ever reports a different minimum.
DEFAULT_BATCH_MIN_RECORDS = 100


def _safe_s3_component(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)


# Per-task question the trial record is judged against. "outcome" here is
# TrialBench's trial-approval-forecasting task (loader.py TASKS), not the
# generic English word.
TASK_QUESTIONS = {
    "mortality_rate_yn": "an all-cause mortality event will occur among this trial's participants",
    "serious_adverse_rate_yn": "a serious adverse event will occur among this trial's participants",
    "patient_dropout_rate_yn": "this trial will experience a patient dropout event",
    "outcome": "this trial's outcome will be a success (approved / positive result)",
}


def _build_verbalized_prompt(question: str, trial_text: str) -> str:
    return (
        "You are a clinical trial risk assessor. Read the trial record below and estimate "
        f"the probability (0-100) that {question}.\n\n"
        f"Trial record:\n{trial_text}\n\n"
        'Respond with ONLY a JSON object: {"probability": <integer 0-100>}.'
    )


@register("llm_probability")
class LLMProbability(BaseMethod):
    feature_view = "raw"

    def __init__(self, task_type: str = "binary", num_classes: int = 2, seed: int = 42, **params):
        super().__init__(task_type=task_type, num_classes=num_classes, seed=seed, **params)
        self.task = params.get("task")
        self.arm = params.get("llm_arm")
        self.model = params.get("llm_model")
        self.temperature = params.get("llm_temperature", 0.0)
        self.primary_elicitation = params.get("primary_elicitation", DEFAULT_PRIMARY_ELICITATION)
        self.max_calls = params.get("llm_max_calls")
        self.results_dir = params.get("results_dir", "results")
        self.service_tier = params.get("llm_service_tier", DEFAULT_SERVICE_TIER)
        self.router_arn = params.get("llm_router_arn")
        self.region = params.get("llm_region", DEFAULT_REGION)
        # P13.8, wired 2026-08-27: real batch submission needs a bucket to
        # stage records in and a role Bedrock assumes to read/write it --
        # both required (fit() checks) before service_tier="batch" runs for
        # real; --llm-service-tier sync needs neither.
        self.s3_bucket = params.get("llm_s3_bucket")
        self.batch_role_arn = params.get("llm_batch_role_arn")
        # `or`, not `.get(key, default)`: run_benchmark.py passes this
        # through as an explicit `None` keyword when --llm-batch-min-records
        # wasn't given, which makes the key *present* with value None --
        # `.get(key, default)` would then return None, not the default,
        # since the key isn't actually missing.
        self.batch_min_records = params.get("llm_batch_min_records") or DEFAULT_BATCH_MIN_RECORDS
        # Test-only injection points: fake objects standing in for
        # boto3.client("bedrock-runtime")/("s3")/("bedrock") respectively,
        # so tests never need boto3 installed or a live AWS account (see
        # BedrockClient's own `boto_client` param, same pattern).
        self._injected_boto_client = params.get("llm_boto_client")
        self._injected_s3_client = params.get("llm_boto_s3_client")
        self._injected_bedrock_control_client = params.get("llm_boto_bedrock_control_client")
        self._client = None
        self._resolved_model_id = None  # self.model resolved to a concrete Bedrock id; see predict_proba
        self.llm_meter = None  # constructed lazily in fit(); see the note there
        self.llm_n_scored = 0
        self.llm_n_parse_failures = 0
        self.llm_n_refusals = 0

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if not self.arm or not self.model:
            raise NotImplementedError(
                "llm_probability requires --llm-arm and --llm-model (P13.1); neither was configured."
            )
        if self.arm not in ARMS:
            raise ValueError(f"unknown --llm-arm {self.arm!r}, expected one of {ARMS}")
        if self.params.get("llm_fewshot_k"):
            raise NotImplementedError(
                "llm_probability is zero-shot (verbalized probability) per the plan's design -- "
                "llm_fewshot_k exemplars are not implemented, not silently ignored."
            )
        if self.service_tier == "batch" and not self.router_arn and not (self.s3_bucket and self.batch_role_arn):
            raise NotImplementedError(
                "llm_probability service_tier='batch' requires --llm-s3-bucket and "
                "--llm-batch-role-arn (P13.8) -- both are needed to actually submit a batch job. "
                "Use --llm-service-tier sync if a batch bucket/role isn't set up yet; a config "
                "that can't submit must not silently fall back to sync and mislabel the cost."
            )
        # Imported lazily (bedrock.meter has no heavy deps, but keeping the
        # construction here rather than at class-import time matches the
        # pattern the rest of this method already follows for `_client`).
        from ..bedrock.meter import Meter
        self.llm_meter = Meter()
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.task_type != "binary":
            raise NotImplementedError("llm_probability is scoped to binary tasks; multiclass is out of scope.")
        if self.max_calls is not None and len(X) > self.max_calls:
            raise ValueError(
                f"predict_proba got {len(X)} rows but --llm-max-calls={self.max_calls}. "
                f"Reduce the sample via --test-subset-file or --max-test-rows instead of silently "
                f"truncating -- a truncated proba array would break row alignment with y_test."
            )
        question = TASK_QUESTIONS.get(self.task)
        if question is None:
            raise NotImplementedError(
                f"no LLM question template for task {self.task!r}; add one to TASK_QUESTIONS."
            )
        if self._client is None:
            self._client = BedrockClient(region=self.region, boto_client=self._injected_boto_client)
        if self._resolved_model_id is None:
            # self.model is configs/bedrock_prices.yaml's table key (e.g.
            # "anthropic.claude-opus-4-5") -- what --llm-model and
            # configs/wave2_amendment.yaml's P13.10 guard both use. The
            # Converse API needs the concrete id/inference-profile ARN
            # instead (several ladder models reject their bare id --
            # confirmed live, 2026-08-27; see resolve_model_id's docstring).
            # Sending the table key straight to the API is exactly the bug
            # this resolves: it would fail at Bedrock, not here.
            from ..bedrock.prices import load_price_table, resolve_model_id
            self._resolved_model_id = resolve_model_id(self.model, load_price_table())

        # P13.8, wired 2026-08-27: batch is a fundamentally different call
        # shape (submit everything, poll, collect) from the per-row loop
        # below, so it's a separate method rather than a branch inside the
        # loop. Routers force sync at the CLI layer already (P13.9,
        # run_benchmark.py's `--llm-router-arn` handling), so this never
        # fires for a routed cell.
        if self.service_tier == "batch" and not self.router_arn:
            return self._predict_batch(X, question)

        probs = []
        for nct_id, row in X.iterrows():
            rendered = render_arm(row, self.arm, str(nct_id))
            prompt = _build_verbalized_prompt(question, rendered.text)

            if self.router_arn:
                # T31: routed calls go through the router client directly
                # (synchronous only, attribution hard-checked there) rather
                # than through elicit_probability's cache+client wiring,
                # since a router call's cost/attribution accounting differs
                # from a direct model call's.
                from ..bedrock.cache import cache_get, cache_put
                from ..bedrock.client import _looks_like_refusal, _parse_verbalized

                cached = cache_get(self.results_dir, self.router_arn, prompt, self.temperature, "sync")
                if cached is not None:
                    self.llm_meter.record_cache_hit()
                    text = cached["response"]["text"]
                    prob = _parse_verbalized(text)
                    refused = _looks_like_refusal(text)
                else:
                    result = route_call(self._client, self.router_arn, prompt, temperature=self.temperature)
                    self.llm_meter.record_call(result.input_tokens, result.output_tokens,
                                               result.latency_secs, throttle_count=result.retry_count,
                                               routed=True, service_tier="sync")
                    cache_put(self.results_dir, self.router_arn, prompt, self.temperature, "sync",
                             {"text": result.text, "input_tokens": result.input_tokens,
                              "output_tokens": result.output_tokens,
                              "invoked_model_id": result.invoked_model_id})
                    prob = _parse_verbalized(result.text)
                    refused = _looks_like_refusal(result.text)
                parse_ok = prob is not None
                primary_prob = prob if prob is not None else 0.5
            else:
                result = elicit_probability(
                    self._client, self.results_dir, self._resolved_model_id, prompt, self.temperature,
                    primary=self.primary_elicitation, meter=self.llm_meter,
                )
                parse_ok = result.parse_ok
                refused = result.refused
                primary_prob = result.primary_prob

            self.llm_n_scored += 1
            if not parse_ok:
                self.llm_n_parse_failures += 1
            if refused:
                self.llm_n_refusals += 1
            probs.append(primary_prob)
        return np.asarray(probs, dtype=float)

    def _predict_batch(self, X, question: str) -> np.ndarray:
        """P13.8's submit/poll/reassemble loop, actually wired to the
        elicitation path (found missing 2026-08-27, wave2-start-plan.md
        P13.8's status note -- until now, `service_tier="batch"` never
        submitted anything; every call ran real-time synchronous while the
        meter reported the batch discount anyway).

        Renders every prompt up front, reuses the response cache exactly
        like the sync path but keyed "batch", and submits a real Bedrock
        batch job only for the rows that are neither cached nor below
        `self.batch_min_records` -- a below-minimum remainder falls back to
        real synchronous calls via `elicit_probability`, which now always
        records itself honestly as `service_tier="sync"` regardless of
        this method's own tier (src/bedrock/client.py), so the meter's
        realized cost is a correct mix rather than a mislabeled average.
        """
        from ..bedrock import batch as batch_mod
        from ..bedrock import batch_formats
        from ..bedrock.cache import cache_get, cache_put
        from ..bedrock.client import _looks_like_refusal, _parse_verbalized

        rows = list(X.iterrows())
        nct_ids = [str(nct_id) for nct_id, _ in rows]
        prompts = {str(nct_id): _build_verbalized_prompt(question, render_arm(row, self.arm, str(nct_id)).text)
                  for nct_id, row in rows}

        outcomes = {}  # nct_id -> (primary_prob, parse_ok, refused)

        def _record_from_text(nct_id, text):
            prob = _parse_verbalized(text)
            outcomes[nct_id] = (prob if prob is not None else 0.5, prob is not None, _looks_like_refusal(text))

        to_submit = {}
        for nct_id, prompt in prompts.items():
            cached = cache_get(self.results_dir, self._resolved_model_id, prompt, self.temperature, "batch")
            if cached is not None:
                self.llm_meter.record_cache_hit()
                _record_from_text(nct_id, cached["response"].get("text", ""))
            else:
                to_submit[nct_id] = prompt

        if to_submit and len(to_submit) < self.batch_min_records:
            print(f"  llm_probability: {len(to_submit)} uncached row(s) below "
                 f"--llm-batch-min-records={self.batch_min_records} -- falling back to real "
                 f"synchronous calls for these rows only, honestly recorded as service_tier=sync "
                 f"(P13.8's documented per-model-minimum fallback, not a batch call).", flush=True)
            for nct_id, prompt in to_submit.items():
                result = elicit_probability(self._client, self.results_dir, self._resolved_model_id,
                                            prompt, self.temperature, primary=self.primary_elicitation,
                                            meter=self.llm_meter)
                outcomes[nct_id] = (result.primary_prob, result.parse_ok, result.refused)
            to_submit = {}

        if to_submit:
            s3 = self._injected_s3_client
            if s3 is None:
                import boto3
                s3 = boto3.client("s3", region_name=self.region)
            control = self._injected_bedrock_control_client
            if control is None:
                import boto3
                control = boto3.client("bedrock", region_name=self.region)
            run_token = uuid.uuid4().hex[:8]
            prefix = (f"wave2-batch/{_safe_s3_component(self.task or 'task')}/"
                     f"{_safe_s3_component(self.arm or 'arm')}/"
                     f"{_safe_s3_component(self._resolved_model_id)}/{run_token}")

            records = [
                {"recordId": nct_id,
                 "modelInput": batch_formats.build_model_input(self._resolved_model_id, prompt,
                                                                self.temperature, DEFAULT_MAX_TOKENS)}
                for nct_id, prompt in to_submit.items()
            ]
            input_uris = batch_mod.write_batch_jsonl(records, s3, self.s3_bucket, prefix)
            print(f"  llm_probability: {len(records)} record(s) to submit as {len(input_uris)} "
                 f"batch job(s) under s3://{self.s3_bucket}/{prefix}/", flush=True)

            all_raw_records = {}
            for chunk_i, input_uri in enumerate(input_uris):
                output_uri = input_uri.rstrip("/") + "/output/"
                job_id = batch_mod.submit_batch_job(
                    control, job_name=f"wave2-{run_token}-{chunk_i}"[:63], role_arn=self.batch_role_arn,
                    model_id=self._resolved_model_id, input_s3_uri=input_uri, output_s3_uri=output_uri,
                )
                print(f"  llm_probability: submitted batch job {job_id} ({chunk_i + 1}/{len(input_uris)}) "
                     f"-- polling to completion (this can take a while)...", flush=True)
                final = batch_mod.poll_job(control, job_id)
                status = final.get("status")
                if status != "Completed":
                    raise RuntimeError(
                        f"batch job {job_id} finished with status {status!r}, not 'Completed' -- "
                        f"inspect it in the Bedrock console before retrying. Raw response: {final}"
                    )
                output_prefix = output_uri[len(f"s3://{self.s3_bucket}/"):]
                all_raw_records.update(batch_mod.fetch_batch_output_records(s3, self.s3_bucket, output_prefix))

            reassembled = batch_mod.reassemble(all_raw_records)
            for nct_id, prompt in to_submit.items():
                rec = reassembled.get(nct_id)
                if rec is None or rec.get("error") or rec.get("modelOutput") is None:
                    # A record-level failure (P13.8: "treat one as a parse failure, not a job
                    # failure") -- no text was ever produced. (parse_ok=False, refused=False)
                    # so the counting loop below tallies it as a parse failure, same as any
                    # other unparseable response.
                    outcomes[nct_id] = (0.5, False, False)
                    continue
                text = batch_formats.extract_text(self._resolved_model_id, rec["modelOutput"])
                in_tok, out_tok = batch_formats.extract_usage(self._resolved_model_id, rec["modelOutput"])
                self.llm_meter.record_call(in_tok, out_tok, wall_clock_secs=0.0, service_tier="batch")
                cache_put(self.results_dir, self._resolved_model_id, prompt, self.temperature, "batch",
                         {"text": text, "input_tokens": in_tok, "output_tokens": out_tok,
                          "invoked_model_id": self._resolved_model_id})
                _record_from_text(nct_id, text)

        probs = []
        for nct_id in nct_ids:
            primary_prob, parse_ok, refused = outcomes[nct_id]
            self.llm_n_scored += 1
            if not parse_ok:
                self.llm_n_parse_failures += 1
            if refused:
                self.llm_n_refusals += 1
            probs.append(primary_prob)
        return np.asarray(probs, dtype=float)

    @property
    def llm_parse_failure_rate(self) -> float:
        return self.llm_n_parse_failures / self.llm_n_scored if self.llm_n_scored else 0.0

    @property
    def llm_refusal_rate(self) -> float:
        return self.llm_n_refusals / self.llm_n_scored if self.llm_n_scored else 0.0

    @property
    def resolved_model_id(self):
        """The concrete Bedrock id/inference-profile ARN self.model resolved
        to -- None until predict_proba has run at least once. Provenance:
        the run record should carry both self.model (the table key/what was
        requested) and this (what was actually invoked)."""
        return self._resolved_model_id

    def llm_meter_summary(self) -> dict:
        """P13.5: loads P15's pinned price table and reports both the
        realized and normalized cost bases for this fit's calls.

        Token cost is priced against ``self.model``. A router fit
        (``self.router_arn`` set) mixes calls across whichever member the
        router actually invoked per request -- that per-call member isn't
        tracked here (T31's analysis reconstructs it from ``invoked_model_id``
        in the cache/raw response instead), so a router-only fit still
        requires ``--llm-model`` to be set to a priced entry for the token-
        cost side of this summary; the router's own per-request fee is what
        ``Meter.summary`` adds on top from ``self.llm_meter.routing_requests``.
        A missing/unpriced model id fails closed via ``UnpricedModelError``
        (P15) rather than guessing.
        """
        from ..bedrock.prices import is_verified, load_price_table
        table = load_price_table()
        summary = self.llm_meter.summary(table, self.model, self.service_tier)
        summary["price_verified"] = is_verified(self.model, table)
        return summary
