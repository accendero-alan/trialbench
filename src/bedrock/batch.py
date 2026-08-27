"""P13.8 (wave2-start-plan.md): the batch runner. Batch is 50% off
(P15's ``batch_multiplier``) and the default service tier for every ladder
arm.

Splits a request list at ``max_records_per_job`` (the plan's ``[W1]``-flagged
figure of 10,000 is the default here -- override once W1.4 confirms the
documented maximum). Below a model's minimum record count (also a W1.4
number), callers should fall back to synchronous calls via ``client.py`` and
record that the fallback happened, since it changes the price basis.
"""
from __future__ import annotations

import json
import time
import uuid

DEFAULT_MAX_RECORDS_PER_JOB = 10_000   # [W1] -- AWS's documented maximum is unconfirmed

_TERMINAL_STATUSES = {"Completed", "Failed", "Stopped", "PartiallyCompleted", "Expired"}


def write_batch_jsonl(records: list, s3_client, bucket: str, prefix: str,
                      max_records_per_job: int = DEFAULT_MAX_RECORDS_PER_JOB) -> list:
    """``records`` is a list of ``{"recordId": ..., "modelInput": {...}}``
    dicts. Splits into chunks of ``max_records_per_job``, uploads each chunk
    as one JSONL object under its own sub-prefix -- ``InputDataConfig``
    needs the **folder**, not a single file, per the batch data format doc.
    Returns the list of S3 folder URIs, one per chunk, each of which a
    separate job (:func:`submit_batch_job`) points at."""
    uris = []
    for start in range(0, len(records), max_records_per_job):
        chunk = records[start:start + max_records_per_job]
        chunk_id = uuid.uuid4().hex[:8]
        key = f"{prefix.rstrip('/')}/{chunk_id}/input.jsonl"
        body = "\n".join(json.dumps(r) for r in chunk).encode("utf-8")
        s3_client.put_object(Bucket=bucket, Key=key, Body=body)
        uris.append(f"s3://{bucket}/{prefix.rstrip('/')}/{chunk_id}/")
    return uris


def submit_batch_job(control_client, job_name: str, role_arn: str, model_id: str,
                     input_s3_uri: str, output_s3_uri: str) -> str:
    """``control_client`` is ``boto3.client("bedrock")`` (control plane) or
    a fake injected for testing. Returns the job's ARN/identifier."""
    resp = control_client.create_model_invocation_job(
        jobName=job_name, roleArn=role_arn, modelId=model_id,
        inputDataConfig={"s3InputDataConfig": {"s3Uri": input_s3_uri}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
    )
    job_id = resp.get("jobArn") or resp.get("jobIdentifier")
    if not job_id:
        raise RuntimeError(f"create_model_invocation_job returned no job id: {resp}")
    return job_id


def fetch_batch_output_records(s3_client, bucket: str, output_prefix: str) -> dict:
    """Read a completed batch job's output back from S3 and return
    ``{recordId: raw_jsonl_line}``, ready for :func:`reassemble`.

    **[W1]-style caveat, same posture as the rest of this repo's live-AWS
    assumptions: the exact output key layout is not independently confirmed
    against a live account.** Bedrock batch inference writes one ``.out``
    file per input file under ``{output_s3_uri}/{job_id}/``, plus a
    ``manifest.json.out`` summary
    (`batch data format doc <https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-data.html>`_).
    This function is deliberately permissive rather than hardcoded to one
    exact filename pattern: it lists every object under ``output_prefix``,
    skips anything with "manifest" in the key, and JSONL-parses the rest --
    correct against the documented shape and tolerant of a filename detail
    changing. **Verify this once against a real completed job**
    (`deploy/w1_bedrock_inventory.py --run-batch-probe`, or `poll_job`'s own
    acceptance check) before trusting it for a real grid pass.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    out = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=output_prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "manifest" in key.lower():
                continue
            body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
            for line in body.decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record_id = json.loads(line).get("recordId")
                except json.JSONDecodeError:
                    continue
                if record_id is not None:
                    out[record_id] = line
    return out


def poll_job(control_client, job_identifier: str, poll_interval_secs: float = 30.0,
            timeout_secs: float = 6 * 3600) -> dict:
    """Blocks until the job reaches a terminal status or ``timeout_secs``
    elapses. Batch job latency is itself a W1 number (separate from
    on-demand's RPM/TPM quotas) -- a long poll here is expected, not stuck."""
    t0 = time.time()
    while True:
        resp = control_client.get_model_invocation_job(jobIdentifier=job_identifier)
        status = resp.get("status")
        if status in _TERMINAL_STATUSES:
            return resp
        if time.time() - t0 > timeout_secs:
            raise TimeoutError(f"batch job {job_identifier} still {status!r} after {timeout_secs}s")
        time.sleep(poll_interval_secs)


def reassemble(records_by_id_text: dict) -> dict:
    """``records_by_id_text`` maps ``recordId`` to the raw output-manifest
    line (already read back from S3 by the caller -- this module doesn't own
    S3 reads for output, only input writes, since output layout is job- and
    format-specific). A record-level failure (``modelOutput`` absent, or an
    error field present) is reported as a parse failure in the returned
    dict's ``"error"`` slot, not raised -- one bad record must not fail the
    whole job's reassembly (P13.8)."""
    out = {}
    for record_id, line in records_by_id_text.items():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            out[record_id] = {"error": f"JSONDecodeError: {e}", "modelOutput": None}
            continue
        if "modelOutput" not in obj:
            out[record_id] = {"error": obj.get("error", "no modelOutput in record"), "modelOutput": None}
        else:
            out[record_id] = {"error": None, "modelOutput": obj["modelOutput"]}
    return out
