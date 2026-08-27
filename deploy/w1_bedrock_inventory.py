"""W1 (wave2-start-plan.md): account/access/quota/router/logprob feasibility,
run from wherever real AWS credentials live (the EC2 instance -- this
script was written and reviewed from a session with no AWS access, per
W1's own gate: "BLOCKING for everything billable").

Automates what a boto3 call can answer (items 1, 4, 6, 8, and half of 7 --
see below) and writes/updates `deploy/bedrock_inventory.md`, every entry
dated, per W1's acceptance check. Two items resist automation entirely and
are emitted as a manual checklist inside the same file instead of being
skipped silently:

  - Item 3 (live console prices) -- the public pricing page renders
    client-side and the Bedrock console's "Model providers" price view has
    no API equivalent; someone has to read it and paste the numbers into
    `configs/bedrock_prices.yaml`, flipping each row's `verified: false`.
  - Item 7 (the routing fee) -- confirmed against the Pricing Calculator or
    reconciled in Cost Explorer after a live run, neither of which is an
    API this script can call. What this script DOES do for item 7: if a
    router is created (item 6), it issues one real routed call so a
    genuine routing-fee line exists in the account's bill for later Cost
    Explorer reconciliation, and logs the exact UTC timestamp so that line
    is easy to find.

Item 2 (model access / ids) is NOT re-run here -- W1.2 already ran for
real on 2026-08-27 (`deploy/smoke_test_converse.py`,
`configs/bedrock_prices.yaml`'s `id_verified` fields) and this script's
job is to close out the rest of W1, not repeat it.

Item 5 (batch eligibility) is a separate, explicitly opt-in step
(`--run-batch-probe`) because unlike everything else here it needs an S3
bucket and an execution role wired up first, and it holds real model
inference cost against the plan's $25 W1/P13 verification allowance --
see its own section below.

Usage (from the repo root, on the credentialed instance):

    python deploy/w1_bedrock_inventory.py
    python deploy/w1_bedrock_inventory.py --skip-router --skip-logprob   # cheap re-run of items 1+4 only
    python deploy/w1_bedrock_inventory.py --run-batch-probe \\
        --s3-bucket my-wave2-bucket --batch-role-arn arn:aws:iam::123456789012:role/Wave2BatchRole

Every network call is wrapped so one missing permission (e.g. no
`iam:GetRole` on the execution role) degrades that one section to a
recorded "SKIPPED: <reason>" line rather than crashing the whole script --
W1's own framing is "produce the inventory," not "every item must
succeed on the first attempt."
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import uuid

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # so `from src.bedrock...` resolves regardless of invocation cwd

PRICE_TABLE_PATH = os.path.join(ROOT, "configs", "bedrock_prices.yaml")
DEFAULT_OUT_PATH = os.path.join(HERE, "bedrock_inventory.md")

# Keyword -> quota-name substrings to match against service-quotas' listing.
# Bedrock's quota names are prose ("On-demand model inference requests per
# minute for Anthropic Claude Opus 4.5"), not a stable code per model, so
# this is a substring match, not an exact key -- print the raw match too so
# a near-miss is visible rather than silently absent.
QUOTA_NAME_KEYWORDS = {
    "anthropic.claude-opus-4-5": ["Claude Opus 4.5", "Claude 4.5 Opus"],
    "anthropic.claude-haiku-4-5": ["Claude Haiku 4.5", "Claude 4.5 Haiku"],
    "deepseek.v3-2": ["DeepSeek"],
    "meta.llama4-maverick-17b": ["Llama 4 Maverick", "Llama4 Maverick"],
    "amazon.nova-lite": ["Nova Lite"],
    "amazon.nova-pro": ["Nova Pro"],
}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _load_ladder(region: str) -> dict:
    with open(PRICE_TABLE_PATH, encoding="utf-8") as f:
        table = yaml.safe_load(f)
    return table["models"]


# ---------------------------------------------------------------- item 1 --

def check_account_and_role(sections: list, region: str) -> None:
    import boto3
    lines = [f"### Item 1 -- account, region, execution role ({_now()})", ""]
    try:
        sts = boto3.client("sts", region_name=region)
        ident = sts.get_caller_identity()
        lines.append(f"- Account: `{ident['Account']}`")
        lines.append(f"- Caller ARN: `{ident['Arn']}`")
        lines.append(f"- Region checked: `{region}`")
        role_arn = ident["Arn"]
    except Exception as e:  # noqa: BLE001
        lines.append(f"- SKIPPED get_caller_identity: {e}")
        role_arn = None

    if role_arn and ":assumed-role/" in role_arn:
        role_name = role_arn.split("/")[1]
        try:
            iam = boto3.client("iam", region_name=region)
            attached = iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]
            lines.append(f"- Execution role `{role_name}` attached managed policies:")
            for p in attached:
                lines.append(f"  - `{p['PolicyName']}` (`{p['PolicyArn']}`)")
            inline = iam.list_role_policies(RoleName=role_name)["PolicyNames"]
            if inline:
                lines.append(f"  - inline policies: {inline}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"- SKIPPED role policy introspection (likely missing `iam:List*` on "
                         f"the execution role itself): {e}")
    else:
        lines.append("- Not running as an assumed role (or identity check failed above) -- "
                     "record the execution role's policy by hand from the IAM console instead.")

    lines.append("")
    lines.append("Required per P13.11: `bedrock:InvokeModel`, `bedrock:CreateModelInvocationJob`, "
                 "`bedrock:GetModelInvocationJob`, `bedrock:CreatePromptRouter`, `s3:GetObject`/"
                 "`s3:PutObject` on the batch bucket, nothing else. Diff the policy above against "
                 "this list by hand.")
    sections.append("\n".join(lines))


# ---------------------------------------------------------------- item 4 --

def check_quotas(sections: list, region: str) -> None:
    import boto3
    lines = [f"### Item 4 -- quotas (RPM / TPM, on-demand and batch) ({_now()})", ""]
    try:
        sq = boto3.client("service-quotas", region_name=region)
        paginator = sq.get_paginator("list_aws_default_service_quotas")
        all_quotas = []
        for page in paginator.paginate(ServiceCode="bedrock"):
            all_quotas.extend(page["Quotas"])
        # Applied (account-adjusted) values override the defaults where present.
        applied_paginator = sq.get_paginator("list_service_quotas")
        applied_by_code = {}
        for page in applied_paginator.paginate(ServiceCode="bedrock"):
            for q in page["Quotas"]:
                applied_by_code[q["QuotaCode"]] = q["Value"]
    except Exception as e:  # noqa: BLE001
        lines.append(f"SKIPPED: {e}")
        sections.append("\n".join(lines))
        return

    for key, keywords in QUOTA_NAME_KEYWORDS.items():
        matches = [q for q in all_quotas if any(kw.lower() in q["QuotaName"].lower() for kw in keywords)]
        lines.append(f"**{key}**")
        if not matches:
            lines.append(f"  - no quota entries matched {keywords} -- check the model is granted "
                         f"access in this account/region, or the quota-name wording changed.")
        for q in matches:
            value = applied_by_code.get(q["QuotaCode"], q["Value"])
            adjusted = " (account-adjusted)" if q["QuotaCode"] in applied_by_code and applied_by_code[q["QuotaCode"]] != q["Value"] else ""
            lines.append(f"  - {q['QuotaName']}: **{value}**{adjusted} (code `{q['QuotaCode']}`)")
        lines.append("")

    lines.append("Batch job quotas (min/max records per job, concurrent jobs) are a **separate** "
                 "quota family from on-demand RPM/TPM above -- re-run this same query and look for "
                 "\"batch\" in the quota name, or check the console's Service Quotas page directly; "
                 "AWS does not always expose the per-model minimum-records-per-batch-job figure "
                 "through this API at all, in which case item 5's live probe is the only way to "
                 "learn it (start it at a small guess and read the error).")
    sections.append("\n".join(lines))


# ---------------------------------------------------------------- item 6 --
# (and half of item 7: one real routed call for later Cost Explorer reconciliation)

ROUTER_SPECS = [
    {
        "name": "wave2-nova-lite-pro-v1",
        "members": ["amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
        "fallback": "amazon.nova-lite-v1:0",
        "response_quality_difference": 0.5,
    },
    # Llama 3.x fallback pair per wave2-start-plan.md §4 -- several Llama 3.x
    # model ids on Bedrock carry EOL dates that have already passed, so this
    # is a documented *attempt*, not a confirmed-good spec. If it fails,
    # that failure is itself the W1.6 answer for this pair -- do not swap in
    # a guessed replacement id without re-reading the current supported list
    # (docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html).
    {
        "name": "wave2-llama3-fallback-v1",
        "members": ["meta.llama3-1-8b-instruct-v1:0", "meta.llama3-1-70b-instruct-v1:0"],
        "fallback": "meta.llama3-1-8b-instruct-v1:0",
        "response_quality_difference": 0.5,
    },
]


def check_router(sections: list, region: str, do_fee_probe_call: bool) -> None:
    import boto3
    from src.bedrock.client import BedrockClient
    from src.bedrock.router import RouterSpec, create_routers, route_call, RouterAttributionError

    lines = [f"### Item 6 -- router feasibility, item 7 (partial) -- one real routed call ({_now()})", ""]
    try:
        control = boto3.client("bedrock", region_name=region)
    except Exception as e:  # noqa: BLE001
        lines.append(f"SKIPPED: could not construct bedrock control-plane client: {e}")
        sections.append("\n".join(lines))
        return

    working_handle = None
    for spec_dict in ROUTER_SPECS:
        member_arns = [f"arn:aws:bedrock:{region}::foundation-model/{m}" for m in spec_dict["members"]]
        fallback_arn = f"arn:aws:bedrock:{region}::foundation-model/{spec_dict['fallback']}"
        spec = RouterSpec(name=spec_dict["name"], member_model_ids=member_arns,
                          fallback_model_id=fallback_arn,
                          response_quality_difference=spec_dict["response_quality_difference"])
        lines.append(f"**{spec.name}** (members={spec_dict['members']}, "
                     f"responseQualityDifference={spec.response_quality_difference})")
        try:
            handles = create_routers(control, [spec])
            handle = handles[0]
            lines.append(f"  - PASS: created, ARN=`{handle.arn}`")
            if working_handle is None:
                working_handle = handle
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
            msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
            lines.append(f"  - FAIL ({code}): {msg}")
        lines.append("")

    if working_handle is not None and do_fee_probe_call:
        lines.append(f"Issuing one real routed call through `{working_handle.name}` for item 7's "
                     f"\"small live run reconciled in Cost Explorer\" requirement...")
        try:
            client = BedrockClient(region=region)
            result = route_call(client, working_handle.arn, "Reply with only the word OK.")
            lines.append(f"  - PASS at {_now()}: invoked_model_id=`{result.invoked_model_id}`, "
                         f"input_tokens={result.input_tokens}, output_tokens={result.output_tokens}. "
                         f"**Check Cost Explorer for a Bedrock routing-fee line item around this "
                         f"timestamp** (may take up to 24h to appear) and record the confirmed "
                         f"per-1000-request fee in `configs/bedrock_prices.yaml`'s "
                         f"`routing_fee_per_1k_requests_usd` / `routing_fee_verified: true`.")
        except RouterAttributionError as e:
            lines.append(f"  - FAIL: {e} -- this is the hard-error case P13.9 calls out: the "
                         f"attribution channel is gone and T31 cannot run on this platform as designed.")
        except Exception as e:  # noqa: BLE001
            lines.append(f"  - FAIL: {e}")
    elif working_handle is not None:
        lines.append("(skipped the one-call fee probe -- pass `--probe-routing-fee` to issue it; "
                     "it is a real billable call.)")
    else:
        lines.append("No router spec succeeded -- item 7's live-run half cannot proceed until one "
                     "does. Re-read the current supported-member table before retrying with "
                     "different ids: "
                     "https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html")

    sections.append("\n".join(lines))


# ---------------------------------------------------------------- item 8 --

LOGPROB_CANDIDATES = ["deepseek.v3-2", "meta.llama4-maverick-17b"]


def check_logprobs(sections: list, region: str) -> None:
    import boto3
    lines = [f"### Item 8 -- logprob availability, DeepSeek V3.2 / Llama 4 Maverick ({_now()})", ""]
    ladder = _load_ladder(region)
    client = boto3.client("bedrock-runtime", region_name=region)

    for key in LOGPROB_CANDIDATES:
        entry = ladder.get(key)
        if entry is None:
            lines.append(f"**{key}**: SKIPPED -- not in {PRICE_TABLE_PATH}")
            continue
        model_id = entry["model_id"]
        lines.append(f"**{key}** (`{model_id}`)")
        # No Converse-level "logprobs" field exists in inferenceConfig; the
        # only documented extension point is additionalModelRequestFields,
        # whose accepted keys are provider-specific and undocumented for
        # logprobs on these two models -- this is exactly what makes it a
        # W1 question rather than a known parameter. Attempt a plausible
        # key and print the FULL raw response either way, so a human can
        # visually check for any logprob-shaped field regardless of
        # whether the guessed key was the right one.
        for guess_key in ("logprobs", "return_logprobs"):
            try:
                resp = client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": "Reply with only the word OK."}]}],
                    inferenceConfig={"temperature": 0.0, "maxTokens": 5},
                    additionalModelRequestFields={guess_key: True},
                )
                lines.append(f"  - call with additionalModelRequestFields={{{guess_key!r}: True}} "
                             f"succeeded. Full response (inspect for any logprob-shaped field):")
                lines.append(f"    ```\n    {resp}\n    ```")
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
                msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
                lines.append(f"  - call with additionalModelRequestFields={{{guess_key!r}: True}} "
                             f"-> {code}: {msg}")
        lines.append("")

    lines.append("Record the answer either way in `src/bedrock/client.py`'s module docstring and "
                 "`wave2-start-plan.md` -- a clean fail on both guessed keys is still an answer "
                 "(\"no logprob path found\"), not a blocker; do not leave this re-run indefinitely "
                 "chasing parameter names without checking each provider's own Bedrock API reference "
                 "first.")
    sections.append("\n".join(lines))


# ---------------------------------------------------------------- item 5 --
# (explicitly opt-in; see module docstring)

def run_batch_probe(sections: list, region: str, s3_bucket: str, batch_role_arn: str,
                    min_records: int) -> None:
    import boto3
    from src.bedrock.batch import write_batch_jsonl, submit_batch_job, poll_job

    lines = [f"### Item 5 -- batch eligibility probe, min_records={min_records} ({_now()})", ""]
    ladder = _load_ladder(region)
    s3 = boto3.client("s3", region_name=region)
    control = boto3.client("bedrock", region_name=region)
    prefix = f"wave2-w1-batch-probe/{uuid.uuid4().hex[:8]}"

    for key, entry in ladder.items():
        records = [
            {"recordId": f"probe-{i}",
             "modelInput": {"messages": [{"role": "user", "content": [{"text": "Reply OK."}]}],
                            "inferenceConfig": {"temperature": 0.0, "maxTokens": 5}}}
            for i in range(min_records)
        ]
        lines.append(f"**{key}** (`{entry['model_id']}`)")
        try:
            uris = write_batch_jsonl(records, s3, s3_bucket, f"{prefix}/{key}/input")
            job_id = submit_batch_job(
                control, job_name=f"w1-probe-{key}-{uuid.uuid4().hex[:6]}", role_arn=batch_role_arn,
                model_id=entry["model_id"], input_s3_uri=uris[0],
                output_s3_uri=f"s3://{s3_bucket}/{prefix}/{key}/output/",
            )
            lines.append(f"  - job submitted: `{job_id}` -- polling to terminal status "
                         f"(this can take a while; batch job latency is itself a W1 number)...")
            final = poll_job(control, job_id, poll_interval_secs=30.0, timeout_secs=3600)
            lines.append(f"  - PASS at {min_records} records: final status `{final.get('status')}`")
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
            msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
            lines.append(f"  - FAIL at {min_records} records ({code}): {msg}")
            if "minim" in msg.lower():
                lines.append(f"    -> error text mentions a minimum; raise --batch-min-records and retry.")
        lines.append("")

    sections.append("\n".join(lines))


# --------------------------------------------------------- manual items --

def manual_checklist(sections: list) -> None:
    lines = [f"### Item 3 -- live console prices (MANUAL, not automatable) ({_now()})", "", (
        "The public pricing page (https://aws.amazon.com/bedrock/pricing/) renders its "
        "tables client-side and cannot be fetched by a script or an API call. Do this by "
        "hand, once, right before P15 is closed out:\n\n"
        "1. Open the Bedrock console -> Model catalog / Model providers pricing view, in "
        f"the account this ran in (see item 1 above), region confirmed in "
        f"`configs/bedrock_prices.yaml` (currently `us-west-2`).\n"
        "2. For each of the 7 rows in `configs/bedrock_prices.yaml` (5 ladder models + "
        "Nova Pro + the routing fee), read the live input/output $-per-1M-token price.\n"
        "3. Update `input_per_1m_usd`/`output_per_1m_usd`, set `verified: true`, and bump "
        "`read_on` to today's date, for every row whose price matches or is corrected.\n"
        "4. If a price has moved by an order of magnitude from what's currently recorded, "
        "STOP and reselect the rung before any billable grid call -- per "
        "wave2-start-plan.md §8, risk 4."
    )]
    sections.append("\n".join(lines))

    lines2 = [f"### Item 7 -- routing fee, remaining manual half ({_now()})", "", (
        "The automated half (item 6, above) issues one real routed call so a genuine fee "
        "line exists in the bill. To close item 7 out:\n\n"
        "1. Cross-check the fee against the AWS Pricing Calculator "
        "(https://calculator.aws/) for Bedrock Intelligent Prompt Routing, same region.\n"
        "2. 24h+ after the probe call above, check Cost Explorer for the routing-fee line "
        "item and confirm it matches `routing_fee_per_1k_requests_usd` in "
        "`configs/bedrock_prices.yaml` (currently $1.00/1k, unverified).\n"
        "3. Set `routing_fee_verified: true` once confirmed by either channel."
    )]
    sections.append("\n".join(lines2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH)
    ap.add_argument("--skip-account", action="store_true")
    ap.add_argument("--skip-quotas", action="store_true")
    ap.add_argument("--skip-router", action="store_true")
    ap.add_argument("--skip-logprob", action="store_true")
    ap.add_argument("--probe-routing-fee", action="store_true",
                    help="issue one real routed call for item 7's Cost Explorer reconciliation "
                         "(billable, tiny -- a handful of tokens through Nova Lite/Pro)")
    ap.add_argument("--run-batch-probe", action="store_true",
                    help="opt-in, separate from everything above -- needs --s3-bucket and "
                         "--batch-role-arn, holds real per-request cost against the plan's $25 "
                         "W1/P13 verification allowance")
    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--batch-role-arn", default=None)
    ap.add_argument("--batch-min-records", type=int, default=10,
                    help="starting guess for the per-model minimum batch size; raise and retry "
                         "if the error mentions a minimum")
    args = ap.parse_args()

    sections = [f"# Bedrock inventory (W1, wave2-start-plan.md)\n\nGenerated {_now()} by "
               f"`deploy/w1_bedrock_inventory.py --region {args.region}`. Every section below is "
               f"independently dated -- re-run any subset to refresh just that section without "
               f"invalidating the rest."]

    def _guarded(label, fn, *a):
        # A missing dep (boto3 not installed) or an unexpected exception in
        # one check must not lose every other section's output and the file
        # write along with it -- module docstring's "one missing permission
        # ... degrades that section" promise, generalized to any failure.
        try:
            fn(sections, *a)
        except Exception as e:  # noqa: BLE001
            sections.append(f"### {label} ({_now()})\n\nSKIPPED -- {type(e).__name__}: {e}")

    if not args.skip_account:
        _guarded("Item 1 -- account, region, execution role", check_account_and_role, args.region)
    if not args.skip_quotas:
        _guarded("Item 4 -- quotas", check_quotas, args.region)
    if not args.skip_router:
        _guarded("Item 6/7 -- router feasibility", check_router, args.region, args.probe_routing_fee)
    if not args.skip_logprob:
        _guarded("Item 8 -- logprob availability", check_logprobs, args.region)
    manual_checklist(sections)

    if args.run_batch_probe:
        if not args.s3_bucket or not args.batch_role_arn:
            raise SystemExit("--run-batch-probe requires both --s3-bucket and --batch-role-arn")
        _guarded("Item 5 -- batch eligibility", run_batch_probe, args.region, args.s3_bucket,
                 args.batch_role_arn, args.batch_min_records)
    else:
        sections.append(f"### Item 5 -- batch eligibility ({_now()})\n\nNot run this pass -- "
                        f"pass `--run-batch-probe --s3-bucket <bucket> --batch-role-arn <arn>` "
                        f"once the batch bucket and execution role exist. Do this after item 4's "
                        f"quota check, not before (a probe below the true minimum fails for the "
                        f"wrong reason).")

    content = "\n\n---\n\n".join(sections) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
