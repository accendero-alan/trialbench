"""Preflight for `deploy/w1_bedrock_inventory.py`: check every IAM
permission that script (and its opt-in `--run-batch-probe` extension) needs,
**without spending money or creating any real resource**.

Uses `iam:SimulatePrincipalPolicy` -- the AWS-native "would this be allowed"
check -- instead of actually calling `bedrock:InvokeModel` /
`CreatePromptRouter` / `CreateModelInvocationJob` / `s3:PutObject`, all of
which cost money, create a real router, kick off a real batch job, or write
a real object respectively. `w1_bedrock_inventory.py` already does the live
version of most of these (and reports SKIPPED with the exact AWS error on
denial); this script's job is to answer "which permissions am I missing"
*before* running that, cheaply and repeatably.

Two-tier check:

1. **Read-only, harmless actions** (`sts:GetCallerIdentity`,
   `iam:List*RolePolicies`, `servicequotas:List*`) are just called directly
   -- there's no reason to simulate something that costs nothing to try for
   real, and a direct call is strictly more accurate than a simulation.
2. **Costly / mutating actions** (`bedrock:InvokeModel`,
   `bedrock:CreatePromptRouter`, `bedrock:CreateModelInvocationJob`,
   `bedrock:GetModelInvocationJob`, `s3:PutObject`, `s3:GetObject`) are
   checked via `simulate_principal_policy` against a best-effort resource
   ARN (per-model ARNs for `InvokeModel`, drawn from
   `configs/bedrock_prices.yaml`; the given `--s3-bucket` for the two S3
   actions; a wildcard resource for actions whose resource-ARN pattern
   isn't cleanly documented, e.g. `CreatePromptRouter` against a router
   that doesn't exist yet). A wildcard-resource simulation catches
   identity-based policy grants/denies but **cannot see resource-based
   policies or SCPs** -- flagged in the report, not silently assumed
   complete. If `iam:SimulatePrincipalPolicy` itself isn't allowed (common
   on tightly-scoped execution roles), that tier is reported as
   UNDETERMINED, not silently skipped, with a pointer to the live
   alternative (`w1_bedrock_inventory.py` itself, or `--live-fallback`
   below).

`--live-fallback` upgrades any UNDETERMINED bedrock:InvokeModel row to a
real one-token Converse call per model (tiny, real cost, real inference) --
opt-in for exactly the same reason `w1_bedrock_inventory.py --probe-routing-fee`
is opt-in: this script's whole point is to answer the question without
spending money, so spending money is never the default.

Usage (from the repo root, on the credentialed instance):

    python deploy/w1_permissions_check.py
    python deploy/w1_permissions_check.py --s3-bucket my-wave2-bucket
    python deploy/w1_permissions_check.py --live-fallback   # only if simulate itself is denied
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PRICE_TABLE_PATH = os.path.join(ROOT, "configs", "bedrock_prices.yaml")
DEFAULT_OUT_PATH = os.path.join(HERE, "w1_permissions_report.md")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _load_ladder():
    with open(PRICE_TABLE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def _model_arn(region: str, account_id: str, entry: dict) -> str:
    """Mirrors src/bedrock/prices.py's resolve_model_id logic for *which*
    id to use, then builds the matching ARN: foundation-model (no account
    segment) for a bare model_id, inference-profile (account-scoped) for a
    us./global.-prefixed one -- the same distinction W1.2 already
    established live per model (configs/bedrock_prices.yaml's `profile`
    field)."""
    model_id = entry["model_id"]
    if entry.get("profile"):
        return f"arn:aws:bedrock:{region}:{account_id}:inference-profile/{model_id}"
    return f"arn:aws:bedrock:{region}::foundation-model/{model_id}"


# ------------------------------------------------------------------ tier 1
# Read-only, zero-cost -- just call them for real; a direct call beats a
# simulation for accuracy and these have no side effects worth avoiding.

def check_readonly_actions(results: list, region: str) -> str | None:
    """Returns the caller's role ARN (for tier-2's PolicySourceArn) if
    resolvable, else None."""
    import boto3
    import botocore.exceptions

    def _try(action, fn):
        try:
            fn()
            results.append({"tier": 1, "action": action, "resource": "(direct call)",
                            "decision": "ALLOWED", "note": ""})
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            results.append({"tier": 1, "action": action, "resource": "(direct call)",
                            "decision": "DENIED" if code in ("AccessDenied", "AccessDeniedException",
                                                             "UnauthorizedOperation") else f"ERROR ({code})",
                            "note": str(e)})
        except Exception as e:  # noqa: BLE001 -- e.g. no credentials at all, boto3 missing
            results.append({"tier": 1, "action": action, "resource": "(direct call)",
                            "decision": "UNDETERMINED", "note": f"{type(e).__name__}: {e}"})

    sts = boto3.client("sts", region_name=region)
    role_arn = {"v": None}

    def _sts():
        ident = sts.get_caller_identity()
        role_arn["v"] = ident["Arn"]
    _try("sts:GetCallerIdentity", _sts)

    if role_arn["v"] and ":assumed-role/" in role_arn["v"]:
        account_id = role_arn["v"].split(":")[4]
        role_name = role_arn["v"].split("/")[1]
        iam_role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
        iam = boto3.client("iam", region_name=region)
        _try("iam:ListAttachedRolePolicies", lambda: iam.list_attached_role_policies(RoleName=role_name))
        _try("iam:ListRolePolicies", lambda: iam.list_role_policies(RoleName=role_name))
    else:
        iam_role_arn = None
        results.append({"tier": 1, "action": "iam:ListAttachedRolePolicies / iam:ListRolePolicies",
                        "resource": "(direct call)", "decision": "UNDETERMINED",
                        "note": "caller is not an assumed role (or identity check failed) -- "
                               "can't resolve a role name to check these against."})

    sq = boto3.client("service-quotas", region_name=region)
    _try("servicequotas:ListAWSDefaultServiceQuotas",
        lambda: next(iter(sq.get_paginator("list_aws_default_service_quotas").paginate(ServiceCode="bedrock"))))
    _try("servicequotas:ListServiceQuotas",
        lambda: next(iter(sq.get_paginator("list_service_quotas").paginate(ServiceCode="bedrock"))))

    return iam_role_arn


# ------------------------------------------------------------ tier 2 (sim)

def _simulate_one(iam, policy_source_arn: str, action: str, resource: str):
    resp = iam.simulate_principal_policy(
        PolicySourceArn=policy_source_arn, ActionNames=[action], ResourceArns=[resource],
    )
    evals = resp.get("EvaluationResults", [])
    if not evals:
        return "UNDETERMINED", "simulate_principal_policy returned no EvaluationResults"
    decision = evals[0]["EvalDecision"]
    label = {"allowed": "ALLOWED", "explicitDeny": "DENIED (explicit)",
            "implicitDeny": "DENIED (implicit -- no matching Allow)"}.get(decision, decision)
    matched = [s.get("SourcePolicyId", "?") for s in evals[0].get("MatchedStatements", [])]
    note = f"matched policy: {matched}" if matched else ""
    return label, note


def check_simulated_actions(results: list, region: str, role_arn: str, s3_bucket: str | None) -> None:
    import boto3
    import botocore.exceptions

    if role_arn is None:
        results.append({"tier": 2, "action": "(all tier-2 actions)", "resource": "-",
                        "decision": "UNDETERMINED",
                        "note": "no resolvable IAM role ARN (see tier 1) -- can't call "
                               "simulate_principal_policy without a PolicySourceArn."})
        return

    iam = boto3.client("iam", region_name=region)
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    ladder = _load_ladder()
    rows = []
    for key, entry in ladder.items():
        rows.append(("bedrock:InvokeModel", _model_arn(region, account_id, entry), f"model={key}"))

    router_wildcard = f"arn:aws:bedrock:{region}:{account_id}:prompt-router/*"
    job_wildcard = f"arn:aws:bedrock:{region}:{account_id}:model-invocation-job/*"
    rows.append(("bedrock:CreatePromptRouter", router_wildcard,
                "resource doesn't exist yet -- wildcard, so a resource-scoped Deny may not show up here"))
    rows.append(("bedrock:CreateModelInvocationJob", job_wildcard,
                "wildcard -- same caveat"))
    rows.append(("bedrock:GetModelInvocationJob", job_wildcard, "wildcard -- same caveat"))

    if s3_bucket:
        rows.append(("s3:PutObject", f"arn:aws:s3:::{s3_bucket}/*", f"bucket={s3_bucket}"))
        rows.append(("s3:GetObject", f"arn:aws:s3:::{s3_bucket}/*", f"bucket={s3_bucket}"))
    else:
        rows.append(("s3:PutObject", "arn:aws:s3:::*/*",
                    "no --s3-bucket given -- wildcard bucket, least precise row in this report"))
        rows.append(("s3:GetObject", "arn:aws:s3:::*/*", "no --s3-bucket given -- wildcard bucket"))

    for action, resource, note in rows:
        try:
            decision, sim_note = _simulate_one(iam, role_arn, action, resource)
            results.append({"tier": 2, "action": action, "resource": resource,
                            "decision": decision, "note": f"{note}; {sim_note}".strip("; ")})
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            if code in ("AccessDenied", "AccessDeniedException"):
                results.append({"tier": 2, "action": action, "resource": resource,
                                "decision": "UNDETERMINED",
                                "note": f"{note}; iam:SimulatePrincipalPolicy itself denied -- "
                                       f"cannot check this without spending money for real "
                                       f"(see --live-fallback for bedrock:InvokeModel)"})
            else:
                results.append({"tier": 2, "action": action, "resource": resource,
                                "decision": f"ERROR ({code})", "note": f"{note}; {e}"})
        except Exception as e:  # noqa: BLE001
            results.append({"tier": 2, "action": action, "resource": resource,
                            "decision": "UNDETERMINED", "note": f"{note}; {type(e).__name__}: {e}"})


# --------------------------------------------------------- live fallback

def live_fallback_invoke_model(results: list, region: str) -> None:
    """Only reached with --live-fallback: a real one-token Converse call
    per ladder model, for exactly the rows tier 2 left UNDETERMINED because
    simulate_principal_policy itself was denied. Real (tiny) cost -- this
    is the same tradeoff w1_bedrock_inventory.py's --probe-routing-fee
    makes, opt-in for the same reason."""
    import boto3
    import botocore.exceptions

    client = boto3.client("bedrock-runtime", region_name=region)
    ladder = _load_ladder()
    for key, entry in ladder.items():
        try:
            client.converse(
                modelId=entry["model_id"],
                messages=[{"role": "user", "content": [{"text": "Reply with only the word OK."}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 5},
            )
            results.append({"tier": "live", "action": "bedrock:InvokeModel", "resource": key,
                            "decision": "ALLOWED", "note": "live Converse call succeeded"})
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            results.append({"tier": "live", "action": "bedrock:InvokeModel", "resource": key,
                            "decision": "DENIED" if "AccessDenied" in code else f"ERROR ({code})",
                            "note": str(e)})
        except Exception as e:  # noqa: BLE001
            results.append({"tier": "live", "action": "bedrock:InvokeModel", "resource": key,
                            "decision": "UNDETERMINED", "note": f"{type(e).__name__}: {e}"})


# --------------------------------------------------------------- report

def render_report(results: list, region: str) -> str:
    lines = [f"# W1 permissions preflight\n\nGenerated {_now()} by "
            f"`deploy/w1_permissions_check.py --region {region}`.\n"]

    denied_or_undetermined = [r for r in results if r["decision"] != "ALLOWED"]
    if denied_or_undetermined:
        lines.append("## Not confirmed allowed -- fix these before running `w1_bedrock_inventory.py`\n")
        lines.append("| Action | Resource | Decision | Note |")
        lines.append("|---|---|---|---|")
        for r in denied_or_undetermined:
            lines.append(f"| `{r['action']}` | `{r['resource']}` | **{r['decision']}** | {r['note']} |")
        lines.append("")
    else:
        lines.append("## Every checked permission came back ALLOWED\n")

    lines.append("## Full results\n")
    lines.append("| Tier | Action | Resource | Decision | Note |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['tier']} | `{r['action']}` | `{r['resource']}` | {r['decision']} | {r['note']} |")

    lines.append("")
    lines.append(
        "**Reading this report.** Tier 1 rows are real calls -- ALLOWED/DENIED there is exact. "
        "Tier 2 rows are `iam:SimulatePrincipalPolicy` results against a best-effort resource ARN "
        "(wildcarded where the true resource doesn't exist yet, e.g. a not-yet-created router) -- "
        "**a simulated ALLOWED can still fail live** if a resource-based policy or an SCP applies "
        "that the simulator can't see from an identity-policy-only evaluation. Treat ALLOWED here as "
        "'no identity-policy reason this should fail,' not an absolute guarantee, and treat any DENIED "
        "as a real blocker to fix in IAM before spending money. UNDETERMINED rows mean this script "
        "couldn't get an answer at all (usually `iam:SimulatePrincipalPolicy` itself was denied) -- "
        "re-run with `--live-fallback` to resolve the `bedrock:InvokeModel` rows with a real tiny "
        "call, or check the other UNDETERMINED rows by hand against your IAM policy."
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH)
    ap.add_argument("--s3-bucket", default=None,
                    help="the intended Wave 2 batch bucket, for precise S3 permission rows -- "
                         "omitted rows fall back to a wildcard bucket ARN (less precise)")
    ap.add_argument("--live-fallback", action="store_true",
                    help="if iam:SimulatePrincipalPolicy is denied, resolve the bedrock:InvokeModel "
                         "rows with a real one-token Converse call per ladder model instead (tiny "
                         "but real cost -- opt-in, same posture as w1_bedrock_inventory.py's "
                         "--probe-routing-fee)")
    args = ap.parse_args()

    results = []
    try:
        role_arn = check_readonly_actions(results, args.region)
    except Exception as e:  # noqa: BLE001 -- e.g. boto3 not installed
        print(f"FATAL during tier 1 (read-only checks): {type(e).__name__}: {e}", file=sys.stderr)
        role_arn = None
        results.append({"tier": 1, "action": "(tier 1 setup)", "resource": "-", "decision": "UNDETERMINED",
                        "note": f"{type(e).__name__}: {e}"})

    try:
        check_simulated_actions(results, args.region, role_arn, args.s3_bucket)
    except Exception as e:  # noqa: BLE001
        print(f"FATAL during tier 2 (simulated checks): {type(e).__name__}: {e}", file=sys.stderr)
        results.append({"tier": 2, "action": "(tier 2 setup)", "resource": "-", "decision": "UNDETERMINED",
                        "note": f"{type(e).__name__}: {e}"})

    if args.live_fallback:
        sim_denied_invoke = any(
            r["action"] == "bedrock:InvokeModel" and r["decision"] == "UNDETERMINED" for r in results
        )
        if sim_denied_invoke:
            print("--live-fallback: simulate was undetermined for bedrock:InvokeModel -- "
                 "issuing real one-token Converse calls instead.")
            live_fallback_invoke_model(results, args.region)
        else:
            print("--live-fallback given but tier 2 already resolved bedrock:InvokeModel -- skipping "
                 "the real calls (nothing to gain by spending money to re-confirm).")

    report = render_report(results, args.region)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nwrote {args.out}")

    n_denied = sum(1 for r in results if str(r["decision"]).startswith("DENIED"))
    n_undetermined = sum(1 for r in results if r["decision"] == "UNDETERMINED")
    print(f"{n_denied} denied, {n_undetermined} undetermined, "
         f"{len(results) - n_denied - n_undetermined} allowed, of {len(results)} checked.")
    if n_denied:
        sys.exit(1)


if __name__ == "__main__":
    main()
