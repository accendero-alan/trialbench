"""W1.2's remaining open item: does each ladder model (+ Nova Pro, the
router member) accept a direct Converse call on its bare model_id, or does
it need a us./global. inference-profile ARN instead? One minimal call per
candidate id (max_tokens=5, trivial prompt) -- cheap, and the only way to
answer this; a catalogue listing (list-foundation-models) doesn't say.

Run from the repo root, with AWS credentials resolved and boto3 installed
(both already true on the EC2 instance per the requirements.txt install):

    python deploy/smoke_test_converse.py
    python deploy/smoke_test_converse.py --region us-west-2   # override if needed

Phase 1 tries every configs/bedrock_prices.yaml model's bare `model_id`.
2026-08-27's run found four of six ("anthropic.claude-opus-4-5",
"anthropic.claude-haiku-4-5", "meta.llama4-maverick-17b", "amazon.nova-pro")
fail with "on-demand throughput isn't supported ... use an inference
profile" -- PROFILE_CANDIDATES below is `aws bedrock list-inference-profiles`'s
answer for those four (both us. and global. exist for the two Claude
models; only us. exists for Llama 4 Maverick and Nova Pro). Phase 2 tries
every candidate for those four, so a price-relevant question (does global.
actually cost less than us. on the same model, per the plan's suspicion)
gets answered by two real calls, not by picking one and hoping.
"""
from __future__ import annotations

import argparse
import os

import boto3
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRICE_TABLE = os.path.join(ROOT, "configs", "bedrock_prices.yaml")

# key -> candidate inference-profile ids, from `aws bedrock list-inference-profiles`
# (2026-08-27, us-west-2). Only populated for keys phase 1 found need one.
PROFILE_CANDIDATES = {
    "anthropic.claude-opus-4-5": [
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "global.anthropic.claude-opus-4-5-20251101-v1:0",
    ],
    "anthropic.claude-haiku-4-5": [
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
    "meta.llama4-maverick-17b": ["us.meta.llama4-maverick-17b-instruct-v1:0"],
    "amazon.nova-pro": ["us.amazon.nova-pro-v1:0"],
}


def _try_converse(client, label, model_id):
    """One minimal Converse call. Returns (status, detail) -- detail is the
    response text + token counts on success, or the AWS error code/message
    on failure (a ValidationException naming "inference profile" means the
    id needs a us./global. prefix; anything else is a different problem)."""
    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Reply with only the word OK."}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 5},
        )
        text = resp.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        usage = resp.get("usage", {})
        print(f"PASS  {label:45s} {model_id:50s} -> {text!r} "
             f"(in={usage.get('inputTokens')}, out={usage.get('outputTokens')})")
        return "PASS", None
    except Exception as e:  # noqa: BLE001 -- diagnostic script, want every failure mode printed
        code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
        msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
        print(f"FAIL  {label:45s} {model_id:50s} -> {code}: {msg}")
        return f"FAIL ({code})", msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()

    with open(PRICE_TABLE) as f:
        table = yaml.safe_load(f)

    client = boto3.client("bedrock-runtime", region_name=args.region)
    print(f"region={args.region}\n--- phase 1: bare model_id ---")

    phase1 = {}
    for key, entry in table["models"].items():
        status, _ = _try_converse(client, key, entry["model_id"])
        phase1[key] = status

    needs_profile = {k: v for k, v in phase1.items() if v.startswith("FAIL") and k in PROFILE_CANDIDATES}
    phase2 = {}
    if needs_profile:
        print("\n--- phase 2: inference-profile candidates for models phase 1 failed on ---")
        for key in needs_profile:
            phase2[key] = []
            for candidate_id in PROFILE_CANDIDATES[key]:
                status, _ = _try_converse(client, f"{key} ({candidate_id.split('.')[0]}.)", candidate_id)
                phase2[key].append((candidate_id, status))

    print("\nSummary:")
    for key, status in phase1.items():
        print(f"  {key}: bare={status}", end="")
        if key in phase2:
            working = [cid for cid, s in phase2[key] if s == "PASS"]
            print(f", working profile(s)={working or 'NONE -- investigate'}")
        else:
            print()

    n_unresolved = sum(1 for key, status in phase1.items()
                       if status.startswith("FAIL") and not any(s == "PASS" for _, s in phase2.get(key, [])))
    if n_unresolved:
        print(f"\n{n_unresolved} model(s) have NO working id (bare or profile) -- needs investigation "
             f"before that rung can be used at all.")
    else:
        print("\nEvery model has at least one confirmed-working id. Paste this output back so "
             "configs/bedrock_prices.yaml can be updated with the final model_id/profile per rung.")


if __name__ == "__main__":
    main()
