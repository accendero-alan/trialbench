"""W1.2's remaining open item: does each ladder model (+ Nova Pro, the
router member) accept a direct Converse call on its bare model_id, or does
it need a us./global. inference-profile ARN instead? One minimal call per
model (max_tokens=5, trivial prompt) -- cheap, and the only way to answer
this; a catalogue listing (list-foundation-models) doesn't say.

Run from the repo root, with AWS credentials resolved and boto3 installed
(both already true on the EC2 instance per the requirements.txt install):

    python deploy/smoke_test_converse.py
    python deploy/smoke_test_converse.py --region us-west-2   # override if needed

Reads model ids straight from configs/bedrock_prices.yaml so this can't
drift from what W1.2 already confirmed live. Prints one PASS/FAIL line per
model; on FAIL, prints the exact error code/message, since
"ValidationException: ... use an inference profile" vs. a throttling/access
error need different fixes.
"""
from __future__ import annotations

import argparse
import os

import boto3
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRICE_TABLE = os.path.join(ROOT, "configs", "bedrock_prices.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()

    with open(PRICE_TABLE) as f:
        table = yaml.safe_load(f)

    client = boto3.client("bedrock-runtime", region_name=args.region)
    print(f"region={args.region}\n")

    results = {}
    for key, entry in table["models"].items():
        model_id = entry["model_id"]
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "Reply with only the word OK."}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 5},
            )
            text = resp.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
            usage = resp.get("usage", {})
            print(f"PASS  {key:35s} {model_id:45s} -> {text!r} "
                 f"(in={usage.get('inputTokens')}, out={usage.get('outputTokens')})")
            results[key] = "PASS"
        except Exception as e:  # noqa: BLE001 -- diagnostic script, want every failure mode printed
            code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
            msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
            print(f"FAIL  {key:35s} {model_id:45s} -> {code}: {msg}")
            results[key] = f"FAIL ({code})"

    print("\nSummary:")
    for key, status in results.items():
        print(f"  {key}: {status}")
    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    if n_fail:
        print(f"\n{n_fail} model(s) failed on the bare model_id -- likely need an inference-profile "
             f"ARN (e.g. 'us.{next(iter(results))}' or 'global.<id>') instead. Check the error code: "
             f"ValidationException mentioning 'inference profile' confirms it; anything else "
             f"(AccessDenied, ResourceNotFound) is a different problem.")


if __name__ == "__main__":
    main()
