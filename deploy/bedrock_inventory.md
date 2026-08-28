# Bedrock inventory (W1, wave2-start-plan.md)

Generated 2026-08-28 16:41 UTC by `deploy/w1_bedrock_inventory.py --region us-west-2`. Every section below is independently dated -- re-run any subset to refresh just that section without invalidating the rest.

---

### Item 1 -- account, region, execution role (2026-08-28 16:41 UTC)

- Account: `199751540033`
- Caller ARN: `arn:aws:sts::199751540033:assumed-role/BedrockCliAccessRole/i-0b3ac76e78e5cf1c2`
- Region checked: `us-west-2`
- SKIPPED role policy introspection (likely missing `iam:List*` on the execution role itself): An error occurred (AccessDenied) when calling the ListAttachedRolePolicies operation: User: arn:aws:sts::199751540033:assumed-role/BedrockCliAccessRole/i-0b3ac76e78e5cf1c2 is not authorized to perform: iam:ListAttachedRolePolicies on resource: role BedrockCliAccessRole because no identity-based policy allows the iam:ListAttachedRolePolicies action. Go to https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-1#/authorization-details/ejsk06qtimkkubln9lbjmt1h for complete details, or call the GetRequestAuthorizationDetails API with the following authorization id: ejsk06qtimkkubln9lbjmt1h

Required per P13.11: `bedrock:InvokeModel`, `bedrock:CreateModelInvocationJob`, `bedrock:GetModelInvocationJob`, `bedrock:CreatePromptRouter`, `s3:GetObject`/`s3:PutObject` on the batch bucket -- **plus, confirmed live 2026-08-28 (Claude Opus 4.5's first synchronous Converse call from this exact role failed AccessDeniedException), `aws-marketplace:ViewSubscriptions`, `aws-marketplace:Subscribe`, `aws-marketplace:Unsubscribe`** -- every third-party model needs these on first invocation per account (AWS's own 'automatic model access' subscription flow), not just the batch execution role P13.8's probe found this on originally. Two more per-account prerequisites this role's policy cannot fix: Anthropic models specifically require completing the 'First Time Use' (FTU) form once per account before first invocation (Bedrock console, separate from IAM), and the account needs a valid AWS Marketplace payment method on file. Diff the policy above against this full list by hand.

---

### Item 4 -- quotas (RPM / TPM, on-demand and batch) (2026-08-28 16:41 UTC)

SKIPPED: An error occurred (AccessDeniedException) when calling the ListAWSDefaultServiceQuotas operation: User: arn:aws:sts::199751540033:assumed-role/BedrockCliAccessRole/i-0b3ac76e78e5cf1c2 is not authorized to perform: servicequotas:ListAWSDefaultServiceQuotas because no identity-based policy allows the servicequotas:ListAWSDefaultServiceQuotas action

---

### Item 6 -- router feasibility, item 7 (partial) -- one real routed call (2026-08-28 16:41 UTC)

**wave2-nova-lite-pro-v1** (members=['amazon.nova-lite-v1:0', 'amazon.nova-pro-v1:0'], responseQualityDifference=0.5)
  - FAIL (AccessDeniedException): User: arn:aws:sts::199751540033:assumed-role/BedrockCliAccessRole/i-0b3ac76e78e5cf1c2 is not authorized to perform: bedrock:CreatePromptRouter on resource: arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-lite-v1:0 because no identity-based policy allows the bedrock:CreatePromptRouter action

**wave2-llama3-fallback-v1** (members=['meta.llama3-1-8b-instruct-v1:0', 'meta.llama3-1-70b-instruct-v1:0'], responseQualityDifference=0.5)
  - FAIL (AccessDeniedException): User: arn:aws:sts::199751540033:assumed-role/BedrockCliAccessRole/i-0b3ac76e78e5cf1c2 is not authorized to perform: bedrock:CreatePromptRouter on resource: arn:aws:bedrock:us-west-2::foundation-model/meta.llama3-1-8b-instruct-v1:0 because no identity-based policy allows the bedrock:CreatePromptRouter action

No router spec succeeded -- item 7's live-run half cannot proceed until one does. Re-read the current supported-member table before retrying with different ids: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html

---

### Item 8 -- logprob availability, DeepSeek V3.2 / Llama 4 Maverick (2026-08-28 16:41 UTC)

**deepseek.v3-2** (`deepseek.v3.2`)
  - call with additionalModelRequestFields={'logprobs': True} succeeded. Full response (inspect for any logprob-shaped field):
    ```
    {'ResponseMetadata': {'RequestId': 'ba9252fe-b4e3-43e3-9169-8aa42c1a808c', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Fri, 28 Aug 2026 16:41:19 GMT', 'content-type': 'application/json', 'content-length': '203', 'connection': 'keep-alive', 'x-amzn-requestid': 'ba9252fe-b4e3-43e3-9169-8aa42c1a808c'}, 'RetryAttempts': 0}, 'output': {'message': {'role': 'assistant', 'content': [{'text': 'OK'}]}}, 'stopReason': 'end_turn', 'usage': {'inputTokens': 11, 'outputTokens': 2, 'totalTokens': 13}, 'metrics': {'latencyMs': 166}}
    ```
  - call with additionalModelRequestFields={'return_logprobs': True} succeeded. Full response (inspect for any logprob-shaped field):
    ```
    {'ResponseMetadata': {'RequestId': '3d08a002-c949-45f4-92a1-903d65cf315c', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Fri, 28 Aug 2026 16:41:19 GMT', 'content-type': 'application/json', 'content-length': '203', 'connection': 'keep-alive', 'x-amzn-requestid': '3d08a002-c949-45f4-92a1-903d65cf315c'}, 'RetryAttempts': 0}, 'output': {'message': {'role': 'assistant', 'content': [{'text': 'OK'}]}}, 'stopReason': 'end_turn', 'usage': {'inputTokens': 11, 'outputTokens': 2, 'totalTokens': 13}, 'metrics': {'latencyMs': 239}}
    ```

**meta.llama4-maverick-17b** (`us.meta.llama4-maverick-17b-instruct-v1:0`)
  - call with additionalModelRequestFields={'logprobs': True} -> ValidationException: The model returned the following errors: Malformed input request: #: extraneous key [logprobs] is not permitted, please reformat your input and try again.
  - call with additionalModelRequestFields={'return_logprobs': True} -> ValidationException: The model returned the following errors: Malformed input request: #: extraneous key [return_logprobs] is not permitted, please reformat your input and try again.

Record the answer either way in `src/bedrock/client.py`'s module docstring and `wave2-start-plan.md` -- a clean fail on both guessed keys is still an answer ("no logprob path found"), not a blocker; do not leave this re-run indefinitely chasing parameter names without checking each provider's own Bedrock API reference first.

---

### Item 3 -- live console prices (MANUAL, not automatable) (2026-08-28 16:41 UTC)

The public pricing page (https://aws.amazon.com/bedrock/pricing/) renders its tables client-side and cannot be fetched by a script or an API call. Do this by hand, once, right before P15 is closed out:

1. Open the Bedrock console -> Model catalog / Model providers pricing view, in the account this ran in (see item 1 above), region confirmed in `configs/bedrock_prices.yaml` (currently `us-west-2`).
2. For each of the 7 rows in `configs/bedrock_prices.yaml` (5 ladder models + Nova Pro + the routing fee), read the live input/output $-per-1M-token price.
3. Update `input_per_1m_usd`/`output_per_1m_usd`, set `verified: true`, and bump `read_on` to today's date, for every row whose price matches or is corrected.
4. If a price has moved by an order of magnitude from what's currently recorded, STOP and reselect the rung before any billable grid call -- per wave2-start-plan.md §8, risk 4.

---

### Item 7 -- routing fee, remaining manual half (2026-08-28 16:41 UTC)

The automated half (item 6, above) issues one real routed call so a genuine fee line exists in the bill. To close item 7 out:

1. Cross-check the fee against the AWS Pricing Calculator (https://calculator.aws/) for Bedrock Intelligent Prompt Routing, same region.
2. 24h+ after the probe call above, check Cost Explorer for the routing-fee line item and confirm it matches `routing_fee_per_1k_requests_usd` in `configs/bedrock_prices.yaml` (currently $1.00/1k, unverified).
3. Set `routing_fee_verified: true` once confirmed by either channel.

---

### Item 5 -- batch eligibility (2026-08-28 16:41 UTC)

Not run this pass -- pass `--run-batch-probe --s3-bucket <bucket> --batch-role-arn <arn>` once the batch bucket and execution role exist. Do this after item 4's quota check, not before (a probe below the true minimum fails for the wrong reason).
