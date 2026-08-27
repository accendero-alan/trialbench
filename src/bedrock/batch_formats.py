"""Per-provider ``modelInput``/output shapes for Bedrock batch inference
(``CreateModelInvocationJob``).

Batch inference does not go through Converse -- it requires each record's
``modelInput`` to match that model provider's own native ``InvokeModel``
request body, which is genuinely different per provider (unlike Converse,
which is one shape across providers by design). This module is the
provider-dispatch layer :func:`src.methods.llm.LLMProbability._predict_batch`
needs to build records and parse them back.

**Confidence varies sharply by provider, and that is stated per function,
not glossed over:**

- **DeepSeek -- CONFIRMED live 2026-08-27.** `deploy/w1_bedrock_inventory.py
  --run-batch-probe` ran a real 100-record batch job end to end (submit,
  poll, S3 output read-back, text extraction) and got 100/100 real
  responses back, extracted correctly (`'OK'`). The build/extract pair for
  this provider is verified, not just documented.
- **Amazon Nova -- format HIGH confidence but UNTESTABLE in `us-west-2`.**
  The same live probe found `amazon.nova-lite-v1:0` batch inference is not
  supported in this region at all (`ValidationException: Batch inference is
  not supported for model amazon.nova-lite-v1:0 in region us-west-2`) --
  a hard platform constraint, not an IAM or format problem. Nova Pro's job
  submitted but failed on a permissions error before reaching model
  invocation (see below), so its format is also still unverified in
  practice despite the shape itself mirroring Converse closely.
- **Anthropic, Meta Llama, Nova Pro -- blocked on permissions as of
  2026-08-27, format unverified.** All three failed before ever reaching
  model invocation: Anthropic (Opus, Haiku) with an explicit AWS Marketplace
  subscription error naming the execution role; Llama 4 Maverick and Nova
  Pro with a vaguer "Customer doesn't have permissions to invokeModel" once
  the job was already running. The common thread is the **batch execution
  role** (distinct from the CLI-caller role, which already invokes all of
  these fine synchronously) lacking `bedrock:InvokeModel` and/or AWS
  Marketplace grants of its own -- see `wave2-start-plan.md`'s P13.8 status
  note for the exact IAM additions to try. **Do not trust Llama or Anthropic's
  modelInput shape for a real paid grid pass until a job actually reaches
  model invocation and returns real output** -- a permissions failure before
  invocation proves nothing about the format either way.

A malformed ``modelInput`` fails at the **record level** in Bedrock's batch
output (or the whole job fails validation at submission, which is loud), not
silently -- :func:`src.bedrock.batch.reassemble` already reports a
record-level failure as a parse failure rather than crashing the caller, so
a wrong body for one provider degrades that provider's arm to a high
parse-failure-rate warning (`run_benchmark.py`'s existing >2% check), not a
corrupted result set for everyone.
"""
from __future__ import annotations

_SYSTEM_LESS_INSTRUCTION_SUFFIX = ""  # placeholder for symmetry if a provider ever needs different framing


def detect_provider(model_id: str) -> str:
    """``model_id`` is the concrete Bedrock id/inference-profile id (e.g.
    ``"us.anthropic.claude-opus-4-5-20251101-v1:0"``, ``"deepseek.v3.2"``),
    not the price-table key -- callers already resolve that first
    (:func:`src.bedrock.prices.resolve_model_id`)."""
    m = model_id.lower()
    if "anthropic" in m:
        return "anthropic"
    if "nova" in m:
        return "amazon_nova"
    if "llama" in m or "meta." in m:
        return "meta_llama"
    if "deepseek" in m:
        return "deepseek"
    raise ValueError(
        f"no batch modelInput/output format registered for {model_id!r} -- add a case to "
        f"src/bedrock/batch_formats.py before using --llm-service-tier batch with this model."
    )


def build_model_input(model_id: str, prompt: str, temperature: float, max_tokens: int) -> dict:
    """The ``modelInput`` dict for one batch record. One turn, one user
    message, no system prompt -- matches the sync (Converse) path's own
    shape (:func:`src.methods.llm._build_verbalized_prompt` bakes
    instructions into the single user turn already)."""
    provider = detect_provider(model_id)
    if provider == "anthropic":
        # Messages API, Bedrock invoke body -- HIGH confidence, long-stable.
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
    if provider == "amazon_nova":
        # Nova's native invoke body mirrors Converse's own shape -- HIGH confidence.
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
    if provider == "meta_llama":
        # LOW confidence -- see module docstring. Llama-on-Bedrock's raw-prompt
        # convention wraps the user turn in Llama 3's instruct template tokens;
        # Llama 4's exact template is NOT independently confirmed here.
        wrapped = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return {"prompt": wrapped, "max_gen_len": max_tokens, "temperature": temperature}
    if provider == "deepseek":
        # LOW confidence -- see module docstring. Best guess: DeepSeek's own
        # API is OpenAI-Chat-Completions-shaped; assumed preserved on Bedrock.
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    raise AssertionError(f"unreachable: detect_provider returned unhandled {provider!r}")


def extract_text(model_id: str, model_output: dict) -> str:
    """Inverse of :func:`build_model_input`: pull the generated text back
    out of one batch record's ``modelOutput``. Returns ``""`` on a shape
    that doesn't match what was expected (never raises) -- an empty string
    parses as a parse failure downstream (``_parse_verbalized`` returns
    ``None`` on non-JSON), which is the correct, non-crashing degradation
    for one malformed provider response amid a batch of thousands."""
    provider = detect_provider(model_id)
    try:
        if provider == "anthropic":
            return "".join(b.get("text", "") for b in model_output.get("content", []) if b.get("type") == "text")
        if provider == "amazon_nova":
            return "".join(b.get("text", "")
                          for b in model_output.get("output", {}).get("message", {}).get("content", []))
        if provider == "meta_llama":
            return model_output.get("generation", "")
        if provider == "deepseek":
            choices = model_output.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")
            return ""
    except (AttributeError, TypeError, IndexError):
        return ""
    return ""


def extract_usage(model_id: str, model_output: dict) -> tuple:
    """Returns ``(input_tokens, output_tokens)``, both 0 on an unrecognized
    shape (never raises -- a wrong token count degrades T30's cost figure
    for that one record, not the whole job)."""
    provider = detect_provider(model_id)
    try:
        if provider == "anthropic":
            u = model_output.get("usage", {})
            return int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
        if provider == "amazon_nova":
            u = model_output.get("usage", {})
            return int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0))
        if provider == "meta_llama":
            return int(model_output.get("prompt_token_count", 0)), int(model_output.get("generation_token_count", 0))
        if provider == "deepseek":
            u = model_output.get("usage", {})
            return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
    except (AttributeError, TypeError, ValueError):
        return 0, 0
    return 0, 0
