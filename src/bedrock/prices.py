"""P15 (wave2-start-plan.md): load the pinned price table and compute cost
from recorded tokens. Cost is never read back off a bill -- the bill doesn't
know which arm/cell/model a request belonged to at the granularity this
benchmark needs -- so every dollar figure in a Wave 2 artifact traces back to
this table plus a token count.
"""
from __future__ import annotations

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_PRICE_TABLE_PATH = os.path.join(ROOT, "configs", "bedrock_prices.yaml")


class UnpricedModelError(KeyError):
    """A cost was requested for a model_id absent from the price table.

    Fails closed by design (P15's acceptance check): a run whose model is
    missing from the table must fail, not silently record a null/zero cost
    that would then read as "free" on T30's frontier.
    """


def load_price_table(path: str = DEFAULT_PRICE_TABLE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        table = yaml.safe_load(f)
    if "models" not in table:
        raise ValueError(f"{path} has no top-level 'models' key -- not a valid price table")
    return table


def _model_entry(model_id: str, table: dict) -> dict:
    """``model_id`` may be either the table's own key (e.g.
    ``"anthropic.claude-opus-4-5"``) or the concrete Bedrock ``model_id``/
    inference-profile id recorded on that entry -- callers commonly have the
    latter (it's what actually goes in the Converse call), so both are
    checked rather than forcing every caller to know the table's key
    convention."""
    models = table.get("models", {})
    if model_id in models:
        return models[model_id]
    for entry in models.values():
        if entry.get("model_id") == model_id:
            return entry
    raise UnpricedModelError(
        f"model {model_id!r} is not in the price table ({DEFAULT_PRICE_TABLE_PATH}). "
        f"Add a priced entry before running this model -- see P15, wave2-start-plan.md."
    )


def cost_for(model_id: str, input_tokens: int, output_tokens: int, service_tier: str,
            table: dict) -> float:
    """Token cost only -- excludes any routing fee, which is a separate,
    additive line (see :func:`routing_fee_for`) per the plan's cost
    reporting rule: the fee must never disappear into the total silently."""
    entry = _model_entry(model_id, table)
    cost = (input_tokens / 1_000_000) * entry["input_per_1m_usd"] \
        + (output_tokens / 1_000_000) * entry["output_per_1m_usd"]
    if service_tier == "batch":
        cost *= entry.get("batch_multiplier", 1.0)
    elif service_tier != "sync":
        raise ValueError(f"service_tier must be 'sync' or 'batch', got {service_tier!r}")
    return cost


def routing_fee_for(n_requests: int, table: dict) -> float:
    return (n_requests / 1000) * table["routing_fee_per_1k_requests_usd"]


def resolve_model_id(name: str, table: dict) -> str:
    """Resolves a price-table key (e.g. ``"anthropic.claude-opus-4-5"``, the
    human-friendly label used in ``--llm-model``/``configs/wave2_amendment.yaml``)
    OR an already-concrete Bedrock ``model_id``/inference-profile id to the
    concrete id that must actually be passed to ``Converse``. These are NOT
    interchangeable: several ladder models (Opus 4.5, Haiku 4.5, Llama 4
    Maverick, Nova Pro -- confirmed live, 2026-08-27) reject on-demand
    invocation on their bare id and require a ``us.``/``global.``-prefixed
    inference-profile id instead, which is what ``model_id`` in the table
    holds for those rows. Sending the table *key* to the API instead of its
    ``model_id`` fails at the API, not here -- this function is what stands
    between the two. Fails closed (``UnpricedModelError``) via
    :func:`_model_entry` if ``name`` matches neither a key nor a model_id."""
    return _model_entry(name, table)["model_id"]


def is_verified(model_id: str, table: dict) -> bool:
    """False for every [W1]-flagged row (stale price, unconfirmed model id,
    or both) -- callers that report cost in an artifact should surface this
    alongside the number rather than presenting a snapshot as settled."""
    entry = _model_entry(model_id, table)
    return bool(entry.get("verified", False)) and bool(entry.get("id_verified", False))
