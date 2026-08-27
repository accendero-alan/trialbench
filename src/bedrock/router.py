"""P13.9 (wave2-start-plan.md): the prompt router client.

Routers are immutable (no ``UpdatePromptRouter``), so a threshold sweep is N
routers created up front from a checked-in spec, not N mutations of one
router. Every routed call must return
``trace.promptRouter.invokedModelId`` -- its absence is a **hard error**,
not a fallback, because it means the attribution channel changed and the
experiment is void (the plan's own framing, not a defensive add-on).
"""
from __future__ import annotations

from dataclasses import dataclass

from .client import BedrockClient, ConverseResult


class RouterAttributionError(RuntimeError):
    """``trace.promptRouter.invokedModelId`` was absent from a routed
    Converse response. Per the plan, this means the attribution channel
    changed and the experiment is void -- callers must not degrade to
    "assume the requested member was invoked"."""


@dataclass
class RouterSpec:
    name: str
    member_model_ids: list
    fallback_model_id: str
    response_quality_difference: float


@dataclass
class RouterHandle:
    name: str
    arn: str
    response_quality_difference: float


def create_routers(control_client, specs: list) -> list:
    """``control_client`` is ``boto3.client("bedrock")`` (the control plane,
    not ``bedrock-runtime``) or a fake injected for testing. One
    ``create_prompt_router`` call per spec -- a ``responseQualityDifference``
    sweep is N routers, deterministically named, created up front."""
    handles = []
    for spec in specs:
        resp = control_client.create_prompt_router(
            promptRouterName=spec.name,
            models=[{"modelArn": m} for m in spec.member_model_ids],
            fallbackModel={"modelArn": spec.fallback_model_id},
            routingCriteria={"responseQualityDifference": spec.response_quality_difference},
        )
        arn = resp.get("promptRouterArn") or resp.get("promptRouter", {}).get("arn")
        if not arn:
            raise RuntimeError(f"create_prompt_router for {spec.name!r} returned no ARN: {resp}")
        handles.append(RouterHandle(name=spec.name, arn=arn,
                                    response_quality_difference=spec.response_quality_difference))
    return handles


def route_call(client: BedrockClient, router_arn: str, prompt: str, temperature: float = 0.0,
              max_tokens: int = 32) -> ConverseResult:
    """Synchronous only -- routers cannot be batched
    (``CreateModelInvocationJob``'s ``modelId`` admits foundation models,
    inference profiles and custom models, not prompt-router ARNs). Returns
    the same :class:`~src.bedrock.client.ConverseResult` a direct call does,
    but raises :class:`RouterAttributionError` if the router didn't report
    which member it invoked."""
    result = client.converse(router_arn, prompt, temperature=temperature, max_tokens=max_tokens)
    trace = result.raw_response.get("trace", {}).get("promptRouter", {})
    if "invokedModelId" not in trace:
        raise RouterAttributionError(
            f"router {router_arn} returned no trace.promptRouter.invokedModelId -- the "
            f"attribution channel this experiment depends on is gone. Do not substitute an "
            f"assumption; stop and investigate (P13.9)."
        )
    return result
