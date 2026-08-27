"""P13.5 (wave2-start-plan.md): the usage meter -- the primary measurement
instrument for T30, so it gets the same care ``src/eval/metrics.py`` got.

Records cost on two bases, per the plan:

- **realized**: what this run actually cost, at whatever service tier
  (sync/batch) it actually used.
- **normalized**: the same token counts priced at on-demand list, so arms
  that ran batched and arms that ran synchronous are comparable on one axis.
  The router can't be batched and the ladder can, so only the normalized
  basis is comparable across every arm -- T30/T31 plot both and say which is
  which, rather than letting normalization quietly hide the router's
  structural cost penalty.

Both figures come from :mod:`src.bedrock.prices`, never a literal -- a model
missing from the price table fails the run (P15), it does not read here as
free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .prices import cost_for, routing_fee_for


@dataclass
class Meter:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_secs: float = 0.0
    throttle_count: int = 0
    routing_requests: int = 0     # calls that went through a router (vs. a direct model call)

    def record_call(self, input_tokens: int, output_tokens: int, wall_clock_secs: float,
                    throttle_count: int = 0, routed: bool = False) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.wall_clock_secs += wall_clock_secs
        self.throttle_count += throttle_count
        if routed:
            self.routing_requests += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.calls + self.cache_hits
        return self.cache_hits / total if total else 0.0

    def summary(self, price_table: dict, model_id: str, service_tier: str) -> dict:
        """``service_tier`` is the tier this meter's calls were actually
        billed at ("realized"); "normalized" always prices the same tokens
        at "sync" (on-demand list), regardless of what actually ran."""
        realized = cost_for(model_id, self.input_tokens, self.output_tokens, service_tier, price_table)
        normalized = cost_for(model_id, self.input_tokens, self.output_tokens, "sync", price_table)
        routing_fee = routing_fee_for(self.routing_requests, price_table) if self.routing_requests else 0.0
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_clock_secs": round(self.wall_clock_secs, 2),
            "throttle_count": self.throttle_count,
            "service_tier": service_tier,
            "dollars_realized": round(realized + routing_fee, 6),
            "dollars_normalized": round(normalized + routing_fee, 6),
            "routing_fee_usd": round(routing_fee, 6),
        }
