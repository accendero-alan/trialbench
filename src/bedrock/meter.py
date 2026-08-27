"""P13.5 (wave2-start-plan.md): the usage meter -- the primary measurement
instrument for T30, so it gets the same care ``src/eval/metrics.py`` got.

Records cost on two bases, per the plan:

- **realized**: what this run actually cost. Computed **per call, from the
  service tier that call was actually billed at** (``tokens_by_tier``),
  never from a single tier asserted for the whole run -- that was the
  defect found 2026-08-27 (wave2-start-plan.md P13.8's status note):
  ``--llm-service-tier batch`` was applied as a blanket discount to
  ``summary()``'s cost figure even on a cell where every individual call
  actually ran synchronous (P13.8's batch submission was never wired into
  the elicitation path, so ``batch`` was a label nothing enforced). Now a
  call must say what it actually was when it's recorded
  (:meth:`record_call`'s ``service_tier`` argument), and ``summary()`` sums
  cost per tier from what was recorded -- a batch cell that partially falls
  back to sync below the per-model minimum record count (P13.8) prices
  correctly as a mix, not as one or the other.
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

from collections import defaultdict
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
    # tier ("sync"/"batch") -> [input_tokens, output_tokens] actually billed
    # at that tier -- the source of truth for realized cost; input_tokens/
    # output_tokens above stay as the simple running total (used for
    # "normalized" and for reporting), unaffected by tier.
    tokens_by_tier: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    def record_call(self, input_tokens: int, output_tokens: int, wall_clock_secs: float,
                    throttle_count: int = 0, routed: bool = False, service_tier: str = "sync") -> None:
        """``service_tier`` is what this specific call was actually billed
        at -- the caller must not pass through a nominally-requested tier
        that isn't what happened (e.g. a batch-below-minimum fallback call
        is a real synchronous call and must be recorded ``service_tier="sync"``
        regardless of what the cell was configured for)."""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.wall_clock_secs += wall_clock_secs
        self.throttle_count += throttle_count
        if routed:
            self.routing_requests += 1
        bucket = self.tokens_by_tier[service_tier]
        bucket[0] += input_tokens
        bucket[1] += output_tokens

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.calls + self.cache_hits
        return self.cache_hits / total if total else 0.0

    def summary(self, price_table: dict, model_id: str, requested_service_tier: str) -> dict:
        """``requested_service_tier`` is what the cell was *configured* for
        -- recorded for provenance/labeling only. It plays no part in the
        realized-cost computation; that comes entirely from
        ``tokens_by_tier``, i.e. what each call actually was."""
        realized = sum(
            cost_for(model_id, in_tok, out_tok, tier, price_table)
            for tier, (in_tok, out_tok) in self.tokens_by_tier.items()
            if in_tok or out_tok
        )
        normalized = cost_for(model_id, self.input_tokens, self.output_tokens, "sync", price_table)
        routing_fee = routing_fee_for(self.routing_requests, price_table) if self.routing_requests else 0.0
        tiers_used = {tier: {"input_tokens": in_tok, "output_tokens": out_tok}
                      for tier, (in_tok, out_tok) in self.tokens_by_tier.items() if in_tok or out_tok}
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_clock_secs": round(self.wall_clock_secs, 2),
            "throttle_count": self.throttle_count,
            "service_tier": requested_service_tier,
            "tokens_by_tier": tiers_used,
            "mixed_tier": len(tiers_used) > 1,
            "dollars_realized": round(realized + routing_fee, 6),
            "dollars_normalized": round(normalized + routing_fee, 6),
            "routing_fee_usd": round(routing_fee, 6),
        }
