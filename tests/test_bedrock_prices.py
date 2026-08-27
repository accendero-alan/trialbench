"""P15 acceptance test (wave2-start-plan.md): the pinned price table fails
closed on an unpriced model, and batch pricing applies the right multiplier.

Run:  python tests/test_bedrock_prices.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bedrock.prices import UnpricedModelError, cost_for, load_price_table  # noqa: E402


def test_real_price_table_loads_and_prices_a_known_model():
    table = load_price_table()
    assert "amazon.nova-lite" in table["models"]
    sync_cost = cost_for("amazon.nova-lite", 750, 20, "sync", table)
    # 750/1e6 * 0.06 + 20/1e6 * 0.24
    assert abs(sync_cost - 0.0000498) < 1e-9, sync_cost
    print("real price table OK: nova-lite sync cost =", sync_cost)


def test_batch_multiplier_applied():
    table = load_price_table()
    sync_cost = cost_for("amazon.nova-lite", 750, 20, "sync", table)
    batch_cost = cost_for("amazon.nova-lite", 750, 20, "batch", table)
    assert abs(batch_cost - sync_cost * 0.5) < 1e-12, (sync_cost, batch_cost)
    print("batch multiplier OK:", sync_cost, "->", batch_cost)


def test_lookup_by_concrete_model_id_or_table_key():
    table = load_price_table()
    by_key = cost_for("amazon.nova-lite", 750, 20, "sync", table)
    concrete_id = table["models"]["amazon.nova-lite"]["model_id"]
    by_id = cost_for(concrete_id, 750, 20, "sync", table)
    assert by_key == by_id


def test_unpriced_model_fails_closed():
    table = load_price_table()
    try:
        cost_for("no-such-model-in-the-table", 100, 10, "sync", table)
        raise AssertionError("expected UnpricedModelError")
    except UnpricedModelError:
        print("fail-closed OK: unpriced model raises rather than returning 0")


def test_bad_service_tier_rejected():
    table = load_price_table()
    try:
        cost_for("amazon.nova-lite", 100, 10, "sometimes", table)
        raise AssertionError("expected ValueError for an invalid service_tier")
    except ValueError:
        pass


if __name__ == "__main__":
    test_real_price_table_loads_and_prices_a_known_model()
    test_batch_multiplier_applied()
    test_lookup_by_concrete_model_id_or_table_key()
    test_unpriced_model_fails_closed()
    test_bad_service_tier_rejected()
    print("bedrock prices tests passed")
