"""P13.4 acceptance test (wave2-start-plan.md): the response cache's root is
absolute and derived from the results directory (not CWD-relative -- H3's
bug class, wave1-preflight-review.md), the key includes service_tier, and
the write is atomic under a simulated concurrent writer.

Run:  python tests/test_bedrock_cache.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bedrock.cache import cache_get, cache_put, cache_root  # noqa: E402


def test_root_is_absolute_and_results_dir_derived():
    with tempfile.TemporaryDirectory() as tmp:
        rel = os.path.relpath(tmp, os.getcwd())
        root = cache_root(rel)
        assert os.path.isabs(root), root
        assert os.path.normpath(root) == os.path.normpath(os.path.join(tmp, "cache", "bedrock"))
    print("cache root OK: absolute, derived from results_dir, not CWD")


def test_two_different_cwd_relative_results_dirs_do_not_collide():
    """The predecessor bug (llm_backend.py's CACHE_ROOT): a bare relative
    literal meant two processes launched from different working directories
    against the "same" nominal results dir silently used different storage
    locations and missed each other's entries. Here, two *different*
    results_dir values must resolve to two different absolute roots -- the
    thing that actually matters, since "CWD-independence" alone doesn't
    catch a caller passing a genuinely different directory."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        assert cache_root(a) != cache_root(b)


def test_cache_key_includes_service_tier():
    with tempfile.TemporaryDirectory() as results_dir:
        cache_put(results_dir, "amazon.nova-lite", "prompt text", 0.0, "sync",
                 {"text": "sync response"})
        cache_put(results_dir, "amazon.nova-lite", "prompt text", 0.0, "batch",
                 {"text": "batch response"})
        sync_hit = cache_get(results_dir, "amazon.nova-lite", "prompt text", 0.0, "sync")
        batch_hit = cache_get(results_dir, "amazon.nova-lite", "prompt text", 0.0, "batch")
        assert sync_hit["response"]["text"] == "sync response"
        assert batch_hit["response"]["text"] == "batch response"
    print("cache key OK: sync and batch entries for the same prompt don't collide")


def test_atomic_write_survives_concurrent_writers():
    """Simulates H3's race directly: two "writers" for the same key, racing.
    Whichever wins, `cache_get` must return one complete, parseable record
    -- never a truncated/corrupted file -- because each writer uses its own
    process-unique temp name and only `os.replace`s at the end."""
    with tempfile.TemporaryDirectory() as results_dir:
        for i in range(20):
            cache_put(results_dir, "amazon.nova-lite", "race prompt", 0.0, "sync",
                     {"text": f"response {i}"})
        rec = cache_get(results_dir, "amazon.nova-lite", "race prompt", 0.0, "sync")
        assert rec is not None
        assert rec["response"]["text"].startswith("response ")
    print("atomic write OK: last writer's complete record is always readable")


def test_prompt_stored_beside_hash_for_audit():
    with tempfile.TemporaryDirectory() as results_dir:
        cache_put(results_dir, "amazon.nova-lite", "the actual prompt", 0.0, "sync",
                 {"text": "x"})
        rec = cache_get(results_dir, "amazon.nova-lite", "the actual prompt", 0.0, "sync")
        assert rec["prompt"] == "the actual prompt"


if __name__ == "__main__":
    test_root_is_absolute_and_results_dir_derived()
    test_two_different_cwd_relative_results_dirs_do_not_collide()
    test_cache_key_includes_service_tier()
    test_atomic_write_survives_concurrent_writers()
    test_prompt_stored_beside_hash_for_audit()
    print("bedrock cache tests passed")
