"""Shared helpers for the Part 2 test-plan experiment scripts
(newsletter-part2-test-plan.md). Every test writes a JSON artifact per the
plan's standing rule #1 -- these helpers keep that mechanical part consistent
across T1, T7, T8, T9, T3, T16-T19 instead of reimplementing it each time.
"""
from __future__ import annotations

import json
import os
import subprocess
import time


def git_sha() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root
        ).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root
        ).decode().strip())
    except Exception:  # noqa: BLE001
        dirty = False
    return f"{sha}-dirty" if dirty else sha


def write_artifact(path: str, obj: dict) -> None:
    """Atomic JSON write -- a kill mid-write leaves a stray .tmp, never a
    truncated artifact that could be mistaken for a finished one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.secs = time.time() - self.t0
