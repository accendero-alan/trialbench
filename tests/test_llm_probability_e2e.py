"""P13.1 acceptance test (wave2-start-plan.md), run against a fake
llama-server rather than a real GPU/model -- the CLI/config/resume-guard
wiring is what this test is checking, not model quality, and it should pass
without a pinned model or a running GPU (neither is available in CI, and W1's
model choice is a separate, still-open decision).

The acceptance check, verbatim:
  python -m src.run_benchmark --methods llm_probability --llm-arm L1
    --llm-model <id> --tasks mortality_rate_yn --phases Phase2
    --max-test-rows 20 --results-dir results_llm_smoke
  produces 20 predictions and a run record carrying llm_arm: "L1";
  re-running the same directory with --llm-arm L2 exits non-zero on the
  first cell.

Run:  python tests/test_llm_probability_e2e.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.llm_backend import CACHE_ROOT, _safe_model_dirname  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_MODEL = "test-fixture-fake-model-8b"  # namespaced so cleanup can't touch a real model's cache


class _FakeLlamaServerHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get("n_predict") == 1:
            resp = {
                "completion_probabilities": [{"probs": [
                    {"tok_str": " Yes", "logprob": -0.4},
                    {"tok_str": " No", "logprob": -1.2},
                ]}],
                "timings": {"prompt_n": 50, "predicted_n": 1},
            }
        else:
            resp = {"content": '{"probability": 55}', "timings": {"prompt_n": 50, "predicted_n": 6}}
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _start_fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeLlamaServerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _run_cli(*extra_args):
    return subprocess.run(
        [sys.executable, "-m", "src.run_benchmark", *extra_args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )


def _fake_model_cache_dir():
    return os.path.join(REPO_ROOT, CACHE_ROOT, _safe_model_dirname(FAKE_MODEL))


def test_llm_probability_smoke_and_resume_guard():
    # The response cache (P13.4) is deliberately shared across runs/results
    # dirs -- that's the point of it -- so a prior run of this same test
    # would otherwise serve every prompt from cache and never exercise a
    # live call. FAKE_MODEL is namespaced so this only ever touches this
    # test's own cache entries, never a real model's.
    shutil.rmtree(_fake_model_cache_dir(), ignore_errors=True)
    server, base_url = _start_fake_server()
    try:
        with tempfile.TemporaryDirectory() as results_dir:
            proc = _run_cli(
                "--methods", "llm_probability", "--llm-arm", "L1", "--llm-model", FAKE_MODEL,
                "--llm-base-url", base_url, "--tasks", "mortality_rate_yn", "--phases", "Phase2",
                "--max-test-rows", "20", "--results-dir", results_dir,
            )
            assert proc.returncode == 0, f"first run failed:\n{proc.stdout}\n{proc.stderr}"

            pred_files = glob.glob(os.path.join(results_dir, "predictions", "mortality_rate_yn", "Phase2", "*.parquet"))
            assert pred_files, f"no predictions written under {results_dir}"
            import pandas as pd
            df = pd.read_parquet(pred_files[0])
            n_test = (df["split"] == "test").sum()
            assert n_test == 20, f"expected 20 test predictions, got {n_test}"

            run_files = glob.glob(os.path.join(results_dir, "runs", "mortality_rate_yn__Phase2__llm_probability__*.json"))
            assert run_files, "no run record written"
            with open(run_files[0]) as f:
                rec = json.load(f)
            assert rec["llm_arm"] == "L1", rec
            assert rec["llm_model"] == FAKE_MODEL, rec
            assert rec["llm_meter"]["dollars"] == 0.00, rec["llm_meter"]
            assert rec["llm_meter"]["calls"] > 0, rec["llm_meter"]
            print("first run OK:", run_files[0])

            # B3 guard: resuming the same directory with a different --llm-arm
            # must abort on the first cell, not silently mix arms.
            proc2 = _run_cli(
                "--methods", "llm_probability", "--llm-arm", "L2", "--llm-model", FAKE_MODEL,
                "--llm-base-url", base_url, "--tasks", "mortality_rate_yn", "--phases", "Phase2",
                "--max-test-rows", "20", "--results-dir", results_dir,
            )
            assert proc2.returncode != 0, f"expected non-zero exit on llm_arm mismatch, got 0:\n{proc2.stdout}"
            assert "ABORT" in (proc2.stdout + proc2.stderr), proc2.stdout + proc2.stderr
            print("B3 guard OK: resume with different --llm-arm aborted")
    finally:
        server.shutdown()
        shutil.rmtree(_fake_model_cache_dir(), ignore_errors=True)


if __name__ == "__main__":
    test_llm_probability_smoke_and_resume_guard()
    print("llm_probability e2e test passed")
