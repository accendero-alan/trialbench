"""deploy/sharepoint_transfer.py acceptance test, against a mocked
urllib.request.urlopen -- no live Azure app registration or SharePoint site
needed to check the request-building logic (token flow, site/drive
resolution, the 4MiB simple-vs-chunked upload branch, chunk-size rounding,
and that missing env vars fail with a clear message rather than a raw
traceback).

Run:  python tests/test_sharepoint_transfer.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deploy.sharepoint_transfer as sp  # noqa: E402

ENV = {
    "AZURE_TENANT_ID": "tenant-123", "AZURE_CLIENT_ID": "client-abc",
    "AZURE_CLIENT_SECRET": "shh", "SHAREPOINT_SITE_HOSTNAME": "contoso.sharepoint.com",
    "SHAREPOINT_SITE_PATH": "/sites/Wave2",
}


class _FakeResponse:
    """Mimics enough of ``http.client.HTTPResponse`` for both call sites in
    sharepoint_transfer.py: ``resp.read()`` (whole body, used for JSON Graph
    responses) and ``shutil.copyfileobj(resp, ...)``'s chunked
    ``resp.read(length)`` (used for streaming a download)."""

    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self.body = body
        self._pos = 0
        self.status = status
        self.headers = headers or {}

    def read(self, length: int | None = None):
        if length is None:
            length = len(self.body) - self._pos
        chunk = self.body[self._pos:self._pos + length]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _json_resp(obj, status=200):
    return _FakeResponse(json.dumps(obj).encode("utf-8"), status=status)


def test_missing_env_var_fails_clean_not_raw_traceback():
    with mock.patch.dict(os.environ, {}, clear=True):
        try:
            sp.get_access_token()
            raise AssertionError("expected SharePointTransferError")
        except sp.SharePointTransferError as e:
            assert "AZURE_TENANT_ID" in str(e)
    print("missing-env-var OK: clean SharePointTransferError, not a raw traceback")


def test_token_and_site_and_drive_resolution():
    calls = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        if "login.microsoftonline.com" in url:
            assert b"client_secret=shh" in req.data  # secret goes in the body, not the URL
            return _json_resp({"access_token": "fake-token-xyz"})
        if "/sites/contoso.sharepoint.com:" in url:
            return _json_resp({"id": "site-id-1"})
        if url.endswith("/sites/site-id-1/drive"):
            return _json_resp({"id": "drive-id-1"})
        raise AssertionError(f"unexpected URL: {url}")

    with mock.patch.dict(os.environ, ENV, clear=True), \
        mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        token = sp.get_access_token()
        assert token == "fake-token-xyz"
        site_id = sp.resolve_site_id(token)
        assert site_id == "site-id-1"
        drive_id = sp.resolve_drive_id(token, site_id)
        assert drive_id == "drive-id-1"
    assert any("login.microsoftonline.com" in c for c in calls)
    print("token/site/drive resolution OK:", calls)


def test_small_file_uses_simple_upload_not_chunked_session():
    calls = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append((req.get_method(), url))
        if "login.microsoftonline.com" in url:
            return _json_resp({"access_token": "t"})
        if url.endswith("/sites/contoso.sharepoint.com:/sites/Wave2"):
            return _json_resp({"id": "site-id-1"})
        if url.endswith("/drive"):
            return _json_resp({"id": "drive-id-1"})
        if url.endswith(":/content"):
            assert req.get_method() == "PUT"
            return _FakeResponse(b"{}")
        if "createUploadSession" in url:
            raise AssertionError("a small file must not open an upload session")
        raise AssertionError(f"unexpected URL: {url}")

    with tempfile.TemporaryDirectory() as tmp:
        small_path = os.path.join(tmp, "small.txt")
        with open(small_path, "wb") as f:
            f.write(b"x" * 1000)  # well under the 4MiB simple-upload limit

        with mock.patch.dict(os.environ, ENV, clear=True), \
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            sp.upload_file(small_path, "small.txt")
    puts = [c for c in calls if c[0] == "PUT"]
    assert len(puts) == 1, calls
    print("small-file upload OK: one simple PUT, no upload session")


def test_large_file_uses_chunked_upload_session():
    uploaded_ranges = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if "login.microsoftonline.com" in url:
            return _json_resp({"access_token": "t"})
        if url.endswith("/sites/contoso.sharepoint.com:/sites/Wave2"):
            return _json_resp({"id": "site-id-1"})
        if url.endswith("/drive"):
            return _json_resp({"id": "drive-id-1"})
        if "createUploadSession" in url:
            return _json_resp({"uploadUrl": "https://upload.example/session-abc"})
        if url == "https://upload.example/session-abc":
            uploaded_ranges.append(req.headers.get("Content-range") or req.headers.get("Content-Range"))
            return _FakeResponse(b"{}")
        raise AssertionError(f"unexpected URL: {url}")

    with tempfile.TemporaryDirectory() as tmp:
        big_path = os.path.join(tmp, "big.bin")
        chunk = 320 * 1024  # smallest legal chunk size
        total = chunk * 2 + 100  # forces three chunks, last one partial
        with open(big_path, "wb") as f:
            f.write(b"\x00" * total)

        with mock.patch.dict(os.environ, ENV, clear=True), \
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), \
            mock.patch.object(sp, "SIMPLE_UPLOAD_MAX_BYTES", 100):  # force the chunked path
                                                                     # without writing a real >4MiB file
            sp.upload_file(big_path, "big.bin", chunk_size=chunk)

    assert len(uploaded_ranges) == 3, uploaded_ranges
    assert uploaded_ranges[0] == f"bytes 0-{chunk - 1}/{total}"
    assert uploaded_ranges[-1] == f"bytes {2 * chunk}-{total - 1}/{total}"
    print("large-file chunked upload OK:", uploaded_ranges)


def test_download_streams_to_atomic_temp_file():
    payload = b"downloaded content" * 100

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if "login.microsoftonline.com" in url:
            return _json_resp({"access_token": "t"})
        if url.endswith("/sites/contoso.sharepoint.com:/sites/Wave2"):
            return _json_resp({"id": "site-id-1"})
        if url.endswith("/drive"):
            return _json_resp({"id": "drive-id-1"})
        if url.endswith(":/content"):
            return _FakeResponse(payload)
        raise AssertionError(f"unexpected URL: {url}")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "sub", "out.bin")
        with mock.patch.dict(os.environ, ENV, clear=True), \
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            sp.download_file("remote/out.bin", out_path)
        with open(out_path, "rb") as f:
            assert f.read() == payload
        assert not os.path.exists(out_path + f".tmp-{os.getpid()}")  # temp file cleaned up
    print("download OK: atomic write, correct bytes")


def test_download_http_error_cleans_up_temp_file():
    import io

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if "login.microsoftonline.com" in url:
            return _json_resp({"access_token": "t"})
        if url.endswith("/sites/contoso.sharepoint.com:/sites/Wave2"):
            return _json_resp({"id": "site-id-1"})
        if url.endswith("/drive"):
            return _json_resp({"id": "drive-id-1"})
        if url.endswith(":/content"):
            body = io.BytesIO(b'{"error":{"message":"not found"}}')
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, body)
        raise AssertionError(f"unexpected URL: {url}")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "missing.bin")
        with mock.patch.dict(os.environ, ENV, clear=True), \
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            try:
                sp.download_file("remote/missing.bin", out_path)
                raise AssertionError("expected SharePointTransferError")
            except sp.SharePointTransferError as e:
                assert "404" in str(e)
        assert not os.path.exists(out_path)
        assert not any(f.startswith("missing.bin.tmp-") for f in os.listdir(tmp))
    print("download HTTP error OK: no partial file left behind")


if __name__ == "__main__":
    test_missing_env_var_fails_clean_not_raw_traceback()
    test_token_and_site_and_drive_resolution()
    test_small_file_uses_simple_upload_not_chunked_session()
    test_large_file_uses_chunked_upload_session()
    test_download_streams_to_atomic_temp_file()
    test_download_http_error_cleans_up_temp_file()
    print("sharepoint_transfer tests passed")
