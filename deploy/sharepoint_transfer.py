"""Upload/download files through a SharePoint document library via the
Microsoft Graph API, authenticated as an Azure AD app registration
(client-credentials flow -- no user sign-in, no browser). Used as the
SharePoint midpoint for moving files (e.g. data/external/*, results
artifacts) between a local machine and the AWS/EC2 side, since
deploy/sync_to_ec2.sh and deploy/fetch_wave2_results.sh both go direct over
SSH/S3 and don't cover that path.

Plain stdlib (``urllib``) only, matching this repo's existing pattern for a
one-off HTTP client (the predecessor llama-server client in the removed
``src/methods/llm_backend.py`` did the same) -- no new dependency for what's
a handful of REST calls.

Credentials and site identification, all from the environment, never a CLI
argument or a file in this repo (so nothing secret ends up in shell history
or git):

    AZURE_TENANT_ID           Azure AD tenant id (GUID or verified domain)
    AZURE_CLIENT_ID           App registration (application) id
    AZURE_CLIENT_SECRET       App registration client secret
    SHAREPOINT_SITE_HOSTNAME  e.g. "contoso.sharepoint.com"
    SHAREPOINT_SITE_PATH      e.g. "/sites/YourSiteName"
    SHAREPOINT_DRIVE_NAME     optional; defaults to the site's default
                              document library if unset

The app registration needs an application (not delegated) permission grant
for ``Sites.ReadWrite.All`` (or a narrower ``Sites.Selected`` grant scoped
to just this site), admin-consented -- that's an Azure AD portal step, not
something this script can do.

Usage:
    python deploy/sharepoint_transfer.py upload   <local_path> <remote_path>
    python deploy/sharepoint_transfer.py download <remote_path> <local_path>

``remote_path`` is the file's path inside the document library, e.g.
"wave2/icd10c-tabular-April-1-2026.xml" (no leading slash).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

# Microsoft Graph's hard limit for a single PUT ("simple upload"); above
# this, a chunked upload session is required. Not a style choice -- the API
# rejects a larger simple upload outright.
SIMPLE_UPLOAD_MAX_BYTES = 4 * 1024 * 1024
# Upload-session chunk size must be a multiple of 320 KiB (Graph's
# requirement) except the final chunk. 10 MiB keeps well under Graph's
# per-request size ceiling while still being few enough requests for a
# multi-GB file (e.g. the pinned AACT snapshot).
DEFAULT_CHUNK_SIZE = 32 * 320 * 1024  # 10 MiB, a multiple of 320 KiB


class SharePointTransferError(RuntimeError):
    pass


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SharePointTransferError(
            f"{name} is not set. Required: AZURE_TENANT_ID, AZURE_CLIENT_ID, "
            f"AZURE_CLIENT_SECRET, SHAREPOINT_SITE_HOSTNAME, SHAREPOINT_SITE_PATH "
            f"(see this module's docstring)."
        )
    return value


def _request(method: str, url: str, token: str, data: bytes | None = None,
            headers: dict | None = None) -> tuple[int, bytes, dict]:
    """One HTTP call. Never logs ``token`` or any header value -- error
    messages below surface Graph's JSON error body (useful for diagnosing a
    permissions/site-path problem) but not the Authorization header."""
    req_headers = {"Authorization": f"Bearer {token}"}
    req_headers.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read()
            return resp.status, body, dict(resp.headers)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            detail = json.loads(body).get("error", {}).get("message", body.decode("utf-8", "replace"))
        except (json.JSONDecodeError, AttributeError):
            detail = body.decode("utf-8", "replace")
        raise SharePointTransferError(f"{method} {url} -> HTTP {e.code}: {detail}") from e


def get_access_token() -> str:
    tenant_id = _env("AZURE_TENANT_ID")
    client_id = _env("AZURE_CLIENT_ID")
    client_secret = _env("AZURE_CLIENT_SECRET")
    # Client-credentials grant: the secret goes in the POST body (form-
    # encoded), never a URL query string or a log line.
    payload = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }).encode("ascii")
    url = TOKEN_URL_TMPL.format(tenant_id=tenant_id)
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read()).get("error_description", "")
        raise SharePointTransferError(f"token request failed: HTTP {e.code}: {detail}") from e
    token = body.get("access_token")
    if not token:
        raise SharePointTransferError(f"token response had no access_token: {list(body.keys())}")
    return token


def resolve_site_id(token: str) -> str:
    hostname = _env("SHAREPOINT_SITE_HOSTNAME")
    site_path = _env("SHAREPOINT_SITE_PATH")
    if not site_path.startswith("/"):
        site_path = "/" + site_path
    url = f"{GRAPH_ROOT}/sites/{hostname}:{site_path}"
    _, body, _ = _request("GET", url, token)
    return json.loads(body)["id"]


def resolve_drive_id(token: str, site_id: str) -> str:
    drive_name = os.environ.get("SHAREPOINT_DRIVE_NAME")
    if not drive_name:
        _, body, _ = _request("GET", f"{GRAPH_ROOT}/sites/{site_id}/drive", token)
        return json.loads(body)["id"]
    _, body, _ = _request("GET", f"{GRAPH_ROOT}/sites/{site_id}/drives", token)
    for drive in json.loads(body).get("value", []):
        if drive.get("name") == drive_name:
            return drive["id"]
    raise SharePointTransferError(f"no drive named {drive_name!r} on this site")


def _item_url(site_id: str, drive_id: str, remote_path: str) -> str:
    quoted = urllib.parse.quote(remote_path.lstrip("/"))
    return f"{GRAPH_ROOT}/sites/{site_id}/drives/{drive_id}/root:/{quoted}"


def upload_file(local_path: str, remote_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    size = os.path.getsize(local_path)
    token = get_access_token()
    site_id = resolve_site_id(token)
    drive_id = resolve_drive_id(token, site_id)

    if size <= SIMPLE_UPLOAD_MAX_BYTES:
        with open(local_path, "rb") as f:
            data = f.read()
        _request("PUT", f"{_item_url(site_id, drive_id, remote_path)}:/content", token, data=data,
                 headers={"Content-Type": "application/octet-stream"})
        print(f"uploaded {local_path} ({size} bytes) -> {remote_path} [simple upload]")
        return

    # Large file: an upload session, PUT in chunks. Graph's session PUT is
    # unauthenticated on its own pre-signed uploadUrl -- no Authorization
    # header on those requests (sending one is harmless but unnecessary;
    # omitted here since the URL itself is the credential for this session).
    _, body, _ = _request(
        "POST", f"{_item_url(site_id, drive_id, remote_path)}:/createUploadSession", token,
        data=b"{}", headers={"Content-Type": "application/json"},
    )
    upload_url = json.loads(body)["uploadUrl"]

    with open(local_path, "rb") as f:
        start = 0
        while start < size:
            chunk = f.read(chunk_size)
            end = start + len(chunk) - 1
            req = urllib.request.Request(
                upload_url, data=chunk, method="PUT",
                headers={"Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {start}-{end}/{size}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300):
                    pass
            except urllib.error.HTTPError as e:
                raise SharePointTransferError(
                    f"chunk upload failed at byte {start}: HTTP {e.code}: "
                    f"{e.read().decode('utf-8', 'replace')}"
                ) from e
            start = end + 1
            print(f"  uploaded {start}/{size} bytes ({start * 100 // size}%)", flush=True)
    print(f"uploaded {local_path} ({size} bytes) -> {remote_path} [chunked, {chunk_size} bytes/chunk]")


def download_file(remote_path: str, local_path: str) -> None:
    token = get_access_token()
    site_id = resolve_site_id(token)
    drive_id = resolve_drive_id(token, site_id)

    url = f"{_item_url(site_id, drive_id, remote_path)}:/content"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
    tmp_path = local_path + f".tmp-{os.getpid()}"
    try:
        # urlopen is evaluated before `open(tmp_path, ...)` in this `with`,
        # so an HTTPError here means tmp_path was never created -- only
        # remove it below if the failure happened after that point.
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp_path, "wb") as out:
            # urlopen already follows Graph's redirect to the pre-authenticated
            # download URL; stream in chunks so a multi-GB file (e.g. the
            # pinned AACT snapshot) never sits fully in memory.
            shutil.copyfileobj(resp, out, length=1024 * 1024)
    except urllib.error.HTTPError as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        detail = e.read().decode("utf-8", "replace")
        raise SharePointTransferError(f"download failed: HTTP {e.code}: {detail}") from e
    os.replace(tmp_path, local_path)  # atomic: a kill mid-download leaves only the stray .tmp
    print(f"downloaded {remote_path} -> {local_path} ({os.path.getsize(local_path)} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="local file -> SharePoint")
    up.add_argument("local_path")
    up.add_argument("remote_path")
    up.add_argument("--chunk-size-mb", type=float, default=DEFAULT_CHUNK_SIZE / (1024 * 1024))

    down = sub.add_parser("download", help="SharePoint -> local file")
    down.add_argument("remote_path")
    down.add_argument("local_path")

    args = ap.parse_args()
    try:
        if args.command == "upload":
            chunk_size = int(args.chunk_size_mb * 1024 * 1024)
            chunk_size -= chunk_size % (320 * 1024)  # Graph requires a multiple of 320 KiB
            upload_file(args.local_path, args.remote_path, chunk_size=max(chunk_size, 320 * 1024))
        else:
            download_file(args.remote_path, args.local_path)
    except SharePointTransferError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
