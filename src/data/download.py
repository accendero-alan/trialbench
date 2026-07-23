"""Fetch TrialBench data into ``data/`` (the layout src/data/loader.py expects).

Default route downloads just the 5 classification-task zips directly from
Zenodo (stdlib only, no extra dependencies) and unzips them. This is
preferred over the official ``trialbench`` pip package for this benchmark:
that package hard-imports ``torch`` even just to call its download function,
and its ``download_all_data`` fetches all 8 TrialBench tasks -- including the
~250MB of out-of-scope regression/generation data (duration, dose,
eligibility-criteria-design) this project doesn't use. Pass ``--via-package``
to use it anyway if you already have torch installed. As a last resort you
can point at a local clone of ML2ClinicalTrials to use its small toy samples
for development.

    python -m src.data.download                            # direct Zenodo (default)
    python -m src.data.download --via-package               # official trialbench pkg (needs torch)
    python -m src.data.download --from-clone /path/to/ML2ClinicalTrials

The five classification tasks map to these Zenodo files (record 15455785):
    trial-approval-forecasting.zip
    mortality-event-prediction.zip
    serious-adverse-event-forecasting.zip
    patient-dropout-event-forecasting.zip
    trial-failure-reason-identification.zip
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from .loader import TASKS

ZENODO_BASE = "https://zenodo.org/records/15455785/files"
CLASSIFICATION_FOLDERS = sorted({folder for folder, _, _ in TASKS.values()})


def via_zenodo(data_root: str):
    """Download just the 5 classification-task zips directly from Zenodo and
    unzip each into data_root/<folder>/. Skips a folder that's already
    populated, so re-running after a partial failure only fetches what's
    still missing."""
    os.makedirs(data_root, exist_ok=True)
    for folder in CLASSIFICATION_FOLDERS:
        dest_dir = os.path.join(data_root, folder)
        if os.path.isdir(dest_dir) and os.listdir(dest_dir):
            print(f"  {folder}: already present, skipping")
            continue
        url = f"{ZENODO_BASE}/{folder}.zip?download=1"
        print(f"  {folder}: downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = resp.read()

        tmp = tempfile.mkdtemp(prefix=f"{folder}-")
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                zf.extractall(tmp)
            # These zips ship a single top-level folder (matching the zip's
            # own name) wrapping the Phase1..4 dirs. Extract from wherever
            # that content actually landed rather than assuming the wrapper
            # name matches `folder` exactly -- extracting straight into
            # dest_dir (already named `folder`) would double it up into
            # dest_dir/folder/Phase1/... instead of dest_dir/Phase1/....
            entries = os.listdir(tmp)
            if len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])):
                src = os.path.join(tmp, entries[0])
            else:
                src = tmp
            shutil.move(src, dest_dir)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"  {folder}: extracted to {dest_dir}")


def via_package(data_root: str):
    """Use the official trialbench package to download all 8 datasets.
    Requires torch (a hard import in trialbench.function, unrelated to any
    GPU use) and pulls out-of-scope tasks too -- prefer via_zenodo for this
    CPU-only, classification-only benchmark."""
    import trialbench  # noqa
    os.makedirs(data_root, exist_ok=True)
    trialbench.function.download_all_data(data_root if data_root.endswith("/") else data_root + "/")
    print(f"Downloaded TrialBench data to {data_root}/")


def from_clone(data_root: str, clone_path: str):
    """Copy the toy samples bundled in a local ML2ClinicalTrials clone."""
    src = os.path.join(clone_path, "Trialbench", "data")
    if not os.path.isdir(src):
        raise FileNotFoundError(f"expected toy data at {src}")
    os.makedirs(data_root, exist_ok=True)
    copied = []
    for folder in CLASSIFICATION_FOLDERS:
        s = os.path.join(src, folder)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(data_root, folder), dirs_exist_ok=True)
            copied.append(folder)
    # ontology assets used by multimodal methods
    mesh = os.path.join(src, "mesh-embeddings")
    if os.path.isdir(mesh):
        shutil.copytree(mesh, os.path.join(data_root, "mesh-embeddings"), dirs_exist_ok=True)
    print(f"Copied toy samples for: {copied}")


def print_zenodo_hint(data_root: str):
    print("Manual download (Zenodo record 15455785):")
    for folder in CLASSIFICATION_FOLDERS:
        print(f"  {ZENODO_BASE}/{folder}.zip?download=1  ->  unzip into {data_root}/{folder}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--from-clone", help="path to a local ML2ClinicalTrials clone")
    ap.add_argument("--via-package", action="store_true",
                     help="use the official trialbench pip package instead of a direct "
                          "Zenodo download (needs torch; downloads all 8 tasks, not just "
                          "the 5 classification ones)")
    args = ap.parse_args()

    if args.from_clone:
        from_clone(args.data_root, args.from_clone)
        return

    route = via_package if args.via_package else via_zenodo
    try:
        route(args.data_root)
    except Exception as e:  # noqa: BLE001
        print(f"data download failed ({e}).")
        print_zenodo_hint(args.data_root)
        sys.exit(1)  # propagate failure -- callers (e.g. bootstrap_ec2.sh) rely on this


if __name__ == "__main__":
    main()
