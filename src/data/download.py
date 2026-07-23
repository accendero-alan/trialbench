"""Fetch TrialBench data into ``data/`` (the layout src/data/loader.py expects).

Preferred route is the official ``trialbench`` package. Zenodo zips are the
fallback. As a last resort you can point at a local clone of ML2ClinicalTrials
to use its small toy samples for development.

    python -m src.data.download                    # via trialbench package
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
import os
import shutil

from .loader import TASKS

ZENODO_BASE = "https://zenodo.org/records/15455785/files"
CLASSIFICATION_FOLDERS = sorted({folder for folder, _, _ in TASKS.values()})


def via_package(data_root: str):
    """Use the trialbench package to download the full datasets."""
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
    args = ap.parse_args()

    if args.from_clone:
        from_clone(args.data_root, args.from_clone)
        return
    try:
        via_package(args.data_root)
    except Exception as e:  # noqa: BLE001
        print(f"trialbench package route failed ({e}).")
        print_zenodo_hint(args.data_root)


if __name__ == "__main__":
    main()
