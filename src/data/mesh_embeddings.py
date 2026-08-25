"""Shipped MeSH graph embeddings (node2vec over MeSH tree numbers), for P11
(disease-representation-test-plan.md, T25 arm (b)).

``mesh_embeddings.txt.gz`` + ``mesh_ui_to_id.pickle`` + ``term_to_ui.pkl``
(``data/external/mesh-embeddings/``) are TrialBench's own shipped files --
downloaded from its own repo (ML2Health/ML2ClinicalTrials, MIT-licensed,
tracing back to github.com/helboukkouri/mesh-embeddings) rather than the
``trialbench`` PyPI package, which pulls in torch and ~250MB of
out-of-scope data (PLAN.md's data-acquisition section) just to place this
one file. Loading logic mirrors TrialBench's own
``Trialbench/models/mesh_encode.py`` exactly: 29,638 terms, 256-d, term ->
UI -> row index.
"""
from __future__ import annotations

import gzip
import os
import pickle
from functools import lru_cache

import numpy as np

MESH_DIR = os.path.join("data", "external", "mesh-embeddings")
EMBEDDINGS_PATH = os.path.join(MESH_DIR, "mesh_embeddings.txt.gz")
UI_TO_ID_PATH = os.path.join(MESH_DIR, "mesh_ui_to_id.pickle")
TERM_TO_UI_PATH = os.path.join(MESH_DIR, "term_to_ui.pkl")


@lru_cache(maxsize=1)
def _load(mesh_dir: str = MESH_DIR):
    ui_to_id_path = os.path.join(mesh_dir, "mesh_ui_to_id.pickle")
    term_to_ui_path = os.path.join(mesh_dir, "term_to_ui.pkl")
    embeddings_path = os.path.join(mesh_dir, "mesh_embeddings.txt.gz")
    if not (os.path.exists(ui_to_id_path) and os.path.exists(term_to_ui_path)
            and os.path.exists(embeddings_path)):
        raise FileNotFoundError(
            f"MeSH embedding files not found under {mesh_dir}. P11 needs all three "
            f"of mesh_embeddings.txt.gz, mesh_ui_to_id.pickle, term_to_ui.pkl from "
            f"github.com/ML2Health/ML2ClinicalTrials "
            f"(Trialbench/data/mesh-embeddings/) copied there."
        )
    with open(ui_to_id_path, "rb") as f:
        ui_to_id = pickle.load(f)
    with open(term_to_ui_path, "rb") as f:
        term_to_ui = pickle.load(f)

    vectors = {}
    with gzip.open(embeddings_path, "rt") as f:
        _n, dim = f.readline().strip().split()
        dim = int(dim)
        for line in f:
            parts = line.strip().split()
            idx = int(parts[0])
            vectors[idx] = np.array(parts[1:], dtype=np.float32)
            assert len(vectors[idx]) == dim
    return term_to_ui, ui_to_id, vectors, dim


def mesh_term_vector(term: str, mesh_dir: str = MESH_DIR):
    """The 256-d embedding for one MeSH term string, or ``None`` if the term
    isn't in TrialBench's shipped vocabulary (30,764 terms map to 29,638
    embeddings; not every term resolves)."""
    term_to_ui, ui_to_id, vectors, _dim = _load(mesh_dir)
    ui = term_to_ui.get(term)
    if ui is None:
        return None
    idx = ui_to_id.get(ui)
    if idx is None:
        return None
    return vectors.get(idx)


def mesh_trial_vectors(term_lists, mesh_dir: str = MESH_DIR) -> np.ndarray:
    """``term_lists``: one list of MeSH term strings per trial (e.g. parsed
    from ``condition_browse/mesh_term`` via
    ``src.data.features._recursive_parse_terms``) -> ``(n, 256)``, each row
    the mean of that trial's resolved term vectors (all-zero if none
    resolve -- pair with a presence control, per the T21 pattern)."""
    _term_to_ui, _ui_to_id, _vectors, dim = _load(mesh_dir)
    out = np.zeros((len(term_lists), dim), dtype=np.float32)
    for i, terms in enumerate(term_lists):
        rows = [v for t in terms if (v := mesh_term_vector(t, mesh_dir)) is not None]
        if rows:
            out[i] = np.mean(rows, axis=0)
    return out
