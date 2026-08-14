"""Chemically aware featurization of the ``smiless`` column (RDKit).

Three molecule views, all computed **per unique SMILES** and cached, then
aggregated per trial:

- **descriptors** — ~22 interpretable physicochemical / drug-likeness values
  (MolWt, cLogP, TPSA, HBD/HBA, rotatable bonds, ring counts, FractionCSP3,
  QED, Lipinski violations, ...). Chemistry the model can reason about, and the
  block most likely to carry ADMET-flavoured signal for safety endpoints.
- **Morgan/ECFP fingerprint** — radius 2, 1024 bits: which *substructures* are
  present. Aggregated across a trial's drugs by bitwise OR (i.e. "some drug in
  this trial contains this substructure").
- **Bemis–Murcko scaffold** — the molecule's ring-system skeleton, used both as
  a coarse chemical-class feature and to build a scaffold-novelty diagnostic
  (does a molecule model still work on test trials whose chemistry is unseen in
  train?).

Nothing here fits on labels, so it is safe to compute over all splits at once;
the *encoders* built from these views (vocabularies, imputers, scalers) are fit
on train only by the caller.

RDKit is imported lazily so the module can be imported without it installed.
"""
from __future__ import annotations

import ast
import hashlib
import os
import pickle
from typing import Iterable

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join("results", "cache", "mol_features")
CACHE_VERSION = "v1"
FP_BITS = 1024
FP_RADIUS = 2

# (name, callable-name) — resolved against the rdkit modules in _descriptor_fns.
DESCRIPTOR_NAMES = [
    "MolWt", "HeavyAtomCount", "MolLogP", "MolMR", "TPSA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
    "FractionCSP3", "NumHeteroatoms", "NOCount", "NHOHCount",
    "BertzCT", "BalabanJ", "HallKierAlpha", "NumRadicalElectrons",
    "qed", "formal_charge", "lipinski_violations",
]


# --------------------------------------------------------------------------
# SMILES parsing off the raw column
# --------------------------------------------------------------------------
def parse_smiles_cell(value) -> list:
    """``smiless`` holds a stringified python list of SMILES. Be forgiving."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s or s in {"[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        parsed = [s]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        return []
    return [m.strip() for m in parsed if isinstance(m, str) and m.strip()]


def smiles_lists(X: pd.DataFrame, col: str = "smiless") -> list:
    if col not in X.columns:
        return [[] for _ in range(len(X))]
    return [parse_smiles_cell(v) for v in X[col].values]


# --------------------------------------------------------------------------
# Per-molecule featurization (cached across task/phase cells)
# --------------------------------------------------------------------------
def _descriptor_fns():
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    def lipinski_violations(mol):
        v = 0
        v += Descriptors.MolWt(mol) > 500
        v += Crippen.MolLogP(mol) > 5
        v += Lipinski.NumHDonors(mol) > 5
        v += Lipinski.NumHAcceptors(mol) > 10
        return float(v)

    fns = {
        "MolWt": Descriptors.MolWt,
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "MolLogP": Crippen.MolLogP,
        "MolMR": Crippen.MolMR,
        "TPSA": rdMolDescriptors.CalcTPSA,
        "NumHDonors": Lipinski.NumHDonors,
        "NumHAcceptors": Lipinski.NumHAcceptors,
        "NumRotatableBonds": Lipinski.NumRotatableBonds,
        "RingCount": Descriptors.RingCount,
        "NumAromaticRings": Lipinski.NumAromaticRings,
        "NumAliphaticRings": Lipinski.NumAliphaticRings,
        "NumSaturatedRings": Lipinski.NumSaturatedRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "NumHeteroatoms": Lipinski.NumHeteroatoms,
        "NOCount": Lipinski.NOCount,
        "NHOHCount": Lipinski.NHOHCount,
        "BertzCT": Descriptors.BertzCT,
        "BalabanJ": Descriptors.BalabanJ,
        "HallKierAlpha": Descriptors.HallKierAlpha,
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
        "qed": QED.qed,
        "formal_charge": lambda m: float(Chem.GetFormalCharge(m)),
        "lipinski_violations": lipinski_violations,
    }
    return [(n, fns[n]) for n in DESCRIPTOR_NAMES]


def _cache_path() -> str:
    key = hashlib.sha1(
        f"{CACHE_VERSION}|{FP_RADIUS}|{FP_BITS}|{','.join(DESCRIPTOR_NAMES)}".encode()
    ).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"mols_{key}.pkl")


def _load_cache() -> dict:
    p = _cache_path()
    if os.path.exists(p):
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    p = _cache_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def featurize_molecules(smiles: Iterable[str], use_cache: bool = True) -> dict:
    """{smiles -> {'desc': (D,) float array, 'fp': (FP_BITS,) uint8, 'scaffold': str}}.

    Unparseable SMILES are omitted from the result (the caller treats them as
    "no molecule"). Results are cached across cells keyed by featurizer config.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    wanted = sorted({s for s in smiles if s})
    cache = _load_cache() if use_cache else {}
    todo = [s for s in wanted if s not in cache]

    if todo:
        fns = _descriptor_fns()
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
        for smi in todo:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                cache[smi] = None
                continue
            desc = np.empty(len(fns), dtype=float)
            for i, (_, fn) in enumerate(fns):
                try:
                    v = float(fn(mol))
                except Exception:
                    v = np.nan
                desc[i] = v if np.isfinite(v) else np.nan
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaf = ""
            cache[smi] = {
                "desc": desc,
                "fp": gen.GetFingerprintAsNumPy(mol).astype(np.uint8),
                "scaffold": scaf,
            }
        if use_cache:
            _save_cache(cache)

    return {s: cache[s] for s in wanted if cache.get(s) is not None}


# --------------------------------------------------------------------------
# Per-trial aggregation
# --------------------------------------------------------------------------
def aggregate(mol_lists: list, feats: dict):
    """Aggregate per-molecule features up to one row per trial.

    Returns ``(presence, desc, fp, scaffold_sets)``:
      presence      (n, 2)          [has_molecule, n_molecules]
      desc          (n, D)          mean over the trial's molecules; NaN if none
      fp            (n, FP_BITS)    bitwise OR over the trial's molecules
      scaffold_sets list[set[str]]  Murcko scaffolds present in the trial
    """
    n = len(mol_lists)
    D = len(DESCRIPTOR_NAMES)
    presence = np.zeros((n, 2), dtype=float)
    desc = np.full((n, D), np.nan, dtype=float)
    fp = np.zeros((n, FP_BITS), dtype=np.uint8)
    scaffolds = []

    for i, mols in enumerate(mol_lists):
        known = [feats[m] for m in mols if m in feats]
        presence[i] = [1.0 if known else 0.0, float(len(known))]
        scaffolds.append({f["scaffold"] for f in known if f["scaffold"]})
        if not known:
            continue
        desc[i] = np.nanmean(np.vstack([f["desc"] for f in known]), axis=0)
        fp[i] = np.bitwise_or.reduce(np.vstack([f["fp"] for f in known]), axis=0)

    return presence, desc, fp, scaffolds


def vocab_matrix(item_sets: list, vocab: list) -> np.ndarray:
    """Multi-hot encode list-of-sets against a fixed (train-derived) vocabulary."""
    idx = {v: i for i, v in enumerate(vocab)}
    M = np.zeros((len(item_sets), len(vocab)), dtype=float)
    for r, items in enumerate(item_sets):
        for it in items:
            j = idx.get(it)
            if j is not None:
                M[r, j] = 1.0
    return M
