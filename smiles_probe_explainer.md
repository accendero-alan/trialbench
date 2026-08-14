# A quick explanation of the SMILES exploration

_Plain-language companion to `smiles_chemical_signal.md`, which has the full
tables and caveats. Two-minute version._

## The question

TrialBench gives every trial a `smiless` column — the SMILES strings of the drugs
being tested. Our benchmark has never used it: the tabular view deliberately
excludes it, and the one method that would (`fingerprint_fusion`) is still a stub.

So: **does that column actually know anything?** And can we find out cheaply —
locally, on CPU, without provisioning anything?

Yes on both counts. RDKit is a single 25 MB wheel with no torch dependency, the
data is already on the laptop, and each task/phase cell has only ~450–1,500
*unique* molecules even across thousands of trials — so featurizing a whole cell
takes a few seconds. The full sweep of 16 cells ran in about 8 minutes.

## Why you can't just fit a model on fingerprints

This is the part that makes the exploration interesting. The obvious experiment —
compute Morgan fingerprints, train a classifier, report PR-AUC — produces a
number that looks like chemistry but almost certainly isn't. Two traps:

**Trap 1: having a molecule at all is informative.** Only 43–63% of trials have a
parseable SMILES, and whether one exists tracks what kind of trial it is
(small-molecule drug vs. device, behavioral, biologic). Any molecule model gets
that hint for free. So part of the "chemical signal" is really just *"this is a
drug trial."*

**Trap 2: the same drugs keep showing up.** With ~1,000 unique molecules spread
across thousands of trials, a model can score well by *recognizing specific
drugs* rather than reading their structure. That's a lookup table. It tells you
nothing about a compound you haven't seen — which is the only case anyone would
actually want to predict.

## How the probe handles them

Each trap gets an explicit control to measure against:

| control | what it sees | what it isolates |
|---|---|---|
| `presence` | only `[has_molecule, n_molecules]` | trap 1 |
| `drug_id` | multi-hot of exact molecule identity | trap 2 |

A chemical view only counts as informative if it beats **both**.

Then the sharpest test: score PR-AUC on **only** the test trials whose molecular
scaffolds (ring skeletons) never appear in training. Memorization is impossible
there by construction — `drug_id` has nothing but zeros on those rows. Whatever
survives is structure the model genuinely read.

## What we found

**On the full test set, it's a lookup table.** Fingerprints beat exact-molecule
identity by +0.004 PR-AUC — nothing. Reporting the raw fingerprint number
without this control would have been reporting drug recognition and calling it
chemistry.

**On unseen chemistry, structure genuinely generalizes — but only for the right
endpoints.** Beating the identity control on novel-scaffold rows:

- mortality **+0.167**
- serious adverse events **+0.151**
- trial approval +0.101
- patient dropout **+0.003**

That ordering is the most satisfying part of the result. Mortality and serious
adverse events are *toxicity* outcomes, where molecular structure should matter —
and it does, consistently (8 of 8 cells). Patient dropout is an *operational*
outcome, driven by protocol burden and site logistics, where chemistry has no
business helping — and it doesn't. Dropout doubles as a negative control proving
the probe isn't just manufacturing signal wherever it looks.

The descriptors doing the work are chemically legible too: `MolLogP`
(lipophilicity), `TPSA` (polar surface area), `qed` (drug-likeness),
`FractionCSP3` — the ADMET axes a chemist would reach for unprompted.

**But it won't move the leaderboard.** Concatenating molecule features onto the
tabular view gains a mean of +0.017 PR-AUC, and the few nominal "wins" land in
the cells where standalone chemical signal was *weakest* — i.e. noise. The
chemistry appears largely redundant with what sponsor class, intervention counts,
and condition already encode. That echoes what `signal-analysis.md` kept finding:
in this dataset, modalities tend to *share* signal rather than add to it.

## Why it was worth doing anyway

The value isn't a new leaderboard row — it's a defensible statement about the
dataset: *molecular structure carries real, chemically generalizing signal on
TrialBench's two toxicity endpoints, none on its operational endpoint, and it is
redundant with the tabular view.* That's a modality ablation, and it's the kind
of claim that survives someone pushing back on it, because the controls are there.

One process note earned by mistake: the first sweep used early stopping that the
benchmark's own `lightgbm` config doesn't, which quietly collapsed the tabular
baseline in 4 of 16 cells and showed fusion gains of +0.26 and +0.33 that weren't
real. The probe now reproduces the leaderboard's LightGBM to within 0.006, and
that check runs as part of the summary. **A baseline you haven't verified against
a known number can invent an effect larger than the one you're measuring.**

## Files

| file | what it is |
|---|---|
| [src/data/mol_features.py](src/data/mol_features.py) | RDKit featurization of `smiless` (descriptors, fingerprints, scaffolds), cached |
| [experiments/smiles_signal_probe.py](experiments/smiles_signal_probe.py) | the probe and its controls |
| [experiments/smiles_probe_summary.py](experiments/smiles_probe_summary.py) | cross-cell tables + the baseline-integrity check |
| [smiles_chemical_signal.md](smiles_chemical_signal.md) | full findings, numbers, and limitations |

```bash
python -m experiments.smiles_signal_probe --task serious_adverse_rate_yn --phase Phase3
```
