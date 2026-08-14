# Does `smiless` carry chemical signal? — an RDKit probe

_Companion to `signal-analysis.md` / `text_vs_tabular_signal.md`. Asks whether the
`smiless` (molecule) column carries predictive signal that is **chemical** —
readable from molecular structure — as opposed to an artifact of which trials
happen to have a drug attached, or of recognizing drugs the model has already
seen. Run entirely **locally on CPU** against the real TrialBench data; no EC2
round trip, no torch._
_Date: 2026-07-24 · seed 42 · all 4 binary tasks × Phases 1–4 (16 cells)_

---

## 1. Why this needs three controls, not one

`smiless` looks like free predictive signal, but two confounds make a naive
"fit a model on fingerprints, report PR-AUC" number close to meaningless:

1. **Molecule presence is itself informative.** Only **42.7%–63.0%** of test rows
   have a parseable SMILES, and that presence tracks trial type (small-molecule
   drug vs. device / behavioral / biologic). A fingerprint model gets the
   `has_molecule` bit for free, so part of any apparent "chemistry" is really
   "this is a drug trial."
2. **The same drug recurs across many trials.** A cell has only ~450–1,500
   unique SMILES across thousands of trials. A model can score well by
   recognizing *molecules* without reading *structure* — which would not
   generalize to a new compound, the case anyone would actually want to predict.

So each molecule view is scored against controls that isolate exactly these:

| block | what it sees | role |
|---|---|---|
| `presence (control)` | `[has_molecule, n_molecules]` only | confound 1 |
| `descriptors` | presence + 23 physicochemical descriptors | chemistry (interpretable) |
| `morgan_fp` | presence + Morgan/ECFP r=2, 1024 bits | chemistry (substructural) |
| `drug_id (control)` | presence + multi-hot exact molecule identity | confound 2 |
| `scaffold` | presence + multi-hot Bemis–Murcko scaffold | chemical class |
| `tabular (reference)` | the repo's `TabularFeaturizer` output | leaderboard reference |
| `tabular+descriptors`, `tabular+morgan_fp` | concatenation | does it *add*? |

Plus a **novel-scaffold diagnostic**: PR-AUC restricted to test trials whose
Murcko scaffolds are *entirely absent* from train. Drug memorization cannot help
there — `drug_id` and `scaffold` have all-zero features on those rows and
collapse to the presence baseline by construction. This is the sharpest test
available for chemically *generalizing* signal.

## 2. Method

- `src/data/mol_features.py` — RDKit featurization per **unique SMILES**, cached
  across cells (`results/cache/mol_features/`). Per-trial aggregation: descriptors
  by mean over the trial's drugs, fingerprints by bitwise OR ("some drug in this
  trial contains this substructure"). Featurizing a whole cell takes **~0.5–4 s**.
- Leakage discipline per `CLAUDE.md`: RDKit featurization is label-free, and every
  *encoder* built from it (drug/scaffold vocabularies, descriptor median-imputer,
  standardizer) is fit on **train only**. Test is scored once per block.
- Models: dense blocks use **exactly** the benchmark's registered `lightgbm`
  config from `src/methods/gbm.py`; sparse/binary blocks (fingerprints, identity
  and scaffold vocabularies) use L2 logistic regression with `C` chosen on
  validation PR-AUC — the standard ECFP baseline. Every contrast that carries a
  conclusion below is within one model family.
- **Baseline integrity check.** The `tabular (reference)` block reproduces the
  benchmark's own `lightgbm` PR-AUC to **max |diff| 0.006, mean 0.002** across all
  16 cells. This check earned its place: an earlier version of the probe
  early-stopped on validation PR-AUC (which the benchmark's config does *not* do),
  which collapsed the tabular baseline in 4 of 16 cells and made fusion look
  dramatically better than it is.

Reproduce:

```bash
python -m experiments.smiles_signal_probe --all-binary --phases Phase1 Phase2 Phase3 Phase4 --out probe.json
```

```bash
python -m experiments.smiles_probe_summary probe.json
```

## 3. Findings

### 3.1 There is real signal beyond the presence confound — and it is endpoint-specific

`morgan_fp` beats the presence control in **16 of 16 cells**, mean **+0.086**
PR-AUC (range +0.015 to +0.225). But the magnitude splits cleanly by what the
endpoint *is*:

| task | mean Δ PR-AUC vs presence control |
|---|---|
| `mortality_rate_yn` | **+0.147** |
| `serious_adverse_rate_yn` | **+0.099** |
| `outcome` | +0.071 |
| `patient_dropout_rate_yn` | +0.027 |

The two pharmacological safety endpoints carry the most molecular signal; patient
dropout carries almost none. That ordering is what pharmacology predicts —
mortality and serious adverse events are toxicity outcomes, dropout is
operational — and dropout doubles as a **negative control** showing the probe
isn't manufacturing signal everywhere it looks.

### 3.2 On the full test set it is almost entirely drug memorization

`morgan_fp` minus `drug_id` on the full test set: mean **+0.004**, median −0.000,
range −0.012 to +0.041. Structure buys essentially **nothing** over knowing which
molecule it is. Anyone reporting a fingerprint PR-AUC here without this control
would be reporting a drug lookup table.

### 3.3 On unseen chemistry, structure does generalize — on the safety endpoints

Restricted to novel-scaffold test rows (780 rows total; median 40 per cell), where
identity features are useless:

| task | `descriptors` − `drug_id` | `morgan_fp` − `drug_id` |
|---|---|---|
| `mortality_rate_yn` | **+0.167** | **+0.220** |
| `serious_adverse_rate_yn` | **+0.151** | **+0.123** |
| `outcome` | +0.101 | +0.120 |
| `patient_dropout_rate_yn` | +0.003 | +0.020 |

Positive in **8 of 8** mortality and serious-adverse cells. This is the load-bearing
result: the molecule column contains information that transfers to chemistry the
model has never seen, concentrated exactly where toxicity should live. Caveat: per-cell
n is 25–87, so treat individual cells as noisy and the per-task pattern as the finding.

### 3.4 But it does **not** add over the tabular view

Best fusion minus the tabular reference: mean **+0.017**, median +0.014, range
−0.017 to +0.082. Four of 16 cells exceed +0.03; three nominally clear the tabular
block's bootstrap CI (`outcome/Phase4` +0.082, `patient_dropout/Phase4` +0.051,
`patient_dropout/Phase3` +0.031). At 16 tests that is roughly chance, and the
"wins" land in the cells where standalone chemical signal was **weakest** while
mortality and serious-adverse show +0.007 and +0.012 — the opposite of a coherent
effect. **No convincing evidence that adding molecules moves the leaderboard.**

The natural reading: the chemical signal is real but largely *redundant* with what
the tabular view already encodes — sponsor class, intervention counts, condition
and phase act as proxies for drug class. Consistent with `signal-analysis.md`,
where modalities kept turning out to share signal rather than add to it.

### 3.5 Which chemistry matters

Descriptor importance, Borda-summed over the 16 cells:

```
141  MolLogP          101  FractionCSP3      65  BertzCT
137  BalabanJ          93  TPSA              58  MolMR
129  qed               89  HallKierAlpha     51  MolWt
```

Lipophilicity (`MolLogP`), polar surface area (`TPSA`), drug-likeness (`qed`) and
saturation (`FractionCSP3`) leading is chemically legible — these are the
permeability/ADMET axes you would reach for by hand. Caveat: LightGBM's default
importance counts splits, which favors high-cardinality continuous features, so
the topological indices (`BalabanJ`, `HallKierAlpha`) are plausibly inflated.
Read this as suggestive, not causal.

## 4. What this means for the benchmark

- **Don't expect a leaderboard gain from `fingerprint_fusion`.** §3.4 says naive
  concatenation is a wash. Building the Tier D method is still worth doing for
  completeness against the HINT baseline, but it should be framed as a modality
  ablation, not a contender.
- **The result worth reporting is the ablation itself**: molecular structure carries
  genuine, chemically generalizing signal on the two toxicity endpoints, ~0 on
  dropout, and it is redundant with the tabular view. That is a finding about
  TrialBench, not about a model.
- **`presence` and `drug_id` belong in any future molecule work.** Without them the
  headline fingerprint number (§3.2) is indistinguishable from a lookup table.
- Consider persisting per-run test predictions, as `signal-analysis.md` §2 already
  recommended — this probe again had to refit everything from scratch.

## 5. Limitations

- Single seed (42), single train/valid split. §3.3 and §3.4 deltas are unreplicated.
- Novel-scaffold subsets are small (25–87 rows/cell) and have base rates that differ
  from the full test set, so those PR-AUCs compare **across blocks**, never against
  the full-test column.
- Descriptors are mean-aggregated across a trial's drugs, which blurs combination
  therapy; fingerprints use OR, which does not.
- `failure_reason` (multiclass) is not covered — the probe is binary-only.
- Fusion is plain concatenation. A properly tuned late-fusion or per-modality
  ensemble could plausibly do better than §3.4 suggests.
