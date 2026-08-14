# LLM_CONTAMINATION_PLAN.md — quarantined LLM study: how much is recall, not prediction?

The LLM methods are **out of the TrialBench benchmark** (`TEXT_PLAN.md` §4 T4).
They are not deferred, not "Tier E," and not commented out of a config pending
budget. They are disqualified as leaderboard methods and relocated here, where
the question changes from "can an LLM predict trial outcomes?" to **"how much of
an LLM's apparent skill on TrialBench is retrieval of trials it has already
read?"**

That question has a measurable answer, produces a finding whichever way it comes
out, and is worth answering once for everybody who benchmarks LLMs on
clinicaltrials.gov data.

_Status: proposed, not yet executed. Date: 2026-08-11._

---

## 1. Why quarantine rather than descope

Three independent disqualifiers, each sufficient on its own:

1. **Not version-pinnable.** Every method in the benchmark is reproducible from
   `requirements.txt` plus a seed. A model behind an API endpoint is not. A
   number produced in August 2026 may be unreproducible next year, which defeats
   the purpose of a stable comparative grid.
2. **Contamination cannot be ruled out.** TrialBench is built from public
   clinicaltrials.gov records. Many outcomes are published in registry results,
   press releases, and literature. Across the `outcome` task's cells, NCT
   ordinals run from 172 (Phase3) to 6,161,792 (Phase2) — no single cell spans
   that whole range — implying registrations from roughly 1999 to 2024. That
   plausibly places them inside current pretraining corpora, though this is an
   inference from vintage rather than a verified fact about any training set,
   which is exactly why E1 is needed.
3. **The score format is penalized by the metric.** No token logprobs are
   exposed, so a probability must be verbalized, and verbalized probabilities
   cluster on round numbers. `average_precision_score` collapses ties to a single
   threshold and assigns the block its average precision, so ordering inside a
   tie block is lost and PR-AUC understates real discriminative ability.

Plus a framing mismatch that no engineering fixes: the benchmark asks which
method learns best from a given training split. A model that has read the
internet is not answering that question.

**What quarantine buys.** Disqualifier 3 becomes irrelevant once the goal stops
being a leaderboard row (see §4's metric choices). Disqualifier 2 becomes the
subject rather than a caveat. Disqualifier 1 is handled by treating any model
output as a **frozen, committed artifact** rather than as a live method (§6).

---

## 2. Isolation rules

Non-negotiable, because a quarantine that leaks is not a quarantine.

- Own config: `configs/llm_contamination.yaml`. Own results tree:
  `results_llm/`. Nothing writes to `results/` or `results_text/`.
- `configs/text_benchmark.yaml` carries no `llm_*` entry, and
  `tests/test_text_smoke.py` asserts that the *enabled* method list contains
  none. Note the limits of that guard honestly: a test cannot see a commented
  YAML line, so it protects the new config's enabled list and nothing more.
- `configs/benchmark.yaml` currently has `# - llm_fewshot` commented out under
  its Tier D block. Leave it. `TEXT_PLAN.md` §7 item 8 says not to disturb that
  file, and a commented line runs nothing. Add a trailing comment pointing at
  this document instead of deleting the line, so the next reader learns why it
  is not to be uncommented.
- `requirements-extended.txt:27` likewise already carries a commented
  `# anthropic` under Tier D. Leave the line, annotate it with a pointer here.
  (`deploy/bootstrap_ec2.sh` strips comments before installing, so it has never
  been installed.)
- `src/methods/llm.py` stays a stub in the benchmark's registry. This study's
  code lives in `experiments/llm_contamination/`. That follows the *spirit* of
  the existing `experiments/` scripts, which read through the real loader and
  write nothing under `results/`, with two departures worth naming: those scripts
  are a flat set of `python -m experiments.x` modules with no sub-packages, so a
  subdirectory is a new layout, and `experiments/` is currently untracked in git.
- Own dependency file, `requirements-llm.txt`, carrying `anthropic`. Not
  referenced by `requirements.txt`, `requirements-text.txt`, or
  `requirements-extended.txt`.
- Every number this study produces is labeled **not comparable to the
  leaderboard** in its own writeup. No merge into any `leaderboard.md`.
- **Gates nothing** in `TEXT_PLAN.md`. The reverse is not true: E2 and E3 depend
  on that plan's harness items and step-3 output (see §4). E1, the decisive
  experiment, depends on nothing and can run immediately — which is another
  reason it is the one to run first.

---

## 3. Target cells, and why

**Primary: `outcome`/Phase3** (trial approval, n_test 1,913, train prevalence
0.563, test prevalence 0.555). Approval outcomes are the most publicly reported
label in the benchmark, so it is the most contaminated cell if any cell is, and
near-balanced test prevalence makes accuracy interpretable alongside AUROC.

**Replication: `outcome`/Phase2** (n_test 2,514, train prevalence 0.385, test
prevalence 0.368).

### Registration vintage is confounded with the label. Everywhere.

An earlier draft of this plan claimed `outcome` prevalence was flat across
vintage and built E2 on that. **It is not flat, and the claim came from a broken
join.** `test_x.csv` and `test_y.csv` are stored in *different row orders* for
this task (`test_y` is sorted by NCT id, `test_x` is not), so a positional
pairing of features to labels silently mismatches rows. The benchmark's own
loader is unaffected — `src/data/loader.py:_read_xy` does
`x.index.intersection(y.index)` and `.loc[common]`, which is correct — but the
ad-hoc analysis behind that draft did not. Index-aligned, the real distribution
is:

| NCT ordinal bin (approx. registration) | `outcome`/P3 n | P3 prevalence | `outcome`/P2 n | P2 prevalence |
|---|---|---|---|---|
| < NCT01000000 (~pre-2010) | 779 | 0.530 | 945 | 0.350 |
| NCT01–02M (~2010–13) | 635 | 0.619 | 780 | 0.429 |
| NCT02–03M (~2014–16) | 362 | 0.624 | 475 | 0.442 |
| NCT03–04M (~2017–19) | 89 | 0.337 | 212 | 0.236 |
| NCT04–05M (~2020–21) | 41 | **0.000** | 83 | **0.000** |
| NCT05M+ (~2022+) | 7 | **0.000** | 19 | **0.000** |

The two newest bins contain **zero positives** in both cells, which is
mechanically sensible: recently registered trials have not yet been approved.
Average precision is undefined with no positives, so those bins cannot be scored
at all. And `mortality_rate_yn`/Phase1, which the earlier draft excluded for
being confounded (0.294 / 0.510 / 0.367 / 0.199 / 0.090 / 0.000), is confounded
no worse than the cells the draft preferred.

**Consequences, taken honestly.** There is no unconfounded cell in this data, so
the vintage-gradient design is weak by construction and cannot be the study's
primary evidence. Two adjustments follow, both reflected in §4:

- **E1 becomes the load-bearing experiment.** ID-only recall needs no vintage
  assumption at all, and its per-bin version compares the model against *that
  bin's own* majority rate rather than against another model, so the confound
  cancels.
- **E2 is demoted to supporting evidence** and restricted to bins 1–3 (1,776 of
  1,913 rows on Phase3, 2,200 of 2,514 on Phase2), scored as a *paired delta*
  against `tfidf_logreg` on identical rows, which is far less prevalence-
  sensitive than either method's absolute PR-AUC.

**Binning.** Bins 1–3 only, for the reasons above. The NCT-ordinal-to-year
mapping is approximate (IDs are assigned roughly sequentially at registration)
and is used only for ordering, never as a date.

---

## 4. The experiments

### E1 — ID-only recall. The decisive test.

**Prompt contains the NCT ID and nothing else.** No title, no condition, no
eligibility text. Ask for the label.

If the model beats the prevalence baseline on IDs alone, it is recalling
specific trials, and contamination is **demonstrated rather than suspected**.
There is no predictive-signal explanation for skill on an opaque accession
number.

Controls, all required:

- **Prevalence baseline.** Always predicting the majority class gives **0.555**
  accuracy on `outcome`/Phase3 (test prevalence 0.555; the 0.563 figure is *train*
  prevalence and is the wrong baseline). On Phase2 the majority-class accuracy is
  0.632, since test prevalence is 0.368. The comparison is against those, not
  against 0.5.
- **Per-bin baseline, which is where the vintage question actually lives.**
  Score ID-only recall within each of bins 1–3 against **that bin's own** majority
  rate (Phase3: 0.530 / 0.619 / 0.624). Because both sides of the comparison come
  from the same bin, the vintage-label confound in §3 cancels, which is why this
  replaces the model-versus-model gradient of E2 as the primary vintage evidence.
  A recall advantage that grows toward older trials is the memorization signature.
- **Shuffled-ID control.** Re-run with IDs randomly reassigned to labels. This
  measures the floor including any parse artifacts and any tendency to answer
  with the majority class, and it is the number E1's result must clear.
- **Fabricated-ID control.** A block of syntactically valid but non-existent NCT
  IDs. High confidence on these indicates the model is guessing from ID
  structure (e.g. inferring era from the numeric range) rather than recalling,
  which is a *different* and much weaker form of leakage that must be separated
  out.
- **Refusal and non-answer accounting.** A model may decline or say it does not
  know. Report the refusal rate explicitly; score only answered rows, and report
  answered-row count alongside every figure. A high refusal rate is itself a
  result.

Metric: accuracy and AUROC over answered rows, versus the two controls. ~600
calls per arm on the primary cell.

### E2 — Vintage gradient, full text. Supporting evidence only.

Run the full-text zero-shot method, then compare its per-bin score against
`tfidf_logreg`'s on identical rows. TF-IDF learned only from the training split
and has no memory of 2004, so it is the "no contamination possible" reference.

**Restricted and demoted, per §3.** Bins 1–3 only (bins 4–6 are unscorable or
near-unscorable: zero positives in 4 and 5, n=7 in 6). Report the **paired
delta** on identical rows rather than each method's absolute PR-AUC, since the
delta is much less sensitive to the bin's prevalence than either level is.
Report AUROC alongside PR-AUC, as it degrades more gracefully at skewed
prevalence. Even so, prevalence still moves 0.530 → 0.624 across the three
usable bins, so a modest gradient here is not by itself evidence of anything.
E1's per-bin arm is the cleaner test; this one exists to check that the two
agree.

**Dependencies on the benchmark** (contra the "gated on nothing" framing this
plan previously used, see §2): the TF-IDF side needs persisted test predictions
(`TEXT_PLAN.md` §6.1, harness item 3), the text input needs the `"text"` feature
view (item 1), and any claim that the TF-IDF reference is reproducible needs the
deterministic `concat_text` fix (item 0) plus the `TEXT_COLS` name fix. E2
therefore cannot run before `TEXT_PLAN.md` step 1 completes.

### E3 — Identifiability ablation

Zero-shot with progressively less identifying input: full text, then
text-minus-`brief_title`, then `eligibility/criteria/textblock` only. A trial
title is close to a unique key; eligibility boilerplate is not.

If LLM performance collapses when the title is removed while TF-IDF's barely
moves (the field ablation in `TEXT_PLAN.md` §5.1 supplies the TF-IDF side at no
extra cost, which makes E3 dependent on that plan's step 3), the model was
keying on identity rather than on clinical content.

### E4 — Contamination-adjusted ceiling

Only after E1–E3: report zero-shot and few-shot performance on the primary
cells, framed as **an upper bound on what public knowledge plus recall can do**,
explicitly not as a benchmark entry. If E1–E3 show contamination, this number is
reported *only* alongside them and never in isolation.

Score-granularity handling, since this is the one experiment where ranking
resolution matters: ask for an integer 0–100 rather than a decimal; average over
k=5 prompt variants with different exemplar subsets at temperature ~0.7 to break
ties; report the **distinct-value count and largest tie-block size** next to
every PR-AUC so a reader can see how much resolution the score actually had.
Few-shot exemplars come from **train only**.

### E5 — The one conditional route to a *usable artifact* (not back into this benchmark)

If and only if E1–E3 come back clean, or clean enough to bound the effect, the
**featurizer** path is reconsidered, and only under these conditions:

1. **Descriptive schema only.** Fields a human can verify from the source text:
   disease category, population acuity, line of therapy, intervention class. No
   field that asks the model to assess likelihood, quality, or prospects.
2. **Human-audited extraction accuracy.** Sample 50 extractions per cell and
   grade them against the source text. This yields a measured extraction
   accuracy that is **independent of the label**, which converts an
   unfalsifiable prediction claim into a checkable reading claim.
3. **Frozen artifact, in a tracked location.** Extract once, write to
   `artifacts/llm_features/<cell>.csv`, commit it. **Not** under `data/` — the
   repo's `.gitignore` excludes `/data/`, so an artifact written there could not
   be committed and the "frozen and reproducible" guarantee would be empty.
   The consuming method becomes "logistic regression on these frozen columns,"
   seed-stable and reproducible without an API. The model call is data
   preparation, not a method.
4. **Recorded provenance.** Resolved model version, prompt text, schema version,
   and extraction date live in the artifact's header. A future reader can tell
   exactly what produced the column.

Note the cost shape the earlier draft got wrong twice. The featurizer needs
extraction on **train and validation and test**, not a capped test sample: for
`outcome`/Phase2 that is 9,980 train-file rows (of which 7,984 train / 1,996
validation after the loader's split) plus 2,514 test, so **~12,494 calls**, not
the 10,498 previously stated. Validation rows cannot be skipped, because any
model selection on the frozen features needs them.

The payoff, if it survives: a disease/condition column, precisely what
`text_vs_tabular_signal.md` identified as absent from the 38-to-63-column
tabular view. **That use is fusion**, which `TEXT_PLAN.md` §2 puts out of scope,
so E5's output lands in the fusion follow-on. It does not re-enter the text-only
benchmark, and this experiment's name should not be read as implying it does.

---

## 5. Models called

The benchmark names no model anywhere; `src/methods/llm.py` says only "use the
`anthropic` SDK." Specifying it here, since an LLM result without its exact model
string and prompt is not a result.

| Role | Model requested | Why |
|---|---|---|
| E1 controls, E3 sweep, E5 extraction | `claude-haiku-4-5-20251001` | High call volume, and E1 is recall rather than judgment. Cheapest way to establish whether the effect exists at all. |
| E1 primary, E2, E4 | `claude-sonnet-5` | The default working model for the substantive arms. |
| E1 confirmation on one cell | `claude-opus-5` | If a larger model recalls more, that is direct evidence of memorization scaling with capacity, which strengthens the finding. Also serves as the ceiling check for E4. |

**Pinning.** Only the Haiku entry is a dated string; the other two are aliases
that can resolve to different snapshots over time, which is disqualifier 1 in
miniature. Do not invent date suffixes. Instead, **record the `model` field the
API returns on each response** — the resolved version, not the alias requested —
in the cache record and in the writeup. That is the only honest way to state what
produced a number, and it also detects a mid-study snapshot change.

Run E1 across all three: **contamination is expected to scale with model
capacity**, so the Haiku → Sonnet → Opus trend on ID-only recall is itself one of
the study's most informative results, not merely a robustness check.

Every call cached by `(model_string, prompt_hash, nct_id)` to a JSONL store under
`results_llm/cache/`, so re-runs and re-analyses cost nothing. Model string,
full prompt, temperature, and sample count are recorded per row.

---

## 6. Decision rules, written before the data

Stated in advance so the interpretation is not chosen after seeing the numbers.

| Result | Reading | Consequence |
|---|---|---|
| E1 beats both controls by a clear margin | Contamination demonstrated | Publish it as a benchmark-validity finding. E4 is reported only as "recall-inflated." E5 stays closed. |
| E1 at control level, but E2 shows a vintage gradient | Contamination likely, weaker evidence | Report as suggestive. E5 opens only with the extraction audit and a vintage-stratified re-analysis. |
| E1 at control level, E2 flat, E3 shows title dependence | Identity keying rather than outcome recall | Distinct and milder finding. E5 opens; extraction schema must exclude title-derived fields. |
| E1, E2, E3 all null | No detectable contamination on these cells | The strongest possible result for LLM work here. E5 opens fully. Still no leaderboard entry, because disqualifier 1 stands regardless. |
| High refusal rate throughout | Test inconclusive, not negative | Report the refusal rate and stop. Do not infer absence of contamination from unwillingness to answer. |

Note the asymmetry: **no result reopens the leaderboard question.** Even a
perfectly clean contamination finding leaves the version-pinning problem
untouched, so LLM methods stay out of `TEXT_PLAN.md`'s grid either way. What is
at stake is whether LLM-derived *features* become usable, and whether the
contamination finding gets published as a caution to others.

---

## 7. Budget

| Block | Volume | Notes |
|---|---|---|
| Block | Volume | Rough token cost |
|---|---|---|
| E1 primary + 2 controls, 3 models | ~600 answered rows × 3 arms × 3 models ≈ 5.4k calls | Prompts are tens of tokens in, a few out. **Well under 1M tokens total** — the cheapest experiment here despite the call count, and the one that decides everything. |
| E2 zero-shot full text, primary cell, bins 1–3 | ~1,776 calls at ~1.5k input tokens | ~2.7M input tokens |
| E2 replication on `outcome`/Phase2, bins 1–3 | ~2,200 calls | ~3.3M input tokens. Conditional on E2 agreeing with E1's per-bin arm. |
| E3 identifiability sweep | 3 input variants × ~600-row subsample ≈ 1.8k calls | ~1.5M input tokens (two of the three variants are shorter than full text) |
| E4 few-shot, k=5 prompt variants | 5x E2's volume on one cell ≈ 8.9k calls | ~15M+ input tokens once exemplars are in-prompt. The most expensive block, and the least load-bearing. |
| E5 extraction | ~12,494 calls on `outcome`/Phase2 (train + validation + test) | ~19M input tokens on that cell alone. Gated on E1–E3 and on a 50-row audit passing first. |

Token counts assume ~1.5k input tokens per serialized trial, which is an
estimate, not a measurement, and the pilot replaces it. Dollar figures are
deliberately omitted: they depend on the model chosen per block and on
prompt-caching hit rates, both of which the pilot measures directly.

**Pilot first, as before:** run E1's shuffled-ID control and 100 real IDs on
Haiku before anything else. That is a few hundred small calls, it costs almost
nothing, and it either shows a signal worth chasing or it does not. Nothing past
E1 is authorized without it.

Engineering effort is the larger cost: roughly 1–2 days for the harness
(prompting, caching, parsing, refusal accounting, binning, the control arms) plus
the E5 audit if it opens.

---

## 8. Writeup

`llm-contamination.md`, structured as: the question, the controls, E1 through E3
with their per-model trends, the decision-rule table from §6 with the actual
outcome marked, and the honest caveats. It stands alone and is not folded into
the text-benchmark writeup.

Two framings worth keeping in view while writing. If contamination is found,
this is a validity warning that applies to every paper doing LLM prediction on
clinicaltrials.gov data, not just to this benchmark, and that is a more useful
contribution than a leaderboard row would have been. If it is not found, the
result is a clean license for LLM-derived features in clinical-trial modeling,
which is worth having on the record too.

---

## Appendix A — measured facts this plan relies on

**Verified from `data/*/Phase*/{train,test}_{x,y}.csv`:**

**All label-conditional figures below are index-aligned** (`x.index.intersection(y.index)`),
matching `src/data/loader.py:_read_xy`. This matters: `test_x.csv` and
`test_y.csv` are in different row orders for `outcome`, so positional pairing
silently mismatches features to labels and produced a wrong "flat prevalence"
table in an earlier draft. Marginal figures (n, overall prevalence, NCT ranges)
are order-independent and were unaffected.

- `outcome`/Phase3: n_test 1,913, train prevalence 0.563, **test prevalence
  0.555** (1,062 of 1,913 positive; majority-class accuracy 0.555); NCT ordinals
  span 172 to 5,993,143 on train and 173 to 5,643,573 on test.
- `outcome`/Phase2: n_test 2,514, train prevalence 0.385, **test prevalence
  0.368** (majority-class accuracy 0.632); NCT ordinals span 309 to 6,161,792 on
  train and 340 to 5,760,703 on test.
- `outcome`/Phase3 test prevalence by vintage bin: **0.530 / 0.619 / 0.624 /
  0.337 / 0.000 / 0.000** with n = 779 / 635 / 362 / 89 / 41 / 7 and positives =
  413 / 393 / 226 / 30 / 0 / 0. Bins 1–3 collapsed: 0.581 over 1,776 rows (92.8%).
- `outcome`/Phase2 test prevalence by bin: **0.350 / 0.429 / 0.442 / 0.236 /
  0.000 / 0.000** with n = 945 / 780 / 475 / 212 / 83 / 19. Bins 1–3 collapsed:
  0.398 over 2,200 rows (87.5%).
- `mortality_rate_yn`/Phase1 test prevalence by bin: 0.294 / 0.510 / 0.367 /
  0.199 / 0.090 / 0.000 with n = 17 / 51 / 128 / 146 / 67 / 10. Confounded, but
  no more so than the `outcome` cells.
- `failure_reason`/Phase3 class distribution (train): Others 0.449, poor
  enrollment 0.326, efficacy 0.155, safety 0.070. Four classes, so any LLM
  multiclass arm would need a normalized 4-way distribution — one more reason
  the multiclass task is not part of this study.
- The benchmark's label columns: `outcome` for trial approval, `Y/N` for the
  event tasks, `failure_reason` for the multiclass task.
- No date column exists in any task's feature set, which is why NCT ordinal is
  used as the vintage proxy.

**Assumed, not verified:** the NCT-ordinal-to-year mapping (IDs are assigned
roughly sequentially at registration, but the mapping is not exact and no
per-trial dates are in this data drop); that published outcomes for these trials
appear in model pretraining data — §1's "entirely inside any current model's
pretraining window" is an inference from registration vintage, not a verified
fact about any training corpus, and E1 exists precisely because it cannot be
verified from outside; the ~1.5k-tokens-per-trial figure and every volume
estimate in §7.
