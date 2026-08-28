"""T28a -- Contamination, on published instruments (wave2-start-plan.md).

Runs before any T28 grid cell (§5's order of work: W0/W1/P15/P13.0/§6 gate
this, none of which are done from this environment -- no live AWS access,
§6's amendment is a sibling document). This script is written and unit-
tested against a fake Bedrock client (same DI posture as the harness itself,
src/bedrock/client.py's `boto_client` param) so it's ready to run for real
the moment those gates clear.

Four black-box instruments, all run per model, plus a blind (no-model-call)
baseline and an outcome/title recall probe -- not revision 1's home-made
probes:

1. N-Gram Coverage Attack (Hallinan et al. 2025, arXiv:2508.09603): prefix a
   trial's own summary text, sample one completion, score = trigram overlap
   with the true suffix.
2. The tabular memorization suite (Bordt et al., COLM 2024,
   arXiv:2404.06209): header / row / feature / first-token completion,
   averaged into one graded trial-level score.
3. Guided prompting (Golchin & Surdeanu, ICLR 2024, arXiv:2308.08493):
   completion under a dataset-naming instruction vs. one that doesn't name
   it; score = the difference.
4. The blind baseline: predicts the same pre/post-cutoff label from
   metadata alone (enrollment, phase, sponsor class), no model call. If it
   matches the detectors, the detectors are reading temporal drift, not
   memory (the plan's own framing) -- flagged per model as
   ``recognition_uninformative``.

Plus the campaign's outcome-recall probe (full serialized trial, recall
framing) and a title-recall probe (NCT id alone).

AUROC needs a target: there is no ground-truth "was this trial in the
training corpus" label anywhere, so -- deliberately, per the plan's own
awareness that "a naive pre/post-cutoff split is a known trap" -- each
detector's score is evaluated as a discriminator of pre- vs. post-cutoff
**registration date**, with the blind baseline as the control that catches
the trap if metadata alone can do just as well. Registration dates come
from the already-pinned AACT snapshot (src/data/aact.py's
``load_table("studies")``, joined by nct_id), not an NCT-ordinal proxy --
more accurate, and this repo already has it pinned. A model with no
published cutoff (Nova; see configs/bedrock_prices.yaml) gets no AUROC
against this label -- recorded, not guessed.

Usage:
    python -m experiments.t28a_contamination_probes --n-trials 40 \\
        --models amazon.nova-lite anthropic.claude-opus-4-5
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments._common import Timer, git_sha, write_artifact
from src.bedrock.cache import cache_get, cache_put
from src.bedrock.client import BedrockClient
from src.bedrock.meter import Meter
from src.bedrock.prices import is_verified, load_price_table, resolve_model_id
from src.data.aact import load_table as load_aact_table
from src.data.loader import TASKS, load_task_phase
from src.data.serialize import render_arm
from src.methods.llm import TASK_QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AMENDMENT_FILE = os.path.join(ROOT, "configs", "wave2_amendment.yaml")

BINARY_TASKS = [t for t, (_, _, ttype) in TASKS.items() if ttype == "binary"]
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]

OUT_PATH = "results/experiments/t28a_probe_gate.json"

# §T28a's pre-registered thresholds, verbatim.
OUTCOME_RECALL_SHRINK_THRESHOLD = 0.30
TITLE_RECALL_STRATIFY_THRESHOLD = 0.60
# "Within noise" for the blind-baseline-vs-detector comparison: not specified
# numerically by the plan, so fixed here, before any run, at a value smaller
# than what would itself be an interesting detector-vs-blind gap.
BLIND_BASELINE_NOISE_BAND = 0.05
# Family-wise error rate for the per-task outcome-discrimination test (F2),
# corrected across (models x tasks) once the full comparison count is known
# -- see run()'s alpha_corrected.
FAMILY_ALPHA = 0.05

PREFIX_SPLIT_FRAC = 0.6
MIN_SUMMARY_WORDS = 5


def _clopper_pearson_ci(hits: int, n: int, confidence: float = 0.95):
    """Exact (Clopper-Pearson) CI on a hit rate. None/None if there's
    nothing to score (F9: report the interval, not just the point
    estimate -- a rate of 0/200 and 1/200 look identical as point
    estimates but the interval makes the weakness of the evidence visible
    either way)."""
    if n <= 0:
        return None, None
    ci = binomtest(hits, n).proportion_ci(confidence_level=confidence, method="exact")
    return float(ci.low), float(ci.high)


def _default_models() -> list:
    with open(AMENDMENT_FILE) as f:
        return yaml.safe_load(f)["models"]


# ----------------------------------------------------------------------------
# Sample construction
# ----------------------------------------------------------------------------
def _pool_trials(data_root: str, n_trials: int, seed: int) -> pd.DataFrame:
    """Pools test-split trials across every (binary task, phase) cell,
    stratified by task with largest-remainder allocation (same technique
    src/data/subset.py uses for label strata, applied here to task strata),
    then samples `n_trials` from the pooled frame. failure_reason
    (multiclass) is excluded -- TASK_QUESTIONS (src/methods/llm.py) has no
    entry for it, same scope restriction llm_probability itself uses.
    """
    per_task = []
    for task in BINARY_TASKS:
        for phase in PHASES:
            try:
                td = load_task_phase(data_root, task, phase, seed=seed)
            except FileNotFoundError:
                continue
            df = td.X_test.copy()
            df["nct_id"] = df.index.astype(str)
            df["task"] = task
            df["phase"] = phase
            df["y"] = td.y_test
            per_task.append(df)
    if not per_task:
        raise FileNotFoundError(f"no data found under {data_root!r} for any binary task/phase")
    pooled = pd.concat(per_task, ignore_index=True)

    rng = np.random.default_rng(seed)
    counts = pooled.groupby("task", observed=True).size()
    raw_alloc = counts / counts.sum() * n_trials
    alloc = np.floor(raw_alloc).astype(int)
    shortfall = n_trials - alloc.sum()
    remainders = raw_alloc - alloc
    for task in remainders.sort_values(ascending=False).index[:shortfall]:
        alloc[task] += 1

    chosen = []
    for task, n in alloc.items():
        pool = pooled[pooled["task"] == task]
        n = min(int(n), len(pool))
        if n <= 0:
            continue
        idx = rng.choice(pool.index.values, size=n, replace=False)
        chosen.append(pool.loc[idx])
    sample = pd.concat(chosen, ignore_index=True) if chosen else pooled.iloc[0:0]
    return sample.reset_index(drop=True)


def _join_registration_dates(sample: pd.DataFrame, aact_loader=load_aact_table) -> pd.Series:
    """Real ``study_first_posted_date`` per nct_id from the pinned AACT
    snapshot, left null for any sampled trial the snapshot doesn't cover --
    those trials keep their raw per-instrument scores (a covariate for
    contamination layer 3, per the plan) but are excluded from AUROC."""
    studies = aact_loader("studies")[["nct_id", "study_first_posted_date"]].drop_duplicates("nct_id")
    merged = sample[["nct_id"]].merge(studies, on="nct_id", how="left")
    return pd.to_datetime(merged["study_first_posted_date"], errors="coerce")


def _cutoff_to_date(cutoff_str) -> "pd.Timestamp | None":
    if not cutoff_str:
        return None
    return pd.Period(cutoff_str, freq="M").end_time


# ----------------------------------------------------------------------------
# Text scoring primitives
# ----------------------------------------------------------------------------
def _words(text: str) -> list:
    return re.findall(r"\w+", (text or "").lower())


def _trigrams(text: str) -> set:
    w = _words(text)
    return set(zip(w, w[1:], w[2:])) if len(w) >= 3 else set()


def _trigram_recall(completion: str, true_suffix: str) -> float:
    true_tri = _trigrams(true_suffix)
    if not true_tri:
        return 0.0
    return len(true_tri & _trigrams(completion)) / len(true_tri)


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(_words(a)), set(_words(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse_number(text: str):
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", text or "")
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def _split_prefix_suffix(text: str, frac: float = PREFIX_SPLIT_FRAC):
    words = (text or "").strip().split()
    if len(words) < MIN_SUMMARY_WORDS:
        return None, None
    cut = max(1, int(len(words) * frac))
    return " ".join(words[:cut]), " ".join(words[cut:])


# ----------------------------------------------------------------------------
# Bedrock call wiring -- cache + meter, no probability elicitation (these
# are completion/recall prompts, not the disease-slot elicitation llm.py
# uses), so this doesn't go through src.bedrock.client.elicit_probability.
# ----------------------------------------------------------------------------
def _converse_cached(client: BedrockClient, results_dir: str, model_id: str, prompt: str,
                     meter: Meter, max_tokens: int = 60, temperature: float = 0.0) -> str:
    cached = cache_get(results_dir, model_id, prompt, temperature, "sync")
    if cached is not None:
        meter.record_cache_hit()
        return cached["response"]["text"]
    result = client.converse(model_id, prompt, temperature=temperature, max_tokens=max_tokens)
    meter.record_call(result.input_tokens, result.output_tokens, result.latency_secs,
                      throttle_count=result.retry_count)
    cache_put(results_dir, model_id, prompt, temperature, "sync",
             {"text": result.text, "input_tokens": result.input_tokens,
              "output_tokens": result.output_tokens})
    return result.text


# ----------------------------------------------------------------------------
# The four instruments
# ----------------------------------------------------------------------------
def ngram_coverage_score(client, results_dir, model_id, prefix, true_suffix, meter):
    """Hallinan et al. 2025: one completion per trial (the published method
    samples several; this uses one for budget -- a scoping decision, not a
    silent shortcut). Score = trigram recall against the true suffix."""
    prompt = ("Continue the following clinical trial summary exactly as it "
             "originally continues. Output only the continuation text, nothing else.\n\n"
             f"{prefix}")
    completion = _converse_cached(client, results_dir, model_id, prompt, meter, max_tokens=150)
    return _trigram_recall(completion, true_suffix)


def guided_prompting_delta(client, results_dir, model_id, prefix, true_suffix, meter):
    """Golchin & Surdeanu ICLR 2024: guided (dataset-named) completion minus
    general completion, both scored against the same true suffix."""
    general_prompt = ("Continue the following text exactly as it originally continues. "
                      "Output only the continuation, nothing else.\n\n" + prefix)
    guided_prompt = ("This text is a clinical trial summary from the TrialBench benchmark "
                     "dataset (ClinicalTrials.gov), test split. Continue it exactly as the "
                     "original continues. Output only the continuation, nothing else.\n\n" + prefix)
    general = _converse_cached(client, results_dir, model_id, general_prompt, meter, max_tokens=150)
    guided = _converse_cached(client, results_dir, model_id, guided_prompt, meter, max_tokens=150)
    return _trigram_recall(guided, true_suffix) - _trigram_recall(general, true_suffix)


def _feature_completion_score(client, results_dir, model_id, nct_id, true_enrollment, meter):
    prompt = (f"For the ClinicalTrials.gov trial {nct_id}, state only the trial's "
             f"target enrollment count as a single number.")
    text = _converse_cached(client, results_dir, model_id, prompt, meter, max_tokens=20)
    guess = _parse_number(text)
    try:
        true_val = float(true_enrollment)
    except (TypeError, ValueError):
        true_val = None
    if guess is None or not true_val or true_val <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(guess - true_val) / true_val)


def _row_completion_score(client, results_dir, model_id, row, meter):
    prompt = (f"A clinical trial has phase={row.get('phase')}, "
             f"lead sponsor class={row.get('sponsors/lead_sponsor/agency_class')}, "
             f"number of arms={row.get('number_of_arms')}. State only the trial's "
             f"target enrollment count as a single number.")
    text = _converse_cached(client, results_dir, model_id, prompt, meter, max_tokens=20)
    guess = _parse_number(text)
    try:
        true_val = float(row.get("enrollment"))
    except (TypeError, ValueError):
        true_val = None
    if guess is None or not true_val or true_val <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(guess - true_val) / true_val)


def _first_token_score(client, results_dir, model_id, prefix, true_suffix, meter):
    true_next_words = _words(true_suffix)
    if not true_next_words:
        return None
    prompt = f"{prefix}\n\nWhat is the single next word? Answer with only that one word."
    text = _converse_cached(client, results_dir, model_id, prompt, meter, max_tokens=10)
    guess = _words(text)
    return 1.0 if guess and guess[0] == true_next_words[0] else 0.0


def _header_completion_score(client, results_dir, model_id, columns, nct_id, meter):
    """Withheld column chosen deterministically per trial (hash of nct_id)
    so repeated runs against the cache are stable."""
    idx = hash(nct_id) % len(columns)
    withheld, remaining = columns[idx], columns[:idx] + columns[idx + 1:]
    prompt = ("This is a partial list of column names from a ClinicalTrials.gov-derived "
             "tabular dataset. One column name is missing. Name only the missing column, "
             "using the exact original naming convention.\n\n" + "\n".join(remaining))
    text = _converse_cached(client, results_dir, model_id, prompt, meter, max_tokens=20)
    return 1.0 if withheld.lower() in text.lower() else 0.0


def tabular_memorization_score(client, results_dir, model_id, row, prefix, true_suffix,
                                columns, meter):
    """Bordt et al., COLM 2024: header / row / feature / first-token
    completion, averaged into one graded trial-level score. Sub-scores are
    still recorded individually (feature completion is "the sharp one")."""
    sub = {
        "feature_completion": _feature_completion_score(
            client, results_dir, model_id, row["nct_id"], row.get("enrollment"), meter),
        "row_completion": _row_completion_score(client, results_dir, model_id, row, meter),
        "header_completion": _header_completion_score(
            client, results_dir, model_id, columns, row["nct_id"], meter),
    }
    ft = _first_token_score(client, results_dir, model_id, prefix, true_suffix, meter) if prefix else None
    sub["first_token"] = ft
    scored = [v for v in sub.values() if v is not None]
    sub["combined"] = float(np.mean(scored)) if scored else None
    return sub


def outcome_recall_probe(client, results_dir, model_id, task, trial_text, true_label, meter):
    """Full serialized trial, recall framing. Returns the raw response text,
    the parsed Yes/No answer (as 0/1, or None if the response didn't parse
    as a clear Yes/No -- unparseable is never counted as a hit), and
    whether the parsed answer matches the true label. Stored disaggregated
    (F7) rather than as a bare hit bool so the true-vs-predicted 2x2 needed
    for F2's per-task balanced accuracy / Fisher exact is readable directly
    from the artifact instead of reconstructed as hit XOR label."""
    question = TASK_QUESTIONS[task]
    prompt = ("This clinical trial has already completed and its results have been reported. "
             f"Based on what you know about this specific trial, state whether {question}. "
             "Answer with only Yes or No.\n\n" + trial_text)
    text = _converse_cached(client, results_dir, model_id, prompt, meter, max_tokens=10)
    lowered = text.lower()
    has_yes, has_no = "yes" in lowered, "no" in lowered
    if has_yes == has_no:  # both or neither present -> unparseable
        answer = None
    else:
        answer = 1 if has_yes else 0
    hit = None if answer is None else bool(answer == int(true_label))
    return {"raw_response": text, "parsed_answer": answer, "hit": hit}


def title_recall_hit(client, results_dir, model_id, nct_id, true_title, meter):
    if not true_title or not str(true_title).strip():
        return None
    prompt = f"What is the official title of the ClinicalTrials.gov trial {nct_id}? State only the title."
    text = _converse_cached(client, results_dir, model_id, prompt, meter, max_tokens=60)
    return _token_overlap(text, true_title) > 0.5


# ----------------------------------------------------------------------------
# The blind baseline -- no model call
# ----------------------------------------------------------------------------
def blind_baseline_auroc(sample: pd.DataFrame, cutoff_label: pd.Series, seed: int = 42):
    """Predicts the same pre/post-cutoff label from metadata alone
    (enrollment, phase, sponsor class), cross-validated (small N, so no
    separate held-out split). Returns None if there isn't enough data to
    fit or CV meaningfully -- never a fabricated number."""
    mask = cutoff_label.notna()
    y = cutoff_label[mask].astype(int).values
    if len(np.unique(y)) < 2:
        return None
    X = pd.get_dummies(
        sample.loc[mask, ["enrollment", "phase", "sponsors/lead_sponsor/agency_class"]],
        columns=["phase", "sponsors/lead_sponsor/agency_class"],
    ).fillna(0).values
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def _safe_auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    mask = ~np.isnan(scores) & pd.notna(labels)
    if mask.sum() < 2:
        return None
    y = labels[mask].astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores[mask]))


# ----------------------------------------------------------------------------
# F2 -- per-task, base-rate-invariant outcome discrimination
# ----------------------------------------------------------------------------
def per_task_outcome_discrimination(per_trial_scores):
    """Balanced accuracy + Fisher exact on the true-vs-predicted 2x2, one
    cell per task, from parseable outcome-recall answers only.

    Replaces raw pooled accuracy against a single 0.30 threshold: on the
    real two-model pilot, pooling produced a "significant" result for a
    model (nova) that had no signal on any individual task -- four tasks
    with base rates from 0.525 to 0.836 pooled into one 2x2 is Simpson's
    paradox, not evidence. Per-task balanced accuracy is invariant to each
    task's own base rate; the Fisher exact p-value on the same 2x2 is the
    significance test (Bonferroni-corrected across (models x tasks) by the
    caller, once every model's task set is known -- see run()).
    """
    by_task = {}
    for t in per_trial_scores:
        if t["outcome_parsed_answer"] is None:
            continue
        by_task.setdefault(t["task"], []).append((t["outcome_true_label"], t["outcome_parsed_answer"]))

    out = {}
    for task, pairs in sorted(by_task.items()):
        y_true = np.array([p[0] for p in pairs])
        y_pred = np.array([p[1] for p in pairs])
        n = len(pairs)
        majority_class_rate = float(max(np.mean(y_true), 1.0 - np.mean(y_true)))
        table = confusion_matrix(y_true, y_pred, labels=[0, 1])
        _, fisher_p = fisher_exact(table)
        balanced_acc = (
            float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) >= 2 else None
        )
        out[task] = {
            "n": n,
            "majority_class_rate": majority_class_rate,
            "balanced_accuracy": balanced_acc,
            "fisher_exact_p": float(fisher_p),
        }
    return out


# ----------------------------------------------------------------------------
# F1 -- branch decision built on title_recall (the discriminator), not on
# outcome_recall_hit (which a model can satisfy by prediction as much as by
# recall -- deepseek/mortality in the two-model pilot is exactly this case).
# ----------------------------------------------------------------------------
def _decide_branch(title_hits, title_n, title_recall_rate, outcome_recall_rate,
                    outcome_significant_tasks):
    """title_recall_hit has no predictive route: nothing about a trial's
    clinical content lets you infer its registered title from an opaque
    NCT ID alone (same logic as LLM_CONTAMINATION_PLAN.md E1's ID-only
    design). So "title recall significantly above zero" is tested as a
    literal point-null test (H0: p=0) -- if p really were exactly 0, a hit
    would be impossible with certainty, so any confirmed hit (Clopper-
    Pearson lower bound > 0, i.e. hits >= 1) is exact evidence p > 0, not
    an approximation. The CI is still carried on every result (F9) so a
    reviewer can see how thin that evidence is when n is small: a single
    hit at n=200 has a lower bound near 0.0003, which is real evidence of
    *something* but not the same weight as a high rate.

    Three outcomes, matching F1:
      - title recall significant -> recall demonstrated -> SHRINK/STRATIFY
        per the existing pre-registered thresholds.
      - title recall null, outcome discrimination significant on >=1 task
        (Bonferroni-corrected, caller's job) -> predictive signal, not
        recall -> PROCEED_AS_DESIGNED, effect size recorded in the reason.
      - neither -> no signal -> PROCEED_AS_DESIGNED.
    """
    title_lower, _ = _clopper_pearson_ci(title_hits or 0, title_n or 0)
    title_significant = bool(title_n) and bool(title_hits) and title_lower is not None and title_lower > 0.0

    if title_significant:
        if outcome_recall_rate is not None and outcome_recall_rate > OUTCOME_RECALL_SHRINK_THRESHOLD:
            return ("SHRINK_TO_UNRECOGNIZED_STRATUM",
                    f"title recall demonstrated ({title_hits}/{title_n} hits, 95% CI lower bound "
                    f"{title_lower:.4f} > 0) and outcome_recall_rate {outcome_recall_rate:.3f} > "
                    f"{OUTCOME_RECALL_SHRINK_THRESHOLD} baseline -- shrink to the unrecognized "
                    "stratum, primary role hands to T29")
        return ("STRATIFY",
                f"title recall demonstrated ({title_hits}/{title_n} hits, 95% CI lower bound "
                f"{title_lower:.4f} > 0) -- recall shown regardless of outcome_recall_rate, "
                "stratify and report both strata")

    if outcome_significant_tasks:
        tasks_desc = "; ".join(
            f"{task}: balanced_accuracy={s['balanced_accuracy']:.3f} vs 0.500 chance baseline "
            f"(Fisher exact p={s['fisher_exact_p']:.4g}, n={s['n']})"
            for task, s in sorted(outcome_significant_tasks.items())
        )
        return ("PROCEED_AS_DESIGNED",
                f"no title recall ({title_hits or 0}/{title_n or 0} hits) but predictive signal on "
                f"{len(outcome_significant_tasks)} task(s) surviving Bonferroni correction: "
                f"{tasks_desc} -- proceed as designed, not read as contamination")

    return ("PROCEED_AS_DESIGNED",
            f"no title recall ({title_hits or 0}/{title_n or 0} hits) and no task's outcome "
            "discrimination clears the Bonferroni-corrected significance threshold vs. the 0.500 "
            "chance baseline -- no signal detected on either probe")


# ----------------------------------------------------------------------------
# Per-model orchestration
# ----------------------------------------------------------------------------
def _run_one_model(client, results_dir, model_id, sample, columns, pre_cutoff_label, meter,
                   alpha_corrected=FAMILY_ALPHA):
    per_trial_scores = []
    for _, row in sample.iterrows():
        prefix, true_suffix = _split_prefix_suffix(row.get("brief_summary/textblock"))

        ngram = ngram_coverage_score(client, results_dir, model_id, prefix, true_suffix, meter) \
            if prefix else None
        guided_delta = guided_prompting_delta(client, results_dir, model_id, prefix, true_suffix, meter) \
            if prefix else None
        tabular = tabular_memorization_score(client, results_dir, model_id, row, prefix, true_suffix,
                                             columns, meter)

        rendered = render_arm(row, "L7", row["nct_id"])
        outcome_probe = outcome_recall_probe(client, results_dir, model_id, row["task"], rendered.text,
                                             row["y"], meter)
        title_hit = title_recall_hit(client, results_dir, model_id, row["nct_id"],
                                     row.get("brief_title"), meter)

        per_trial_scores.append({
            "nct_id": row["nct_id"], "task": row["task"], "ngram_score": ngram,
            "guided_delta": guided_delta, "tabular": tabular,
            "outcome_true_label": int(row["y"]),
            "outcome_raw_response": outcome_probe["raw_response"],
            "outcome_parsed_answer": outcome_probe["parsed_answer"],
            "outcome_recall_hit": outcome_probe["hit"],
            "title_recall_hit": title_hit,
        })

    ngram_scores = [t["ngram_score"] for t in per_trial_scores]
    guided_deltas = [t["guided_delta"] for t in per_trial_scores]
    tabular_scores = [t["tabular"]["combined"] for t in per_trial_scores]
    outcome_hits = [t["outcome_recall_hit"] for t in per_trial_scores if t["outcome_recall_hit"] is not None]
    title_hits = [t["title_recall_hit"] for t in per_trial_scores if t["title_recall_hit"] is not None]

    detector_aurocs = {
        "ngram_coverage": _safe_auroc([s if s is not None else np.nan for s in ngram_scores], pre_cutoff_label),
        "guided_prompting_delta": _safe_auroc([s if s is not None else np.nan for s in guided_deltas], pre_cutoff_label),
        "tabular_memorization": _safe_auroc([s if s is not None else np.nan for s in tabular_scores], pre_cutoff_label),
    }
    computable = [v for v in detector_aurocs.values() if v is not None]
    best_detector_auroc = max(computable) if computable else None

    blind_auroc = blind_baseline_auroc(sample, pre_cutoff_label)
    # F3: tri-state. The old two-value form read `None`-vs-`None` (control
    # not computed at all -- true on the real two-model artifact, both
    # models) as `False` ("ran, and the detectors are citable"), which is a
    # silent false negative. `None` here means "not computed"; only a real
    # comparison of two real numbers can say true or false.
    if best_detector_auroc is None or blind_auroc is None:
        recognition_uninformative = None
    else:
        recognition_uninformative = abs(best_detector_auroc - blind_auroc) <= BLIND_BASELINE_NOISE_BAND

    outcome_recall_rate = float(np.mean(outcome_hits)) if outcome_hits else None
    title_recall_rate = float(np.mean(title_hits)) if title_hits else None
    title_hit_count = int(np.sum(title_hits)) if title_hits else 0
    title_recall_ci = _clopper_pearson_ci(title_hit_count, len(title_hits))

    per_task_outcome = per_task_outcome_discrimination(per_trial_scores)
    outcome_significant_tasks = {
        task: stats for task, stats in per_task_outcome.items()
        if stats["fisher_exact_p"] is not None and stats["fisher_exact_p"] < alpha_corrected
        and stats["balanced_accuracy"] is not None and stats["balanced_accuracy"] > 0.5
    }
    branch, branch_reason = _decide_branch(
        title_hit_count, len(title_hits), title_recall_rate, outcome_recall_rate,
        outcome_significant_tasks,
    )

    # Cross-instrument agreement: pairwise Spearman correlation between the
    # three detector score series (pooled per model, over trials where both
    # scores exist).
    series = {"ngram_coverage": pd.Series(ngram_scores, dtype=float),
             "guided_prompting_delta": pd.Series(guided_deltas, dtype=float),
             "tabular_memorization": pd.Series(tabular_scores, dtype=float)}
    names = list(series)
    agreement = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = series[names[i]], series[names[j]]
            valid = a.notna() & b.notna()
            corr = float(a[valid].corr(b[valid], method="spearman")) if valid.sum() >= 3 else None
            agreement[f"{names[i]}_vs_{names[j]}"] = corr

    price_table = load_price_table()
    meter_summary = meter.summary(price_table, model_id, "sync")
    meter_summary["price_verified"] = is_verified(model_id, price_table)

    return {
        "per_trial": per_trial_scores,
        "outcome_recall_rate": outcome_recall_rate,
        "outcome_recall_rate_note": (
            "descriptive only -- raw pooled accuracy, inflated by task prevalence "
            "(pooled base rate was 0.615 on the two-model pilot). Not used for the branch "
            "decision; see per_task_outcome_discrimination for the base-rate-invariant "
            "per-task figures the decision is actually made from."
        ),
        "title_recall_rate": title_recall_rate,
        "title_recall_ci_95": {"lower": title_recall_ci[0], "upper": title_recall_ci[1]},
        "per_task_outcome_discrimination": per_task_outcome,
        "outcome_significant_tasks": list(outcome_significant_tasks),
        "detector_aurocs": detector_aurocs, "blind_baseline_auroc": blind_auroc,
        "recognition_uninformative": recognition_uninformative,
        "cross_instrument_agreement": agreement,
        "branch": branch, "branch_reason": branch_reason, "meter": meter_summary,
    }


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def run(data_root="data", results_dir="results", n_trials=40, models=None, seed=42,
       region="us-west-2", boto_client=None, aact_loader=load_aact_table, out_path=OUT_PATH) -> dict:
    models = models or _default_models()
    price_table = load_price_table()

    with Timer() as t:
        sample = _pool_trials(data_root, n_trials, seed)
        columns = [c for c in sample.columns if c not in ("nct_id", "task", "phase", "y")]
        registration_dates = _join_registration_dates(sample, aact_loader)

        # F2: Bonferroni correction across the full (models x tasks) family,
        # computed once up front so every model's per-task significance test
        # uses the same corrected alpha -- not each model's own task count,
        # which would under-correct as more models are added.
        n_tasks = int(sample["task"].nunique())
        n_comparisons = max(1, len(models) * n_tasks)
        alpha_corrected = FAMILY_ALPHA / n_comparisons

        per_model = {}
        for model_id in models:
            cutoff = price_table["models"].get(model_id, {}).get("cutoff")
            cutoff_date = _cutoff_to_date(cutoff)
            if cutoff_date is None:
                pre_cutoff_label = pd.Series([pd.NA] * len(sample))
                cutoff_note = ("no published cutoff for this model (or unpriced) -- AUROC "
                               "against the pre/post-cutoff label not computed")
            else:
                pre_cutoff_label = (registration_dates < cutoff_date).where(registration_dates.notna())
                cutoff_note = None

            # `models` holds price-table keys (e.g. "anthropic.claude-opus-4-5");
            # Converse needs the concrete id/inference-profile ARN instead --
            # several ladder models reject their bare id (confirmed live,
            # 2026-08-27). Resolve once per model; per_model stays keyed by
            # the short key (readability), api_model_id is what's actually
            # called and cached.
            api_model_id = resolve_model_id(model_id, price_table)
            client = BedrockClient(region=region, boto_client=boto_client)
            meter = Meter()
            result = _run_one_model(client, results_dir, api_model_id, sample, columns,
                                    pre_cutoff_label, meter, alpha_corrected=alpha_corrected)
            result["cutoff"] = cutoff
            result["cutoff_note"] = cutoff_note
            result["api_model_id"] = api_model_id
            result["n_trials_with_registration_date"] = int(registration_dates.notna().sum())
            per_model[model_id] = result
            print(f"  {model_id}: branch={result['branch']} ({result['branch_reason']}), "
                 f"uninformative={result['recognition_uninformative']}, "
                 f"dollars_realized={result['meter']['dollars_realized']}", flush=True)

    artifact = {
        "test_id": "T28a",
        "claim_at_stake": "how much of a model's apparent skill on TrialBench trials is "
                          "recognition of trials it already saw during training, not "
                          "genuine prediction",
        "inputs": {
            "n_trials": n_trials, "models": models, "seed": seed, "data_root": data_root,
            "outcome_recall_shrink_threshold": OUTCOME_RECALL_SHRINK_THRESHOLD,
            "title_recall_stratify_threshold": TITLE_RECALL_STRATIFY_THRESHOLD,
            "title_recall_stratify_threshold_note": (
                "recorded for provenance, no longer gates STRATIFY directly since F1: "
                "STRATIFY now fires whenever title recall is statistically significant "
                "(any magnitude), not only above this raw-rate threshold."
            ),
            "blind_baseline_noise_band": BLIND_BASELINE_NOISE_BAND,
            "registration_date_source": "AACT snapshot (src/data/aact.py load_table('studies'))",
            "bonferroni_correction": {
                "family_alpha": FAMILY_ALPHA, "n_models": len(models), "n_tasks": n_tasks,
                "n_comparisons": n_comparisons, "alpha_corrected": alpha_corrected,
            },
        },
        "n_trials_sampled": len(sample),
        "per_model": per_model,
        "decision_rule": (
            "Per model, built on title_recall_hit as the discriminator (F1 -- it has no "
            "predictive route, unlike outcome_recall_hit, which a model can satisfy by "
            "prediction as well as by recall): if title recall is significantly above zero "
            "(Clopper-Pearson 95% CI lower bound > 0 vs. the 0.0 point-null baseline), recall "
            "is demonstrated -> SHRINK_TO_UNRECOGNIZED_STRATUM if outcome_recall_rate > "
            f"{OUTCOME_RECALL_SHRINK_THRESHOLD} (baseline: that raw rate), else STRATIFY. If "
            "title recall is null: outcome discrimination is tested per task via balanced "
            "accuracy vs. the 0.500 chance baseline, significance by Fisher exact on that "
            "task's true-vs-predicted 2x2, Bonferroni-corrected across the full "
            "(models x tasks) family (alpha_corrected recorded in inputs.bonferroni_correction) "
            "-- any task clearing that bar is a predictive signal, not contamination -> "
            "PROCEED_AS_DESIGNED with the effect size recorded in branch_reason. No task "
            "clearing it -> PROCEED_AS_DESIGNED, no signal on either probe. "
            "Independently: if the blind baseline's AUROC is within "
            f"{BLIND_BASELINE_NOISE_BAND} of the best detector's AUROC on a model, "
            "recognition_uninformative=true and that model's detector scores are not "
            "citable as evidence of memorization (temporal drift is the more likely "
            "explanation); recognition_uninformative=null (not the old silent false-negative "
            "false) when either AUROC wasn't computable at all -- see F3."
        ),
        "verdict": "descriptive -- see per_model.<model_id>.branch, .branch_reason, and "
                  ".recognition_uninformative; no single pass/fail verdict for this test.",
        "git_sha": git_sha(),
        "wall_clock_secs": round(t.secs, 1),
    }
    write_artifact(out_path, artifact)
    print(f"\nwrote {out_path}")
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Bedrock model ids to probe; defaults to configs/wave2_amendment.yaml's "
                         "five pre-registered models.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--out-path", default=OUT_PATH,
                    help=f"artifact output path (default {OUT_PATH!r}). F5: the default is a "
                         "single file holding all models under per_model -- pass a distinct "
                         "path for a partial/scratch run so it can't clobber the tracked "
                         "artifact of a completed one.")
    args = ap.parse_args()
    run(data_root=args.data_root, results_dir=args.results_dir, n_trials=args.n_trials,
        models=args.models, seed=args.seed, region=args.region, out_path=args.out_path)


if __name__ == "__main__":
    main()
