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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments._common import Timer, git_sha, write_artifact
from src.bedrock.cache import cache_get, cache_put
from src.bedrock.client import BedrockClient
from src.bedrock.meter import Meter
from src.bedrock.prices import is_verified, load_price_table
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

PREFIX_SPLIT_FRAC = 0.6
MIN_SUMMARY_WORDS = 5


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


def outcome_recall_hit(client, results_dir, model_id, task, trial_text, true_label, meter):
    """Full serialized trial, recall framing. Returns True/False (correct
    recall of the actual reported label) or None if the response didn't
    parse as a clear Yes/No -- unparseable is not counted as a hit."""
    question = TASK_QUESTIONS[task]
    prompt = ("This clinical trial has already completed and its results have been reported. "
             f"Based on what you know about this specific trial, state whether {question}. "
             "Answer with only Yes or No.\n\n" + trial_text)
    text = _converse_cached(client, results_dir, model_id, prompt, meter, max_tokens=10)
    lowered = text.lower()
    has_yes, has_no = "yes" in lowered, "no" in lowered
    if has_yes == has_no:  # both or neither present -> unparseable
        return None
    answer = 1 if has_yes else 0
    return bool(answer == int(true_label))


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
# Per-model orchestration
# ----------------------------------------------------------------------------
def _run_one_model(client, results_dir, model_id, sample, columns, pre_cutoff_label, meter):
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
        outcome_hit = outcome_recall_hit(client, results_dir, model_id, row["task"], rendered.text,
                                         row["y"], meter)
        title_hit = title_recall_hit(client, results_dir, model_id, row["nct_id"],
                                     row.get("brief_title"), meter)

        per_trial_scores.append({
            "nct_id": row["nct_id"], "ngram_score": ngram, "guided_delta": guided_delta,
            "tabular": tabular, "outcome_recall_hit": outcome_hit, "title_recall_hit": title_hit,
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
    recognition_uninformative = (
        best_detector_auroc is not None and blind_auroc is not None
        and abs(best_detector_auroc - blind_auroc) <= BLIND_BASELINE_NOISE_BAND
    )

    outcome_recall_rate = float(np.mean(outcome_hits)) if outcome_hits else None
    title_recall_rate = float(np.mean(title_hits)) if title_hits else None

    if outcome_recall_rate is not None and outcome_recall_rate > OUTCOME_RECALL_SHRINK_THRESHOLD:
        branch = "SHRINK_TO_UNRECOGNIZED_STRATUM"
    elif (title_recall_rate is not None and title_recall_rate > TITLE_RECALL_STRATIFY_THRESHOLD
          and (outcome_recall_rate is None or outcome_recall_rate < OUTCOME_RECALL_SHRINK_THRESHOLD)):
        branch = "STRATIFY"
    else:
        branch = "PROCEED_AS_DESIGNED"

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
        "outcome_recall_rate": outcome_recall_rate, "title_recall_rate": title_recall_rate,
        "detector_aurocs": detector_aurocs, "blind_baseline_auroc": blind_auroc,
        "recognition_uninformative": recognition_uninformative,
        "cross_instrument_agreement": agreement,
        "branch": branch, "meter": meter_summary,
    }


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def run(data_root="data", results_dir="results", n_trials=40, models=None, seed=42,
       region="us-east-1", boto_client=None, aact_loader=load_aact_table, out_path=OUT_PATH) -> dict:
    models = models or _default_models()
    price_table = load_price_table()

    with Timer() as t:
        sample = _pool_trials(data_root, n_trials, seed)
        columns = [c for c in sample.columns if c not in ("nct_id", "task", "phase", "y")]
        registration_dates = _join_registration_dates(sample, aact_loader)

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

            client = BedrockClient(region=region, boto_client=boto_client)
            meter = Meter()
            result = _run_one_model(client, results_dir, model_id, sample, columns,
                                    pre_cutoff_label, meter)
            result["cutoff"] = cutoff
            result["cutoff_note"] = cutoff_note
            result["n_trials_with_registration_date"] = int(registration_dates.notna().sum())
            per_model[model_id] = result
            print(f"  {model_id}: outcome_recall={result['outcome_recall_rate']}, "
                 f"branch={result['branch']}, uninformative={result['recognition_uninformative']}, "
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
            "blind_baseline_noise_band": BLIND_BASELINE_NOISE_BAND,
            "registration_date_source": "AACT snapshot (src/data/aact.py load_table('studies'))",
        },
        "n_trials_sampled": len(sample),
        "per_model": per_model,
        "decision_rule": (
            "Per model: outcome_recall_rate > 0.30 -> SHRINK_TO_UNRECOGNIZED_STRATUM "
            "(that model's T28 grid shrinks to the unrecognized stratum, primary role "
            "hands to T29). Elif title_recall_rate > 0.60 and outcome_recall_rate < 0.30 "
            "-> STRATIFY (proceed, report both strata). Else PROCEED_AS_DESIGNED. "
            "Independently: if the blind baseline's AUROC is within "
            f"{BLIND_BASELINE_NOISE_BAND} of the best detector's AUROC on a model, "
            "recognition_uninformative=true and that model's detector scores are not "
            "citable as evidence of memorization (temporal drift is the more likely "
            "explanation)."
        ),
        "verdict": "descriptive -- see per_model.<model_id>.branch and "
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
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    run(data_root=args.data_root, results_dir=args.results_dir, n_trials=args.n_trials,
        models=args.models, seed=args.seed, region=args.region)


if __name__ == "__main__":
    main()
