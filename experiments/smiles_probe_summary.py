"""Condense `smiles_signal_probe --out ...` JSON into the cross-cell tables.

Three questions, one table each:
  1. chemistry vs. the presence confound   (molecule blocks - presence control)
  2. chemistry vs. drug memorization       (morgan_fp - drug_id, full test and
                                            on novel-scaffold test rows)
  3. does it add over tabular              (fusion - tabular reference)

Also checks the tabular reference against the benchmark's own LightGBM in
results_long.csv, so a drifted baseline can't quietly inflate question 3.

Usage:
    python -m experiments.smiles_probe_summary probe.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

LEADERBOARD = os.path.join("results", "extracted", "trialbench", "results", "results_long.csv")


def load(path: str) -> pd.DataFrame:
    with open(path) as fh:
        cells = json.load(fh)
    rows = []
    for c in cells:
        for r in c["results"]:
            rows.append({
                "task": c["task"], "phase": c["phase"],
                "n_test": c["base"]["n_test"],
                "pos_rate": c["base"]["test_pos_rate"],
                "has_mol": c["base"]["test_has_mol_frac"],
                "n_novel": c["novel_scaffold_test_rows"],
                "block": r["block"], "prauc": r["test_prauc"],
                "lo": r["test_prauc_lo"], "hi": r["test_prauc_hi"],
                "prauc_hasmol": r.get("test_prauc_hasmol", np.nan),
                "prauc_novel": r.get("test_prauc_novel_scaffold", np.nan),
            })
    return pd.DataFrame(rows)


def _wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    w = df.pivot_table(index=["task", "phase"], columns="block", values=value)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    df = load(args.json)
    ctl, ref = "presence (control)", "tabular (reference)"
    pr = _wide(df, "prauc")
    nv = _wide(df, "prauc_novel")
    meta = (df.groupby(["task", "phase"])[["n_test", "pos_rate", "has_mol", "n_novel"]]
            .first())

    out = []

    def emit(title, frame, fmt="{:+.3f}"):
        out.append(f"\n### {title}\n")
        out.append(frame.to_string(float_format=lambda v: fmt.format(v) if pd.notna(v) else "   .  "))
        out.append("")

    # -- 0. baseline integrity ------------------------------------------------
    if os.path.exists(LEADERBOARD):
        lb = pd.read_csv(LEADERBOARD)
        lb = (lb[lb["method"] == "lightgbm"].set_index(["task", "phase"])["prauc_mean"]
              .rename("leaderboard_lightgbm"))
        chk = pd.concat([pr[ref].rename("probe_tabular"), lb], axis=1, join="inner")
        chk["diff"] = chk["probe_tabular"] - chk["leaderboard_lightgbm"]
        emit("0. Baseline integrity — probe's tabular block vs the benchmark's lightgbm",
             chk, "{:+.3f}")
        out.append(f"max |diff| = {chk['diff'].abs().max():.3f}, "
                   f"mean |diff| = {chk['diff'].abs().mean():.3f}\n")

    # -- 1. presence confound ------------------------------------------------
    mol_blocks = ["descriptors", "morgan_fp", "drug_id (control)", "scaffold"]
    q1 = pr[mol_blocks].sub(pr[ctl], axis=0)
    q1.insert(0, "presence_PR", pr[ctl])
    emit("1. Molecule blocks minus the presence control (test PR-AUC)", q1)

    # -- 2. chemistry vs memorization ---------------------------------------
    q2 = pd.DataFrame({
        "fp - drug_id (all test)": pr["morgan_fp"] - pr["drug_id (control)"],
        "n_novel": meta["n_novel"],
        "novel: presence": nv[ctl],
        "novel: descriptors": nv["descriptors"],
        "novel: morgan_fp": nv["morgan_fp"],
        "novel: drug_id": nv["drug_id (control)"],
        "novel: desc - drug_id": nv["descriptors"] - nv["drug_id (control)"],
    })
    emit("2. Chemistry vs drug memorization", q2)

    # -- 3. does it add over tabular ----------------------------------------
    fus = ["tabular+descriptors", "tabular+morgan_fp"]
    q3 = pr[fus].sub(pr[ref], axis=0)
    q3.insert(0, "tabular_PR", pr[ref])
    # is the fusion gain outside the tabular block's own bootstrap CI?
    tab_ci = df[df["block"] == ref].set_index(["task", "phase"])[["lo", "hi"]]
    q3["beats_tab_CI"] = [
        "yes" if max(a, b) > tab_ci.loc[i, "hi"] else ("worse" if max(a, b) < tab_ci.loc[i, "lo"] else "no")
        for i, a, b in zip(q3.index, q3[fus[0]] + pr[ref], q3[fus[1]] + pr[ref])
    ]
    emit("3. Fusion minus tabular reference (test PR-AUC)", q3)

    text = "\n".join(str(x) for x in out)
    print(text)
    if args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(text)
        print(f"\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
