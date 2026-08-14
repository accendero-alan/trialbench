"""Render the benchmark's report figures from the run JSONs (+ the blend probe).

Five figures, each written as PNG (300 dpi, for print) and SVG (for the web), with
a CSV table-view twin beside it so every value is reachable without reading pixels:

  fig1_coverage_grid       16 methods x 20 task-phase cells, run status per square
  fig2_forest_outcome_p1   top-10 methods on one cell, point + bootstrap CI
  fig3_blend_curve         PR-AUC vs text blend weight, approval vs mortality
  fig4_ranking_swap        published mean -> common-cell mean slopegraph
  fig5_skill_vs_prevalence achieved PR-AUC vs the random-classifier baseline

Colors follow the validated two-hue categorical palette (blue #2a78d6 /
orange #eb6834; all-pairs CVD dE 24.7, normal-vision 33.6 -- both clear).
Only ever two series carry identity; everything else is emphasis (one hue) or
recessive ink, so no figure needs a third hue.

Usage:
    python -m experiments.make_figures --blend-json blend_curve.json --out-dir figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- palette ----
C = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834",
    "fill_weak": "#e8e7e1",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["axis"], "axes.linewidth": 0.8,
    "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "xtick.labelcolor": C["ink2"], "ytick.labelcolor": C["ink2"],
    "grid.color": C["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

TASK_ORDER = ["outcome", "mortality_rate_yn", "serious_adverse_rate_yn",
              "patient_dropout_rate_yn", "failure_reason"]
TASK_SHORT = {"outcome": "approval", "mortality_rate_yn": "mortality",
              "serious_adverse_rate_yn": "serious AE",
              "patient_dropout_rate_yn": "dropout", "failure_reason": "failure reason"}
PHASES = ["Phase1", "Phase2", "Phase3", "Phase4"]
# semantic grouping, so the complete Tier A block reads against the ragged deep rows
METHOD_ORDER = ["majority", "logreg_l1", "logreg_l2", "svm_linear", "knn",
                "random_forest", "extra_trees", "hist_gbm",
                "xgboost", "lightgbm", "catboost", "tfidf_logreg",
                "ft_transformer", "tabnet", "clinical_embeddings", "tabpfn"]
TIER_BREAK = 12   # rows 0-11 are Tier A, 12+ are the deep/optional methods


def _titles(ax, title, subtitle):
    """Place title + subtitle above the axes without them colliding.

    Offsets are computed from the axes' physical height, so a 1-line and a
    3-line subtitle both clear the title. (Using set_title(pad=) alongside a
    subtitle at a fixed axes fraction is what collided in the first draft.)
    """
    h_in = ax.get_figure().get_size_inches()[1] * ax.get_position().height
    sub_line = (8 * 1.5 / 72) / h_in
    title_line = (11 * 1.4 / 72) / h_in
    n = subtitle.count("\n") + 1
    y_sub = 1.0 + 0.35 * sub_line
    ax.text(0, y_sub, subtitle, transform=ax.transAxes, fontsize=8,
            color=C["ink2"], va="bottom", linespacing=1.5)
    ax.text(0, y_sub + n * sub_line + 0.35 * title_line, title, transform=ax.transAxes,
            fontsize=11, color=C["ink"], va="bottom", weight="bold")


def _spread(ys, min_gap):
    """Nudge label positions apart, preserving order, by the minimum needed."""
    out = np.asarray(ys, dtype=float).copy()
    order = np.argsort(out)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if out[j] - out[i] < min_gap:
            out[j] = out[i] + min_gap
    return out


def _save(fig, out_dir, name, table: pd.DataFrame | None = None):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    if table is not None:
        table.to_csv(os.path.join(out_dir, f"{name}_table.csv"), index=False)
    plt.close(fig)
    print(f"  wrote {name}.png / .svg" + (" / _table.csv" if table is not None else ""))


def load_runs(runs_glob: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(runs_glob):
        d = json.load(open(f))
        bs = (d.get("bootstrap") or {}).get("prauc") or {}
        rows.append({
            "task": d["task"], "phase": d["phase"], "method": d["method"],
            "status": d.get("status"), "prauc": bs.get("mean"),
            "lo": bs.get("lo"), "hi": bs.get("hi"),
            "n_test": d.get("n_test"), "num_classes": d.get("num_classes"),
        })
    if not rows:
        raise SystemExit(f"no run JSONs matched {runs_glob}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ fig 1 ----
def fig_coverage(df, out_dir):
    cells = [(t, p) for t in TASK_ORDER for p in PHASES]
    status = {(r.method, r.task, r.phase): r.status for r in df.itertuples()}
    n_m = len(METHOD_ORDER)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    rows_txt = []
    for i, m in enumerate(METHOD_ORDER):
        y = n_m - 1 - i                      # METHOD_ORDER[0] reads at the top
        n_ok = 0
        for j, (t, p) in enumerate(cells):
            s = status.get((m, t, p))
            if s == "ok":
                ax.add_patch(Rectangle((j + .09, y + .09), .82, .82, facecolor=C["s1"],
                                       edgecolor="none"))
                n_ok += 1
            elif s == "skipped":
                ax.add_patch(Rectangle((j + .09, y + .09), .82, .82,
                                       facecolor=C["fill_weak"], edgecolor=C["muted"],
                                       lw=0.5, hatch="////"))
            else:  # never reached -- no record of any kind
                ax.add_patch(Rectangle((j + .09, y + .09), .82, .82, facecolor="none",
                                       edgecolor=C["grid"], lw=0.8))
            rows_txt.append({"method": m, "task": t, "phase": p, "status": s or "missing"})
        ax.text(len(cells) + 0.4, y + .5, f"{n_ok}/20", va="center", fontsize=7.5,
                color=C["ink2"] if n_ok == 20 else C["s2"],
                weight="normal" if n_ok == 20 else "bold")

    y_break = n_m - TIER_BREAK               # boundary between Tier A and the rest
    ax.axhline(y_break, color=C["axis"], lw=0.8)
    ax.text(len(cells) + 1.9, (y_break + n_m) / 2, "Tier A", rotation=90, ha="center",
            va="center", fontsize=8, color=C["muted"])
    ax.text(len(cells) + 1.9, y_break / 2, "deep / optional", rotation=90, ha="center",
            va="center", fontsize=8, color=C["s2"])

    # two-tier x labels: phase digit per tick, task name under each group of four
    ax.set_xticks(np.arange(len(cells)) + 0.5)
    ax.set_xticklabels([p[-1] for _ in TASK_ORDER for p in PHASES], fontsize=7)
    for i, t in enumerate(TASK_ORDER):
        ax.annotate(TASK_SHORT[t], xy=(i * 4 + 2, -0.052), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8.5, color=C["ink2"])
        if i:
            ax.axvline(i * 4, color=C["axis"], lw=0.8)

    ax.set_xlim(0, len(cells) + 2.3)
    ax.set_ylim(0, n_m)
    ax.set_yticks(np.arange(n_m) + .5)
    ax.set_yticklabels(METHOD_ORDER[::-1], fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    _titles(ax, "279 of 320 grid cells have a run record",
            "One square per method x task x phase. Empty squares were never reached — the Tier B/C "
            "pass was cut\noff inside serious AE, so its four methods have no record of any kind "
            "for dropout or failure reason.")
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=9, mfc=C["s1"], mec="none", label="completed (269)"),
        Line2D([], [], marker="s", ls="", ms=9, mfc=C["fill_weak"], mec=C["muted"],
               label="skipped, no licence (10)"),
        Line2D([], [], marker="s", ls="", ms=9, mfc="none", mec=C["grid"],
               label="never reached (41)"),
    ], loc="upper left", bbox_to_anchor=(0, -0.135), ncol=3, fontsize=8,
        handletextpad=.4, columnspacing=2.0)
    _save(fig, out_dir, "fig1_coverage_grid", pd.DataFrame(rows_txt))


# ------------------------------------------------------------------ fig 2 ----
def fig_forest(df, out_dir, task="outcome", phase="Phase1", top_n=10, prevalence=None):
    g = (df[(df.task == task) & (df.phase == phase) & (df.status == "ok")]
         .dropna(subset=["prauc"]).sort_values("prauc", ascending=False).head(top_n))
    g = g.iloc[::-1].reset_index(drop=True)          # best at top when plotted
    lead = g.iloc[-1]
    n_overlap = int((g["hi"][:-1] >= lead["lo"]).sum())

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.axvspan(lead["lo"], lead["hi"], color=C["s1"], alpha=0.10, lw=0, zorder=0)
    for xv in (lead["lo"], lead["hi"]):
        ax.axvline(xv, color=C["s1"], lw=0.8, alpha=.5, zorder=1)
    if prevalence is not None:
        ax.axvline(prevalence, color=C["muted"], lw=1.0, zorder=1)
        ax.text(prevalence + .004, 0.35, f"chance = {prevalence:.2f}", fontsize=7.5,
                color=C["muted"], ha="left", va="center")

    for i, r in g.iterrows():
        is_lead = r["method"] == lead["method"]
        col = C["s1"] if is_lead else C["ink2"]
        ax.plot([r["lo"], r["hi"]], [i, i], color=col, lw=2, solid_capstyle="round",
                alpha=1.0 if is_lead else .5, zorder=3)
        ax.plot([r["prauc"]], [i], "o", ms=8, mfc=col, mec=C["surface"], mew=2, zorder=4)
        ax.text(r["hi"] + .006, i, f"{r['prauc']:.3f}", va="center", fontsize=7.5,
                color=C["ink"] if is_lead else C["ink2"])

    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g["method"], fontsize=8.5)
    ax.set_xlabel("test PR-AUC (point estimate, 95% bootstrap CI)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    _titles(ax, f"The ranking is inside the noise — {TASK_SHORT[task]} / {phase[:-1]} {phase[-1]}",
            f"All {n_overlap} of the other top-{top_n} methods have a confidence interval "
            f"overlapping the leader's (shaded band).\nThe rank order carries no statistical claim.")
    tbl = g.iloc[::-1][["method", "prauc", "lo", "hi"]].rename(
        columns={"prauc": "prauc_mean", "lo": "ci_lo", "hi": "ci_hi"})
    _save(fig, out_dir, "fig2_forest_outcome_p1", tbl)


# ------------------------------------------------------------------ fig 3 ----
def fig_blend(blend, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [C["s1"], C["s2"]]
    rows = []
    for k, rec in enumerate(blend):
        w = np.asarray(rec["w"]); y = np.asarray(rec["curve_test"])
        col = colors[k % 2]
        ax.plot(w, y, color=col, lw=2, zorder=3, label=rec["label"])
        # single-source endpoints, hollow
        ax.plot([0, 1], [y[0], y[-1]], "o", ms=7, mfc=C["surface"], mec=col, mew=2, zorder=4)
        # the validation-selected optimum -- the only one the protocol may use
        ws, ys = rec["w_star_valid"], rec["prauc_test_at_w_star"]
        ax.plot([ws], [ys], "o", ms=9, mfc=col, mec=C["surface"], mew=2, zorder=5)
        ax.annotate(f"w*={ws:.2f} · {ys:.3f}", (ws, ys), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=7.5, color=C["ink"])
        # where test would have peaked, if validation didn't pick the same place
        if abs(rec["w_argmax_test"] - ws) > 0.05:
            ax.plot([rec["w_argmax_test"]], [rec["prauc_test_at_argmax"]], "x", ms=6,
                    color=col, mew=1.5, zorder=5)
            ax.annotate("test peak", (rec["w_argmax_test"], rec["prauc_test_at_argmax"]),
                        textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=7, color=C["muted"])
        ax.text(-0.022, y[0], f"{y[0]:.3f}", ha="right", va="center", fontsize=7.5,
                color=C["ink2"])
        ax.text(1.022, y[-1], f"{y[-1]:.3f}", ha="left", va="center", fontsize=7.5,
                color=C["ink2"])
        ax.text(0.52, y[52] + (0.010 if k == 0 else -0.020), rec["label"],
                ha="center", va="bottom" if k == 0 else "top", fontsize=9, color=C["ink"])
        for wi, yi in zip(w, y):
            rows.append({"series": rec["label"], "text_weight": round(float(wi), 2),
                         "test_prauc": yi})

    ax.set_xlim(-0.075, 1.075)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("blend weight on text  ·  w = 0 is tabular consensus only, w = 1 is TF-IDF only")
    ax.set_ylabel("test PR-AUC")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5, handlelength=1.6)
    _titles(ax, "Fusing text with tabular beats either alone — and the best mix flips by task",
            "Blend weight w* is chosen on validation; the curve is scored on test. Approval wants "
            "mostly tabular\n(w*=0.29), mortality mostly text (w*=0.79) — and both peaks sit above "
            "both single-source endpoints.")
    _save(fig, out_dir, "fig3_blend_curve", pd.DataFrame(rows))


# ------------------------------------------------------------------ fig 4 ----
def fig_ranking_swap(df, out_dir, ref_method="ft_transformer"):
    ok = df[df.status == "ok"].dropna(subset=["prauc"])
    ref_cells = set(zip(*[ok[ok.method == ref_method][c] for c in ("task", "phase")]))
    published, common = {}, {}
    for m, g in ok.groupby("method"):
        if m == "majority":
            continue
        published[m] = g["prauc"].mean()
        sub = g[[(t, p) in ref_cells for t, p in zip(g["task"], g["phase"])]]
        if len(sub) == len(ref_cells):
            common[m] = sub["prauc"].mean()
    methods = sorted(common, key=lambda m: -common[m])
    hi = {ref_method: C["s2"], "tfidf_logreg": C["s1"]}

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    span = max(max(published.values()), max(common.values())) - \
        min(min(published.values()), min(common.values()))
    gap = span * 0.042
    lab_l = dict(zip(methods, _spread([published[m] for m in methods], gap)))
    lab_r = dict(zip(methods, _spread([common[m] for m in methods], gap)))

    for m in methods:
        col = hi.get(m, C["axis"])
        z = 4 if m in hi else 2
        ax.plot([0, 1], [published[m], common[m]], color=col, lw=2.2 if m in hi else 1.1,
                zorder=z, solid_capstyle="round")
        ax.plot([0, 1], [published[m], common[m]], "o", ms=7 if m in hi else 4.5,
                mfc=col if m in hi else C["surface"], mec=col, mew=1.5, zorder=z + 1)
        lab_col = C["ink"] if m in hi else C["muted"]
        wt = "bold" if m in hi else "normal"
        # leader lines, because de-collided labels no longer sit at their point
        for x0, x1, y_pt, y_lab, ha, txt in (
                (0, -0.04, published[m], lab_l[m], "right", f"{m}  {published[m]:.3f}"),
                (1, 1.04, common[m], lab_r[m], "left", f"{common[m]:.3f}  {m}")):
            if abs(y_lab - y_pt) > gap * 0.35:
                ax.plot([x0, x1], [y_pt, y_lab], color=C["grid"], lw=0.7, zorder=1)
            ax.text(x1 + (-0.015 if ha == "right" else 0.015), y_lab, txt, ha=ha,
                    va="center", fontsize=8, color=lab_col, weight=wt)

    ax.set_xlim(-0.56, 1.56)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["as published\n(each method's own cells)",
                        f"same {len(ref_cells)} cells\n({ref_method}'s coverage)"], fontsize=8.5)
    ax.set_ylabel("mean PR-AUC")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    _titles(ax, "The leaderboard's #1 is an artifact of partial coverage",
            f"Averaging each method over only the cells it ran puts {ref_method} first. Scored on "
            f"the same\n{len(ref_cells)} cells, tfidf_logreg leads it by 0.030 PR-AUC — the two "
            f"highlighted lines cross.")
    tbl = pd.DataFrame({"method": methods,
                        "published_mean": [published[m] for m in methods],
                        "common_cell_mean": [common[m] for m in methods]})
    _save(fig, out_dir, "fig4_ranking_swap", tbl)


# ------------------------------------------------------------------ fig 5 ----
def fig_skill_vs_prevalence(df, prev, out_dir):
    ok = df[df.status == "ok"].dropna(subset=["prauc"])
    best = ok.loc[ok.groupby(["task", "phase"])["prauc"].idxmax()]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([0, 1], [0, 1], color=C["muted"], lw=1.0, zorder=1)
    rows = []
    for r in best.itertuples():
        x = prev[f"{r.task}|{r.phase}"]
        is_fr = r.task == "failure_reason"
        col = C["s2"] if is_fr else C["s1"]
        ax.plot([x, x], [x, r.prauc], color=col, lw=1.0, alpha=.45, zorder=2)
        ax.plot([x], [r.prauc], "o", ms=8, mfc=col, mec=C["surface"], mew=1.8, zorder=4)
        rows.append({"task": r.task, "phase": r.phase, "prevalence": x,
                     "best_prauc": r.prauc, "skill_over_chance": r.prauc - x,
                     "best_method": r.method})
        if is_fr or r.prauc - x > 0.42 or x > 0.85:
            ax.annotate(f"{TASK_SHORT[r.task]} {r.phase[-1]}", (x, r.prauc),
                        textcoords="offset points", xytext=(9, -1), fontsize=7,
                        color=C["ink2"], va="center")

    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel("test class prevalence  =  what a random classifier scores")
    ax.set_ylabel("best PR-AUC achieved in the cell")
    ax.grid(zorder=0); ax.set_axisbelow(True)
    # label the diagonal at its true on-screen slope (depends on the axes aspect)
    p0, p1 = ax.transData.transform((0.55, 0.55)), ax.transData.transform((0.75, 0.75))
    ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    ax.text(0.66, 0.672, "chance (PR-AUC = prevalence)", fontsize=7.5, color=C["muted"],
            rotation=ang, rotation_mode="anchor", va="bottom", ha="center")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=8, mfc=C["s2"], mec=C["surface"], mew=1.5,
               label="failure reason (multiclass)"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=C["s1"], mec=C["surface"], mew=1.5,
               label="binary tasks"),
    ], loc="lower right", fontsize=8.5)
    _titles(ax, "Most of the apparent difficulty spread is class imbalance",
            "Each point is one task-phase cell at its best method; the stem is skill over chance. "
            "Dropout Phase 3\nscores 0.96 against a 0.91 prevalence, while failure reason's 0.42 "
            "clears a 0.25 baseline by far more.")
    _save(fig, out_dir, "fig5_skill_vs_prevalence", pd.DataFrame(rows))


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob",
                    default="results/extracted/trialbench/results/runs/*.json")
    ap.add_argument("--blend-json", default=None)
    ap.add_argument("--prevalence-json", default=None,
                    help="{'task|phase': prevalence}; produced by --write-prevalence")
    ap.add_argument("--write-prevalence", default=None,
                    help="compute prevalences from data/ and write them here, then exit")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    if args.write_prevalence:
        from src.data.loader import load_task_phase
        prev = {}
        for t in TASK_ORDER:
            for p in PHASES:
                td = load_task_phase(args.data_root, t, p, seed=42)
                y = td.y_test
                if td.task_type == "binary":
                    prev[f"{t}|{p}"] = float(y.mean())
                else:
                    # macro-OvR chance level = mean per-class prevalence over
                    # classes actually present in test (mirrors metrics._prauc)
                    sh = [float((y == c).mean()) for c in range(td.num_classes)
                          if (y == c).sum() > 0]
                    prev[f"{t}|{p}"] = float(np.mean(sh))
                print(f"  {t}/{p}: prevalence {prev[f'{t}|{p}']:.4f}")
        with open(args.write_prevalence, "w") as fh:
            json.dump(prev, fh, indent=2)
        print(f"wrote {args.write_prevalence}")
        return

    df = load_runs(args.runs_glob)
    print(f"loaded {len(df)} run records")
    prev = json.load(open(args.prevalence_json)) if args.prevalence_json else None

    fig_coverage(df, args.out_dir)
    fig_forest(df, args.out_dir,
               prevalence=prev.get("outcome|Phase1") if prev else None)
    if args.blend_json:
        fig_blend(json.load(open(args.blend_json)), args.out_dir)
    fig_ranking_swap(df, args.out_dir)
    if prev:
        fig_skill_vs_prevalence(df, prev, args.out_dir)
    else:
        print("  (skipped fig5 — pass --prevalence-json)")


if __name__ == "__main__":
    main()
