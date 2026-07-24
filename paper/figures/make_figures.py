#!/usr/bin/env python3
# Copyright 2026.
"""W1 — the three manuscript figures, from committed artifacts only.

Every number plotted is either in paper/NUMBERS.json or derived here from a
committed source that the figure's CSV records (source_file + commit). Nothing
is typed in by hand. Colours are the Okabe-Ito colourblind-safe set, assigned to
entities in a fixed order (colour follows the condition, never its rank).

  Fig 1  P(diagnosis) by target_compatibility (high/neutral/low) x condition x
         level, Wilson 95% CI. Derived from scored_full.jsonl @ c4a5ce8
         (behavioral, read-only) -> fig1_data.csv.
  Fig 2  C1 anatomy: Set F loading for plausible / flagged-without-mention /
         flagged-masked / flagged-total, with the registered paired-contrast
         CIs (C1, A1, A2). From results_phase1.json (bc4320a) +
         results_phase1_appendix.json (f39df1f) -> fig2_data.csv.
  Fig 3  Phase 2b: three arms x two DVs; DV1 diagnosis flat at 1.000, DV2
         mention 0.475 -> 0.285 (B1) with B3 at 0.475. Wilson CI on n=200.
         From results_phase2b.json (79d0664) -> fig3_data.csv.

Each figure -> .pdf + .png + _data.csv in paper/figures/.
Run:  .venv/Scripts/python.exe paper/figures/make_figures.py <scratch_beh_dir>
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.patches import Patch          # noqa: E402

HERE = Path(__file__).resolve().parent
WP = HERE.parents[1]
BEH = WP.parent / "reification-gradient"

# Okabe-Ito colourblind-safe palette, fixed entity assignment (never by rank).
COND_COLOR = {
    "DN_plausible": "#0072B2",   # blue
    "DN_flagged":   "#E69F00",   # orange
    "incoherent":   "#009E73",   # bluish green
    "real_anchor":  "#D55E00",   # vermillion
}
COND_LABEL = {"DN_plausible": "DN plausible", "DN_flagged": "DN flagged",
              "incoherent": "incoherent", "real_anchor": "real anchor"}
LEVELS = ["L1_forum", "L2_coach_blog", "L3_wiki", "L4_preprint", "L5_pseudodsm"]
LEVEL_SHORT = ["L1", "L2", "L3", "L4", "L5"]
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150,
    "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ Fig 1

def fig1(beh: Path):
    resp = [r for r in load_jsonl(beh / "responses_770fa9c.jsonl") if "model" in r]
    scored = load_jsonl(beh / "scored_c4a5ce8.jsonl")
    key = lambda r: (r["model"], r["disorder"], r["level"], r["vignette"], r["rep"])
    compat = {key(r): r["target_compatibility"] for r in resp}
    agg = {}                                   # (compat, disorder, level) -> [k, n]
    for s in scored:
        if s.get("error"):
            continue
        c = compat.get(key(s))
        if c is None:
            continue
        kk = (c, s["disorder"], s["level"])
        a = agg.setdefault(kk, [0, 0])
        a[0] += int(s["diagnosis"])
        a[1] += 1

    rows = []
    for (c, d, l), (k, n) in sorted(agg.items()):
        p, lo, hi = wilson(k, n)
        rows.append([c, d, l, LEVELS.index(l) + 1, k, n,
                     round(p, 4), round(lo, 4), round(hi, 4)])
    write_csv(HERE / "fig1_data.csv",
              ["target_compatibility", "disorder", "level", "level_num",
               "k_diagnosis1", "n", "p", "ci95_lo", "ci95_hi"], rows)

    compats = ["high", "neutral", "low"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, comp in zip(axes, compats):
        for d in ["DN_plausible", "real_anchor", "incoherent", "DN_flagged"]:
            xs, ys, los, his = [], [], [], []
            for i, l in enumerate(LEVELS, 1):
                rec = agg.get((comp, d, l))
                if not rec:
                    continue
                p, lo, hi = wilson(rec[0], rec[1])
                xs.append(i); ys.append(p); los.append(p - lo); his.append(hi - p)
            ax.errorbar(xs, ys, yerr=[los, his], color=COND_COLOR[d], lw=2.0,
                        marker="o", ms=5, capsize=2.5, elinewidth=1.0,
                        label=COND_LABEL[d], zorder=3)
        ax.set_title(f"target = {comp}", fontsize=10, color=INK)
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(LEVEL_SHORT)
        ax.set_xlabel("legitimacy level")
        ax.set_ylim(-0.03, 1.03)
        ax.axhline(1.0, color=GRID, lw=0.8, ls="--", zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("P(diagnosis)  [Wilson 95% CI]")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    fig.suptitle("Figure 1 — P(diagnosis) by condition × legitimacy level, stratified by vignette target",
                 fontsize=11, y=1.02, x=0.02, ha="left", color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig1_pdiagnosis.{ext}")
    plt.close(fig)
    return len(rows)


# ------------------------------------------------------------------ Fig 2

def fig2():
    r1 = json.loads((WP / "phase1/data/results_phase1.json").read_text(encoding="utf-8"))
    ap = json.loads((WP / "phase1/data/results_phase1_appendix.json").read_text(encoding="utf-8"))
    a1 = ap["A1_registered_decisive_cell"]
    a2 = ap["A2_new_spanish_mask"]

    plaus = r1["C1"]["plausible_mean_F"]
    without = a1["flagged_without_mention_meanF"]
    masked = a2["flagged_masked_meanF"]
    total = r1["C1"]["flagged_mean_F"]
    bars = [("plausible\n(floor)", plaus, "#0072B2"),
            ("flagged\nwithout-mention", without, "#E69F00"),
            ("flagged\nmasked (ES)", masked, "#E69F00"),
            ("flagged\ntotal", total, "#E69F00")]
    # registered paired contrasts vs plausible: (label, diff, ci_lo, ci_hi, p)
    contrasts = [
        ("A1", 1, without - plaus, a1["paired_test"]["ci95"][0], a1["paired_test"]["ci95"][1],
         a1["paired_test"]["p_two_sided"]),
        ("A2", 2, masked - a2["plausible_masked_meanF"], a2["paired_test"]["ci95"][0],
         a2["paired_test"]["ci95"][1], a2["paired_test"]["p_two_sided"]),
        ("C1", 3, r1["C1"]["test_result"]["mean_diff"],
         r1["C1"]["test_result"]["mean_diff"] - 1.96 * r1["C1"]["test_result"]["se"],
         r1["C1"]["test_result"]["mean_diff"] + 1.96 * r1["C1"]["test_result"]["se"],
         r1["C1"]["test_result"]["p_two_sided"]),
    ]
    write_csv(HERE / "fig2_data.csv",
              ["bar", "mean_F_loading", "source"],
              [["plausible_floor", plaus, "results_phase1.json@bc4320a"],
               ["flagged_without_mention", without, "results_phase1_appendix.json@f39df1f"],
               ["flagged_masked_ES", masked, "results_phase1_appendix.json@f39df1f"],
               ["flagged_total", total, "results_phase1.json@bc4320a"]])
    with (HERE / "fig2_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "vs_bar_index", "diff", "ci95_lo", "ci95_hi", "p_two_sided"])
        for c in contrasts:
            w.writerow(c)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62,
           edgecolor="white", linewidth=1.0, zorder=3)
    for x, b in zip(xs, bars):
        ax.text(x, b[1] + 0.006, f"{b[1]:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("Set F loading (raw-logit units)")
    ax.set_ylim(0, total * 1.28)
    ax.axhline(plaus, color="#0072B2", lw=0.9, ls=":", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    # significance brackets carrying the registered paired-difference CI + p
    y0 = total * 1.05
    for i, (lab, bar_idx, diff, lo, hi, p) in enumerate(contrasts):
        y = y0 + i * total * 0.06
        ax.plot([0, bar_idx], [y, y], color=MUTED, lw=1.0)
        ax.plot([0, 0], [y - total * 0.012, y], color=MUTED, lw=1.0)
        ax.plot([bar_idx, bar_idx], [y - total * 0.012, y], color=MUTED, lw=1.0)
        ax.text((0 + bar_idx) / 2, y + total * 0.005,
                f"{lab}: Δ={diff:.3f} [{lo:.3f}, {hi:.3f}], p={p:.1e}",
                ha="center", va="bottom", fontsize=7.2, color=INK)
    ax.set_title("Figure 2 — C1 anatomy: Set F loading during diagnosis\n"
                 "(bars = means; brackets = registered paired contrasts vs plausible floor)",
                 fontsize=10.5, loc="left", color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig2_c1_anatomy.{ext}")
    plt.close(fig)
    return len(bars)


# ------------------------------------------------------------------ Fig 3

def fig3():
    r2 = json.loads((WP / "phase2b/data/results_phase2b.json").read_text(encoding="utf-8"))
    arms = ["B0_none", "B1_full", "B3_rand"]
    arm_color = {"B0_none": "#5a5a5a", "B1_full": "#0072B2", "B3_rand": "#E69F00"}
    dat = {}
    rows = []
    for a in arms:
        o = r2["arm_overall"][a]
        n = o["n"]
        for dv, rate in (("diagnosis", o["dv1_diagnosis"]), ("mention", o["dv2_mention"])):
            k = round(rate * n)
            p, lo, hi = wilson(k, n)
            dat[(a, dv)] = (p, lo, hi)
            rows.append([a, dv, k, n, round(p, 4), round(lo, 4), round(hi, 4)])
    write_csv(HERE / "fig3_data.csv",
              ["arm", "dv", "k", "n", "p", "ci95_lo", "ci95_hi"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
    for ax, dv, title in ((axes[0], "diagnosis", "DV1 — P(diagnosis)"),
                          (axes[1], "mention", "DV2 — P(ES mention)")):
        for i, a in enumerate(arms):
            p, lo, hi = dat[(a, dv)]
            ax.bar(i, p, color=arm_color[a], width=0.6, edgecolor="white",
                   linewidth=1.0, zorder=3)
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=INK, lw=1.1,
                        capsize=3, zorder=4)
            ax.text(i, p + 0.02, f"{p:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["B0\nnone", "B1\nfull ablation", "B3\nrandom"], fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=10, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("rate  [Wilson 95% CI]")
    # annotate the registered T2/S2 result on the mention panel
    T2 = next(t for t in r2["confirmatory"] if t["test"] == "T2")
    axes[1].annotate(f"T2 (B1−B0): p={T2['p_permutation_one_sided']:.4f}",
                     xy=(1, dat[("B1_full", "mention")][2] + 0.01), xytext=(1.35, 0.66),
                     fontsize=7.6, color=INK, ha="left",
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    fig.suptitle("Figure 3 — Phase 2b projection ablation: behaviour (DV1) intact, verbal channel (DV2) reduced",
                 fontsize=10.5, y=1.02, x=0.02, ha="left", color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig3_2b_ablation.{ext}")
    plt.close(fig)
    return len(rows)


def main() -> int:
    beh = Path(sys.argv[1])
    n1 = fig1(beh)
    n2 = fig2()
    n3 = fig3()
    print(f"Fig1: {n1} stratified cells -> fig1_pdiagnosis.pdf/.png + fig1_data.csv")
    print(f"Fig2: {n2} bars -> fig2_c1_anatomy.pdf/.png + fig2_data.csv/fig2_contrasts.csv")
    print(f"Fig3: {n3} arm×dv cells -> fig3_2b_ablation.pdf/.png + fig3_data.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
