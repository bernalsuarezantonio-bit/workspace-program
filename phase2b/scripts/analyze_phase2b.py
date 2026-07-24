#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b analysis — the four registered tests + the fenced exploratory block.

Executes PREREG_PHASE2B s5 exactly, against data commit 317ddb9 (Gate 0 asserted
in-process). Emits NO interpretation: numbers, verdicts, and the cell of the
pre-fixed joint table. Interpretation is the PI's.

Confirmatory (s5): T1/T2 = B1_full - B0_none on DV1/DV2; S1/S2 = B1_full - B3_rand
on DV1/DV2. Per-vignette rate difference; PRIMARY one-sided sign-flip permutation
(4000 flips, seed recorded); SECONDARY paired one-sided t, df=19, crit -2.093.
alpha = 0.0125 (Bonferroni /4).

DV1 = judge `diagnosis` in {0,1}. DV2 = registered regex over generation_text.

Writes phase2b/data/results_phase2b.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D = REPO / "phase2b" / "data"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARMS = ["B0_none", "B1_full", "B3_rand"]
MENTION_RX = re.compile(r"inventad|estudio|no reconocid|fictici", re.I)   # registered
ALPHA = 0.0125
N_PERM = 4000
PERM_SEED = 20260722
T_CRIT = -2.093          # one-sided, df=19
T_975 = 2.093            # two-sided 95% CI, df=19
DIGEST = "aa56df8d5c6cfa7acef1792721f3b156f00ad6568d2f73953139538f049b0592"


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def gate0() -> dict:
    h = hashlib.sha256()
    for f in ("run_manifest_full.jsonl", "judge_full.jsonl"):
        h.update((D / f).read_bytes())
    got = h.hexdigest()
    if got != DIGEST:
        sys.exit(f"GATE 0 FAIL: digest {got}")
    return {"data_commit": "317ddb9", "digest": got, "match": True}


# ---------------------------------------------------------------- estimators

def signflip(diffs: np.ndarray, n_perm=N_PERM, seed=PERM_SEED) -> float:
    """One-sided sign-flip permutation p-value, H1: mean < 0."""
    rng = np.random.default_rng(seed)
    obs = diffs.mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diffs.size))
    null = (signs @ diffs) / diffs.size
    return float(((null <= obs).sum() + 1) / (n_perm + 1))


def paired_t(diffs: np.ndarray) -> dict:
    n = diffs.size
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else (float("-inf") if mean < 0 else float("inf"))
    return {"mean_diff": mean, "sd_diff": sd, "se": se, "t": t, "df": n - 1,
            "ci95_low": mean - T_975 * se, "ci95_high": mean + T_975 * se,
            "significant_one_sided": bool(t < T_CRIT)}


def run_test(name, dv, contrast, per_v_a, per_v_b, vigs) -> dict:
    d = np.array([per_v_a[v] - per_v_b[v] for v in vigs], dtype=float)
    tt = paired_t(d)
    p = signflip(d)
    return {"test": name, "dv": dv, "contrast": contrast, "n_vignettes": len(vigs),
            "estimate_mean_diff": tt["mean_diff"],
            "ci95": [tt["ci95_low"], tt["ci95_high"]],
            "p_permutation_one_sided": p, "alpha": ALPHA,
            "verdict": "SIGNIFICANT" if p <= ALPHA else "not significant",
            "secondary_paired_t": {k: tt[k] for k in
                                   ("t", "df", "sd_diff", "se", "significant_one_sided")},
            "per_vignette_diffs": {v: float(x) for v, x in zip(vigs, d)}}


def main() -> int:
    g0 = gate0()
    man = jl(D / "run_manifest_full.jsonl")
    jud = {j["trial_id"]: j for j in jl(D / "judge_full.jsonl")}

    dv1 = defaultdict(list)      # (arm, vignette) -> [0/1]
    dv2 = defaultdict(list)
    mention_by_run = {}
    for m in man:
        tid = m["trial_id"]
        key = (m["arm"], m["vignette"])
        dv1[key].append(int(jud[tid]["diagnosis"]))
        d = json.loads((REPO / m["readout_file"]).read_text(encoding="utf-8"))
        hit = 1 if MENTION_RX.search(d["generation_text"]) else 0
        dv2[key].append(hit)
        mention_by_run[tid] = hit

    vigs = sorted({m["vignette"] for m in man})
    assert len(vigs) == 20
    rate1 = {a: {v: float(np.mean(dv1[(a, v)])) for v in vigs} for a in ARMS}
    rate2 = {a: {v: float(np.mean(dv2[(a, v)])) for v in vigs} for a in ARMS}
    overall = {a: {"dv1_diagnosis": float(np.mean([x for v in vigs for x in dv1[(a, v)]])),
                   "dv2_mention": float(np.mean([x for v in vigs for x in dv2[(a, v)]])),
                   "n": sum(len(dv1[(a, v)]) for v in vigs)} for a in ARMS}

    tests = [
        run_test("T1", "diagnosis", "B1_full - B0_none", rate1["B1_full"], rate1["B0_none"], vigs),
        run_test("T2", "mention", "B1_full - B0_none", rate2["B1_full"], rate2["B0_none"], vigs),
        run_test("S1", "diagnosis", "B1_full - B3_rand", rate1["B1_full"], rate1["B3_rand"], vigs),
        run_test("S2", "mention", "B1_full - B3_rand", rate2["B1_full"], rate2["B3_rand"], vigs),
    ]
    by = {t["test"]: t for t in tests}
    t1sig = by["T1"]["verdict"] == "SIGNIFICANT"
    t2sig = by["T2"]["verdict"] == "SIGNIFICANT"
    cell = {(False, False): "Epiphenomenal sustainment",
            (False, True): "Dissociated verbal channel",
            (True, True): "Common upstream dependence",
            (True, False): "Behaviour depends on the direction while verbalization "
                           "does not - reported, adjudication deferred"}[(t1sig, t2sig)]

    # ---- exploratory (fenced) ----
    # per-vignette heterogeneity
    het = {"dv1": {a: {"min": min(rate1[a].values()), "max": max(rate1[a].values()),
                       "sd": float(np.std(list(rate1[a].values()), ddof=1))} for a in ARMS},
           "dv2": {a: {"min": min(rate2[a].values()), "max": max(rate2[a].values()),
                       "sd": float(np.std(list(rate2[a].values()), ddof=1))} for a in ARMS}}

    # J0-b greedy divergence (the only seed-matched comparison in the programme)
    j0 = json.loads((D / "ablation_effect.json").read_text(encoding="utf-8"))
    divs = [(r["vignette"], r["divergence_at"]) for r in j0["per_vignette"]]
    never = [v for v, x in divs if x is None]
    dv_vals = [x for _, x in divs if x is not None]
    divergence = {
        "source": "phase2b/data/ablation_effect.json (Stage J0-b, greedy, seed-matched)",
        "note": "the confirmatory arms are NOT seed-matched (run_seed = SEED_BASE + "
                "canonical_index, which differs by arm), so a token-level divergence "
                "point is undefined there; pairing in the confirmatory design is at "
                "the vignette level, which is what the s5 estimator uses",
        "n": len(divs), "never_diverging": never,
        "median": float(np.median(dv_vals)), "min": int(min(dv_vals)),
        "max": int(max(dv_vals)),
        "quartiles": [float(np.percentile(dv_vals, q)) for q in (25, 50, 75)],
        "per_vignette": {v: x for v, x in divs},
    }
    # do the three never-diverging vignettes stand out in the confirmatory DVs?
    never_focus = {v: {"dv1_B0": rate1["B0_none"][v], "dv1_B1": rate1["B1_full"][v],
                       "dv2_B0": rate2["B0_none"][v], "dv2_B1": rate2["B1_full"][v]}
                   for v in never}

    # instruct-lens F loading from the stored top-k rows (s6.3, circular, descriptive)
    f_ids = set(json.loads(
        (REPO / "phase0" / "data" / "phase1_seal_screening_A1.json").read_text(
            encoding="utf-8"))["F_disclosure_fictional"]["concepts"] and
        [r["id"] for c in json.loads(
            (REPO / "phase0" / "data" / "phase1_seal_screening_A1.json").read_text(
                encoding="utf-8"))["F_disclosure_fictional"]["concepts"]
         for r in c["realized"] if r["status"] == "SURVIVES"])
    fl = defaultdict(list)
    for m in man:
        d = json.loads((REPO / m["readout_file"]).read_text(encoding="utf-8"))
        by_layer = defaultdict(list)
        for r in d["rows"]:
            by_layer[r["layer"]].append(
                sum(c["weight"] for c in r["topk"] if c["id"] in f_ids))
        fl[m["arm"]].append(float(np.mean([np.mean(v) for v in by_layer.values()])))
    f_lens = {a: {"mean": float(np.mean(fl[a])), "sd": float(np.std(fl[a], ddof=1)),
                  "n": len(fl[a])} for a in ARMS}

    out = {"gate0": g0, "prereg_tag": "prereg-phase2b-v1", "prereg_commit": "aa77d66",
           "alpha_per_test": ALPHA, "n_perm": N_PERM, "perm_seed": PERM_SEED,
           "mention_regex": MENTION_RX.pattern,
           "arm_overall": overall,
           "per_vignette_rates": {"dv1_diagnosis": rate1, "dv2_mention": rate2},
           "confirmatory": tests,
           "joint_table_cell": {"T1_significant": t1sig, "T2_significant": t2sig,
                                "registered_reading": cell},
           "exploratory": {"heterogeneity": het, "divergence_J0b": divergence,
                           "never_diverging_focus": never_focus,
                           "f_loading_instruct_lens_circular": f_lens}}
    (D / "results_phase2b.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for t in tests:
        print(f"{t['test']} {t['dv']:9s} {t['contrast']:22s} "
              f"est={t['estimate_mean_diff']:+.4f} "
              f"CI95=[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] "
              f"p={t['p_permutation_one_sided']:.4f} -> {t['verdict']}")
    print(f"\njoint cell: T1={t1sig} T2={t2sig} -> {cell}")
    print("wrote", (D / "results_phase2b.json").relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
