# Copyright 2026 — Phase 1 delegate. Analysis appendix (post-data, cold).
"""Two numbers-only analyses appended (dated) to RESULTS_PHASE1.md, against the
same committed data (commit 8046a12 / data_digest dc522361). Nothing else in the
artifact is altered — this script only APPENDS a delimited dated appendix and
writes phase1/data/results_phase1_appendix.json.

(A1) REGISTERED decisive-cell test (spec b62accd / POSTHOC sub-2 decisive cell):
     flagged-WITHOUT-mention (registered regex) vs plausible, Set F loading,
     SAME estimator as C1 (paired-by-vignette, two-sided). Estimate, CI95, p.

(A2) NEW post-hoc (clearly labelled; motivation: the registered emission mask was
     inert due to ES-generation / EN-operative mismatch, lesson #5). Mention spans
     defined by SPANISH surface forms in the generated text
     (inventad|estudio|ficticio|no reconocid); those generation positions (±2) are
     masked and Set F EN-concept loading is recomputed in flagged — the mask that
     can bite (emission in ES, concept in EN). Reports flagged-masked vs plausible.

Reuses the frozen §2 aggregation. No scipy. Numbers only; no interpretation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "phase1" / "data"
BAND = list(range(17, 27))
BANDSET = set(BAND)
DATA_COMMIT = "8046a12"
DATA_DIGEST = "dc522361096bae30377ecf05d37142cfcb3f52fbb6349c77825bea455f0fb8f1"
APPENDIX_MARK = "## Appendix A —"

# Registered mention regex (POSTHOC sub-1/2, unchanged) — used to DEFINE strata in A1.
REG_RE = re.compile(r"inventad|estudio|no reconocid|fictici", re.IGNORECASE)
# NEW Spanish-surface span regex for the A2 mask (per this task's spec).
ES_SPAN_RE = re.compile(r"inventad|estudio|ficticio|no reconocid", re.IGNORECASE)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


SCREEN = json.loads((REPO / "phase0" / "data" / "phase1_seal_screening_A1.json").read_text(encoding="utf-8"))


def operative(setname):
    out = {}
    for c in SCREEN[setname]["concepts"]:
        for r in c["realized"]:
            if r["status"] == "SURVIVES":
                out[r["folded"]] = r["langs"]
    return out


opF = operative("F_disclosure_fictional")
TOK = Tokenizer.from_file(str(
    REPO / "phase0" / "data" / "hf_cache" / "models--Qwen--Qwen2.5-7B-Instruct"
    / "snapshots" / "a09a35458c702b33eeacc393d103063234e8bc28" / "tokenizer.json"))


# ---- Student-t (no scipy): incomplete beta + bisection inverse ----
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 400, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (FPMIN if abs(d) < FPMIN else d)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (FPMIN if abs(d) < FPMIN else d)
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (FPMIN if abs(d) < FPMIN else d)
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def p_two_sided(t, df):
    if t == 0:
        return 1.0
    return betai(df / 2.0, 0.5, df / (df + t * t))


def t_crit_two(alpha, df):
    """t>0 with P(|T_df|>t)=alpha (e.g. alpha=0.05 -> 95% two-sided)."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p_two_sided(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    df = n - 1
    tc = t_crit_two(0.05, df)
    return {"n": n, "df": df, "mean_diff": mean, "sd_diff": sd, "se": se, "t": t,
            "p_two_sided": p_two_sided(t, df),
            "ci95": [mean - tc * se, mean + tc * se]}


def per_run(readout_file):
    d = json.loads((REPO / readout_file).read_text(encoding="utf-8"))
    rows = d["rows"]
    fpos = {}                      # EN-concept F weight per generation position
    genpos = set()
    for r in rows:
        if r["kind"] != "lens" or r["segment"] != "generation" or r["layer"] not in BANDSET:
            continue
        p = r["position"]
        genpos.add(p)
        for e in r["topk"]:
            lf = opF.get(fold(e["token"]))
            if lf is not None and "EN" in lf:
                fpos[p] = fpos.get(p, 0.0) + e["weight"]
    gp = sorted(genpos)
    ncells = len(BAND) * max(1, len(gp))
    F_en = sum(fpos.values()) / ncells
    text = d["generation_text"]
    mention_reg = bool(REG_RE.search(text))

    # A2 Spanish-surface mask via char-offset -> token alignment
    enc = TOK.encode(text, add_special_tokens=False)
    aligned = (len(enc.ids) == len(gp)) and bool(gp)
    F_masked = None
    n_masked = None
    if aligned:
        base = gp[0]
        offs = enc.offsets
        mask_tok = set()
        for m in ES_SPAN_RE.finditer(text):
            a, b = m.span()
            for i, (s, e) in enumerate(offs):
                if e > s and s < b and e > a:      # nonempty overlap
                    mask_tok.add(i)
        mask_pos = set()
        for i in mask_tok:
            for dp in range(base + i - 2, base + i + 3):
                mask_pos.add(dp)
        unmasked = [p for p in gp if p not in mask_pos]
        n_masked = len(gp) - len(unmasked)
        if unmasked:
            F_masked = sum(fpos.get(p, 0.0) for p in unmasked) / (len(BAND) * len(unmasked))
    return {"F": F_en, "mention": mention_reg, "aligned": aligned,
            "F_masked": F_masked, "n_masked": n_masked, "n_genpos": len(gp)}


def main():
    t0 = time.time()
    # Gate 0 re-assert (digest) — same data.
    dig = hashlib.sha256(
        (DATA / "run_manifest_full.jsonl").read_bytes()
        + (DATA / "judge_full.jsonl").read_bytes()
        + (DATA / "completeness_report.json").read_bytes()).hexdigest()
    assert dig == DATA_DIGEST, "GATE 0 FAIL: digest mismatch"

    manifest = [json.loads(l) for l in (DATA / "run_manifest_full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    judge = [json.loads(l) for l in (DATA / "judge_full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    jmap = {j["trial_id"]: j for j in judge}

    FL, PL = "C1_DN_flagged_L1", "C1_DN_plausible_L1"
    runs = []
    for m in [m for m in manifest if m["cell"] in (FL, PL)]:
        j = jmap.get(m["trial_id"], {})
        dx = j.get("diagnosis") if "judge_error" not in j else None
        r = per_run(m["readout_file"])
        r.update({"cell": m["cell"], "vignette": m["vignette"], "diagnosis": dx})
        runs.append(r)
    print(f"processed {len(runs)} C1 runs in {(time.time()-t0)/60:.1f}m", flush=True)

    vigs = sorted({r["vignette"] for r in runs})
    fl_dx1 = [r for r in runs if r["cell"] == FL and r["diagnosis"] == 1]
    pl_dx1 = [r for r in runs if r["cell"] == PL and r["diagnosis"] == 1]
    fl_wo = [r for r in fl_dx1 if not r["mention"]]

    def vmean(rs, key, filt=lambda r: True):
        out = {}
        for v in vigs:
            xs = [r[key] for r in rs if r["vignette"] == v and filt(r) and r[key] is not None]
            out[v] = float(np.mean(xs)) if xs else None
        return out

    # ---- A1: flagged-without-mention vs plausible, Set F, paired-by-vignette ----
    wo_v = vmean(fl_wo, "F")
    pl_v = vmean(pl_dx1, "F")
    diffs = [wo_v[v] - pl_v[v] for v in vigs if wo_v[v] is not None and pl_v[v] is not None]
    a1_test = paired(diffs)
    a1 = {
        "n_flagged_without_mention": len(fl_wo),
        "n_plausible": len(pl_dx1),
        "flagged_without_mention_meanF": float(np.mean([r["F"] for r in fl_wo])),
        "plausible_meanF": float(np.mean([r["F"] for r in pl_dx1])),
        "n_vignettes_paired": a1_test["n"],
        "paired_test": a1_test,
    }

    # ---- A2: Spanish-surface mask, flagged-masked vs plausible(-masked) ----
    fl_al = [r for r in fl_dx1 if r["aligned"]]
    pl_al = [r for r in pl_dx1 if r["aligned"]]
    flm_v = vmean(fl_al, "F_masked")
    plm_v = vmean(pl_al, "F_masked")
    mdiffs = [flm_v[v] - plm_v[v] for v in vigs if flm_v[v] is not None and plm_v[v] is not None]
    a2_test = paired(mdiffs)
    fl_bite = float(np.mean([r["n_masked"] / r["n_genpos"] for r in fl_al if r["n_masked"] is not None and r["n_genpos"]]))
    pl_bite = float(np.mean([r["n_masked"] / r["n_genpos"] for r in pl_al if r["n_masked"] is not None and r["n_genpos"]]))
    a2 = {
        "n_flagged_aligned": len(fl_al),
        "n_plausible_aligned": len(pl_al),
        "flagged_mean_masked_fraction": fl_bite,
        "plausible_mean_masked_fraction": pl_bite,
        "flagged_masked_meanF": float(np.mean([r["F_masked"] for r in fl_al if r["F_masked"] is not None])),
        "plausible_masked_meanF": float(np.mean([r["F_masked"] for r in pl_al if r["F_masked"] is not None])),
        "flagged_unmasked_meanF": float(np.mean([r["F"] for r in fl_al])),
        "plausible_unmasked_meanF": float(np.mean([r["F"] for r in pl_al])),
        "n_vignettes_paired": a2_test["n"],
        "paired_test": a2_test,
    }

    results = {"data_commit": DATA_COMMIT, "data_digest": dig,
               "A1_registered_decisive_cell": a1, "A2_new_spanish_mask": a2,
               "elapsed_min": (time.time() - t0) / 60.0}
    (DATA / "results_phase1_appendix.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("WROTE", (DATA / "results_phase1_appendix.json").relative_to(REPO))
    append_report(a1, a2, dig)
    print("APPENDED to RESULTS_PHASE1.md")
    # echo
    print(json.dumps({"A1": a1, "A2": a2}, indent=2))
    return 0


def f(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def append_report(a1, a2, dig):
    rp = REPO / "RESULTS_PHASE1.md"
    cur = rp.read_text(encoding="utf-8")
    if APPENDIX_MARK in cur:
        print("appendix already present; not duplicating")
        return
    t1 = a1["paired_test"]
    t2 = a2["paired_test"]
    L = []
    L.append("\n\n---\n")
    L.append(f"## Appendix A — registered decisive-cell test + new language-aware mask (2026-07-22)\n")
    L.append(f"*Appended by `phase1/scripts/analyze_phase1_appendix.py` against data commit "
             f"`{DATA_COMMIT}` (digest `{dig[:16]}…`, Gate 0 re-asserted). Nothing above this line is "
             f"altered. Numbers only.*\n")

    L.append("\n### A1 — REGISTERED decisive cell (spec `b62accd`, POSTHOC sub-2): "
             "flagged-WITHOUT-mention vs plausible, Set F\n")
    L.append(f"Same estimator as C1 (paired-by-vignette, two-sided). Strata by the registered regex "
             f"`inventad|estudio|no reconocid|fictici`.\n")
    L.append("| group | n runs | mean F |")
    L.append("|---|---|---|")
    L.append(f"| flagged — WITHOUT mention | {a1['n_flagged_without_mention']} | {f(a1['flagged_without_mention_meanF'])} |")
    L.append(f"| plausible (diagnosis=1) | {a1['n_plausible']} | {f(a1['plausible_meanF'])} |")
    L.append(f"\nPaired over **{t1['n']}** vignettes: mean diff (flagged-without−plausible) = "
             f"**{f(t1['mean_diff'])}**, 95% CI **[{f(t1['ci95'][0])}, {f(t1['ci95'][1])}]**, "
             f"t({t1['df']}) = {t1['t']:.3f}, two-sided **p = {t1['p_two_sided']:.4g}**.\n")

    L.append("\n### A2 — NEW post-hoc (not preregistered): Spanish-surface emission mask, Set F EN-concept\n")
    L.append(f"Motivation: the registered positional mask (POSTHOC sub-3) was inert (~0.24% of positions) "
             f"because generation is Spanish while the operative tokens are English (lesson #5). This mask "
             f"instead marks generated-text spans matching SPANISH surface forms "
             f"`inventad|estudio|ficticio|no reconocid`, masks those generation positions (±2 tokens, via "
             f"char-offset→token alignment), and recomputes Set F EN-concept loading. Aligned runs only.\n")
    L.append("| cell | n aligned | mean masked frac | masked mean F | (unmasked mean F) |")
    L.append("|---|---|---|---|---|")
    L.append(f"| flagged×diag=1 | {a2['n_flagged_aligned']} | {f(a2['flagged_mean_masked_fraction'])} | "
             f"{f(a2['flagged_masked_meanF'])} | {f(a2['flagged_unmasked_meanF'])} |")
    L.append(f"| plausible×diag=1 | {a2['n_plausible_aligned']} | {f(a2['plausible_mean_masked_fraction'])} | "
             f"{f(a2['plausible_masked_meanF'])} | {f(a2['plausible_unmasked_meanF'])} |")
    L.append(f"\nPaired over **{t2['n']}** vignettes (masked F, flagged−plausible): mean diff = "
             f"**{f(t2['mean_diff'])}**, 95% CI **[{f(t2['ci95'][0])}, {f(t2['ci95'][1])}]**, "
             f"t({t2['df']}) = {t2['t']:.3f}, two-sided **p = {t2['p_two_sided']:.4g}**.\n")
    L.append("\n*Machine-readable: `phase1/data/results_phase1_appendix.json`. Interpretation is the PI's "
             "(R5 asymmetric-informativeness clause).*\n")

    rp.write_text(cur + "\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
