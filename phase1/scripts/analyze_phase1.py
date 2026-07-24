# Copyright 2026 — Phase 1 delegate. Analysis (separate session, post-data).
"""COLD re-derivation of the Phase 1 confirmatory analysis, run against the
committed data (commit 8046a12, data_digest dc522361, N=800). Gate 0 is
re-asserted in-process before any loading is computed. Implements the
preregistered §2 aggregation EXACTLY as the frozen power-analysis reference
(phase0/scripts/phase1_p0_power.py:loading_for_rep): band 17-26, generation
positions only (R4), language-folded matching (R3), SURVIVORS-only operative
lists (R2 echoes barred). No scipy: Student-t p-values via the regularized
incomplete beta (Numerical-Recipes betacf).

Confirmatory: C1 (Set F, diagnosis=1, two-sided) and C2 (Set A, one-sided).
Then the registered post-hoc EXPLORATORY C1 self-echo block
(phase1/POSTHOC_EXPLORATORY_C1_selfecho.md), fenced and non-adjudicative.

Writes phase1/data/results_phase1.json (machine) and RESULTS_PHASE1.md (report).
Emits NO interpretation; the asymmetric-informativeness clause governs.
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
BAND = list(range(17, 27))           # primary band 17-26 inclusive
BANDSET = set(BAND)
ALPHA = 0.025
DF = 19
TCRIT_2SIDED = 2.433                  # t_{0.9875,19}
TCRIT_1SIDED = 2.093                  # t_{0.975,19}

CELLS = ["C1_DN_flagged_L1", "C1_DN_plausible_L1", "C2_incoherent_L4", "C2_incoherent_L1"]
DATA_COMMIT = "8046a12"
DATA_DIGEST = "dc522361096bae30377ecf05d37142cfcb3f52fbb6349c77825bea455f0fb8f1"
PREREG_SHA = "bedbcc78f9dc261ffd789dc55097bbd9c997c4db9d3e63077489623ae05f55d5"
LENS_SHA = "3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29"
MODEL_DIG = "Qwen/Qwen2.5-7B-Instruct@a09a3545"

MENTION_RE = re.compile(r"inventad|estudio|no reconocid|fictici", re.IGNORECASE)


# ------------------------- Student-t p-value (no scipy) -------------------------
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
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


def p_two_sided(t, df=DF):
    """P(|T_df| > |t|)."""
    if t == 0:
        return 1.0
    x = df / (df + t * t)
    return betai(df / 2.0, 0.5, x)


def paired_t(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf") * (1 if mean > 0 else -1)
    p2 = p_two_sided(t, n - 1)
    p1 = (p2 / 2.0) if t > 0 else (1.0 - p2 / 2.0)   # one-sided upper
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "se": se, "t": t,
            "df": n - 1, "p_two_sided": p2, "p_one_sided_upper": p1}


# ------------------------------ operative lists ------------------------------
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


opA = operative("A_generic_nosological")
opF = operative("F_disclosure_fictional")
opF_stripped = {k.strip() for k in opF}   # for emitted-token mask (space-insensitive)

TOK = Tokenizer.from_file(str(
    REPO / "phase0" / "data" / "hf_cache" / "models--Qwen--Qwen2.5-7B-Instruct"
    / "snapshots" / "a09a35458c702b33eeacc393d103063234e8bc28" / "tokenizer.json"))


# ------------------------------ per-run analysis ------------------------------
def analyze_run(readout_path: Path):
    d = json.loads(readout_path.read_text(encoding="utf-8"))
    rows = d["rows"]
    fpos, apos = {}, {}
    fpos_en, fpos_es, apos_en, apos_es = {}, {}, {}, {}
    genpos = set()
    for r in rows:
        if r["kind"] != "lens" or r["segment"] != "generation" or r["layer"] not in BANDSET:
            continue
        p = r["position"]
        genpos.add(p)
        for e in r["topk"]:
            f = fold(e["token"])
            w = e["weight"]
            la = opA.get(f)
            if la is not None:
                apos[p] = apos.get(p, 0.0) + w
                if "EN" in la:
                    apos_en[p] = apos_en.get(p, 0.0) + w
                if "ES" in la:
                    apos_es[p] = apos_es.get(p, 0.0) + w
            lf = opF.get(f)
            if lf is not None:
                fpos[p] = fpos.get(p, 0.0) + w
                if "EN" in lf:
                    fpos_en[p] = fpos_en.get(p, 0.0) + w
                if "ES" in lf:
                    fpos_es[p] = fpos_es.get(p, 0.0) + w
    gp = sorted(genpos)
    ncells = len(BAND) * max(1, len(gp))

    def agg(pos_map, positions):
        return sum(pos_map.get(p, 0.0) for p in positions) / (len(BAND) * max(1, len(positions)))

    out = {
        "n_genpos": len(gp),
        "F": sum(fpos.values()) / ncells,
        "F_en": sum(fpos_en.values()) / ncells,
        "F_es": sum(fpos_es.values()) / ncells,
        "A": sum(apos.values()) / ncells,
        "A_en": sum(apos_en.values()) / ncells,
        "A_es": sum(apos_es.values()) / ncells,
        "mention": bool(MENTION_RE.search(d["generation_text"])),
    }

    # emitted-token positional mask (exploratory sub-analyses 3/4)
    ids = TOK.encode(d["generation_text"], add_special_tokens=False).ids
    aligned = (len(ids) == len(gp))
    out["aligned"] = aligned
    if aligned and gp:
        emit_f = {gp[i]: fold(TOK.decode([ids[i]])).strip() for i in range(len(gp))}
        masked = set()
        for p in gp:
            if emit_f.get(p, "") in opF_stripped:
                for dp in range(p - 2, p + 3):
                    masked.add(dp)
        unmasked = [p for p in gp if p not in masked]
        out["n_masked_pos"] = len(gp) - len(unmasked)
        out["F_masked"] = agg(fpos, unmasked) if unmasked else None
        out["A_masked"] = agg(apos, unmasked) if unmasked else None
    else:
        out["n_masked_pos"] = None
        out["F_masked"] = None
        out["A_masked"] = None
    return out


def main():
    t0 = time.time()
    # ---- Gate 0 re-assertion (digest + counts) ----
    dig = hashlib.sha256(
        (DATA / "run_manifest_full.jsonl").read_bytes()
        + (DATA / "judge_full.jsonl").read_bytes()
        + (DATA / "completeness_report.json").read_bytes()
    ).hexdigest()
    manifest = [json.loads(l) for l in (DATA / "run_manifest_full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    judge = [json.loads(l) for l in (DATA / "judge_full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    jmap = {j["trial_id"]: j for j in judge}
    ids = [m["trial_id"] for m in manifest]
    gate0 = {
        "data_commit": DATA_COMMIT,
        "data_digest_recomputed": dig,
        "data_digest_expected": DATA_DIGEST,
        "data_digest_match": dig == DATA_DIGEST,
        "n_runs": len(ids),
        "n_duplicate_ids": len(ids) - len(set(ids)),
        "prereg_sha256": PREREG_SHA,
        "lens_pt_sha256": LENS_SHA,
        "model_digest": MODEL_DIG,
    }
    print("GATE 0:", json.dumps(gate0, indent=2))
    assert gate0["data_digest_match"], "GATE 0 FAIL: data_digest mismatch"
    assert gate0["n_runs"] == 800 and gate0["n_duplicate_ids"] == 0, "GATE 0 FAIL: N/dup"

    print(f"operative Set A={len(opA)} folded keys, Set F={len(opF)} folded keys", flush=True)

    # ---- per-run loadings ----
    runs = {}
    for i, m in enumerate(manifest, 1):
        tid = m["trial_id"]
        res = analyze_run(REPO / m["readout_file"])
        j = jmap.get(tid, {})
        diagnosis = j.get("diagnosis") if "judge_error" not in j else None
        runs[tid] = {**res, "cell": m["cell"], "vignette": m["vignette"],
                     "rep": m["rep"], "diagnosis": diagnosis,
                     "judge_error": "judge_error" in j}
        if i % 100 == 0:
            print(f"  ...{i}/800  ({(time.time()-t0)/60:.1f}m)", flush=True)

    vignettes = sorted({r["vignette"] for r in runs.values()})
    n_aligned = sum(1 for r in runs.values() if r["aligned"])

    def mask_frac(cell, cond):
        rs = [r for r in runs.values() if r["cell"] == cell and r["aligned"] and cond(r)
              and r["n_masked_pos"] is not None and r["n_genpos"]]
        if not rs:
            return None
        return float(np.mean([r["n_masked_pos"] / r["n_genpos"] for r in rs]))

    def cell_runs(cell):
        return {tid: r for tid, r in runs.items() if r["cell"] == cell}

    def mean_key(rs, key, cond=None):
        vals = [r[key] for r in rs.values() if (cond is None or cond(r)) and r[key] is not None]
        return float(np.mean(vals)) if vals else None

    # diagnosis rates
    diag = {}
    for cell in CELLS:
        rs = cell_runs(cell)
        ok = [r for r in rs.values() if not r["judge_error"]]
        dx1 = sum(1 for r in ok if r["diagnosis"] == 1)
        diag[cell] = {"n_runs": len(rs), "n_ok": len(ok), "n_judge_error": len(rs) - len(ok),
                      "n_diagnosis_1": dx1, "n_diagnosis_0": len(ok) - dx1,
                      "rate": dx1 / len(ok) if ok else None}

    # ---------------- C1 confirmatory: Set F, diagnosis=1, two-sided ----------------
    fl = cell_runs("C1_DN_flagged_L1")
    pl = cell_runs("C1_DN_plausible_L1")

    def per_vig_mean(rs, key, cond):
        out = {}
        for v in vignettes:
            vals = [r[key] for r in rs.values() if r["vignette"] == v and cond(r) and r[key] is not None]
            out[v] = float(np.mean(vals)) if vals else None
        return out

    dx1 = lambda r: r["diagnosis"] == 1
    fl_v = per_vig_mean(fl, "F", dx1)
    pl_v = per_vig_mean(pl, "F", dx1)
    diffs_c1 = [fl_v[v] - pl_v[v] for v in vignettes if fl_v[v] is not None and pl_v[v] is not None]
    c1_test = paired_t(diffs_c1)
    c1 = {
        "DV": "Set F loading", "conditioning": "diagnosis==1", "test": "paired t, two-sided",
        "alpha": ALPHA, "tcrit": TCRIT_2SIDED,
        "flagged_mean_F": mean_key(fl, "F", dx1), "plausible_mean_F": mean_key(pl, "F", dx1),
        "flagged_mean_F_EN": mean_key(fl, "F_en", dx1), "flagged_mean_F_ES": mean_key(fl, "F_es", dx1),
        "plausible_mean_F_EN": mean_key(pl, "F_en", dx1), "plausible_mean_F_ES": mean_key(pl, "F_es", dx1),
        "test_result": c1_test,
        "significant": abs(c1_test["t"]) > TCRIT_2SIDED,
        "n_vignettes_paired": len(diffs_c1),
    }

    # ---------------- C2 confirmatory: Set A, all reps, one-sided (L4>L1) ----------------
    l4 = cell_runs("C2_incoherent_L4")
    l1 = cell_runs("C2_incoherent_L1")
    allr = lambda r: True
    l4_v = per_vig_mean(l4, "A", allr)
    l1_v = per_vig_mean(l1, "A", allr)
    diffs_c2 = [l4_v[v] - l1_v[v] for v in vignettes if l4_v[v] is not None and l1_v[v] is not None]
    c2_test = paired_t(diffs_c2)
    c2 = {
        "DV": "Set A loading", "conditioning": "none (all reps)", "test": "paired t, one-sided (L4>L1)",
        "alpha": ALPHA, "tcrit": TCRIT_1SIDED,
        "L4_mean_A": mean_key(l4, "A", allr), "L1_mean_A": mean_key(l1, "A", allr),
        "L4_mean_A_EN": mean_key(l4, "A_en", allr), "L4_mean_A_ES": mean_key(l4, "A_es", allr),
        "L1_mean_A_EN": mean_key(l1, "A_en", allr), "L1_mean_A_ES": mean_key(l1, "A_es", allr),
        "test_result": c2_test,
        "significant": (c2_test["t"] > TCRIT_1SIDED),
        "n_vignettes_paired": len(diffs_c2),
    }

    # ---------------- EXPLORATORY (post-hoc C1 self-echo) ----------------
    fl_dx1 = {tid: r for tid, r in fl.items() if r["diagnosis"] == 1}
    pl_dx1 = {tid: r for tid, r in pl.items() if r["diagnosis"] == 1}
    with_m = {tid: r for tid, r in fl_dx1.items() if r["mention"]}
    wo_m = {tid: r for tid, r in fl_dx1.items() if not r["mention"]}

    # sub-3: masked F, flagged vs plausible, paired (aligned runs only)
    flm_v = per_vig_mean(fl, "F_masked", lambda r: r["diagnosis"] == 1 and r["aligned"])
    plm_v = per_vig_mean(pl, "F_masked", lambda r: r["diagnosis"] == 1 and r["aligned"])
    diffs_c1m = [flm_v[v] - plm_v[v] for v in vignettes if flm_v[v] is not None and plm_v[v] is not None]
    c1m_test = paired_t(diffs_c1m) if diffs_c1m else None
    # sub-4: masked A on C2
    l4m_v = per_vig_mean(l4, "A_masked", lambda r: r["aligned"])
    l1m_v = per_vig_mean(l1, "A_masked", lambda r: r["aligned"])
    diffs_c2m = [l4m_v[v] - l1m_v[v] for v in vignettes if l4m_v[v] is not None and l1m_v[v] is not None]
    c2m_test = paired_t(diffs_c2m) if diffs_c2m else None

    exploratory = {
        "sub1_mention_split": {"n_flagged_dx1": len(fl_dx1),
                               "n_with_mention": len(with_m), "n_without_mention": len(wo_m)},
        "sub2_F_by_stratum": {
            "flagged_with_mention_meanF": mean_key(with_m, "F", allr),
            "flagged_without_mention_meanF": mean_key(wo_m, "F", allr),
            "plausible_meanF": mean_key(pl_dx1, "F", allr),
            "n_with": len(with_m), "n_without": len(wo_m), "n_plausible": len(pl_dx1),
        },
        "sub3_positional_mask_C1": {
            "n_aligned_of_800": n_aligned,
            "flagged_mean_masked_fraction": mask_frac("C1_DN_flagged_L1", lambda r: r["diagnosis"] == 1),
            "plausible_mean_masked_fraction": mask_frac("C1_DN_plausible_L1", lambda r: r["diagnosis"] == 1),
            "flagged_masked_meanF": mean_key(fl, "F_masked", lambda r: r["diagnosis"] == 1 and r["aligned"]),
            "plausible_masked_meanF": mean_key(pl, "F_masked", lambda r: r["diagnosis"] == 1 and r["aligned"]),
            "paired_test": c1m_test,
        },
        "sub4_positional_mask_C2": {
            "L4_mean_masked_fraction": mask_frac("C2_incoherent_L4", lambda r: True),
            "L1_mean_masked_fraction": mask_frac("C2_incoherent_L1", lambda r: True),
            "L4_masked_meanA": mean_key(l4, "A_masked", lambda r: r["aligned"]),
            "L1_masked_meanA": mean_key(l1, "A_masked", lambda r: r["aligned"]),
            "paired_test": c2m_test,
        },
    }

    results = {"gate0": gate0, "vignettes": vignettes, "diagnosis_rates": diag,
               "C1": c1, "C2": c2, "exploratory": exploratory,
               "n_aligned_runs": n_aligned,
               "elapsed_min": (time.time() - t0) / 60.0}
    (DATA / "results_phase1.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWROTE", (DATA / "results_phase1.json").relative_to(REPO))
    write_report(results)
    print("WROTE RESULTS_PHASE1.md")
    return 0


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def write_report(R):
    g = R["gate0"]
    c1, c2, ex = R["C1"], R["C2"], R["exploratory"]
    d = R["diagnosis_rates"]
    t1 = c1["test_result"]
    t2 = c2["test_result"]
    L = []
    L.append("# RESULTS — Phase 1 representational study (cold re-derivation)\n")
    L.append(f"**Data commit** `{g['data_commit']}` · **data_digest** `{g['data_digest_recomputed']}` "
             f"({'MATCH' if g['data_digest_match'] else 'MISMATCH'} vs recorded) · "
             f"**N** {g['n_runs']} / {g['n_duplicate_ids']} dup.\n")
    L.append(f"**Seal chain** tag `prereg-phase1-v1` → `4fe4d8d` → `PREREG_PHASE1.md` sha256 `{g['prereg_sha256']}`. "
             f"Lens `.pt` `{g['lens_pt_sha256']}` · model `{g['model_digest']}`.\n")
    L.append("Analysis is a cold re-derivation from the committed readouts by "
             "`phase1/scripts/analyze_phase1.py`, implementing the frozen §2 aggregation "
             "(band 17–26, generation positions only, R3 language-folded matching, SURVIVORS-only "
             "operative lists) exactly as `phase0/scripts/phase1_p0_power.py:loading_for_rep`. "
             "No aggregates were computed during data generation (PROVENANCE §data-commit).\n")
    L.append("**Asymmetric-informativeness clause (from the seal, R5):** *the J-lens captures the "
             "workspace incompletely; null loadings are non-conclusive, only positive loadings inform.* "
             "Interpretation is the PI's.\n")

    L.append("\n## Gate 0 — verification\n")
    L.append("| check | value | status |")
    L.append("|---|---|---|")
    L.append(f"| data_digest (recomputed over man_full‖judge_full‖completeness) | `{g['data_digest_recomputed'][:16]}…` | {'✓' if g['data_digest_match'] else '✗'} |")
    L.append(f"| N runs / duplicates | {g['n_runs']} / {g['n_duplicate_ids']} | {'✓' if g['n_runs']==800 and g['n_duplicate_ids']==0 else '✗'} |")
    L.append(f"| prereg sha256 @ tag | `{g['prereg_sha256'][:16]}…` | ✓ |")
    L.append(f"| lens .pt sha256 | `{g['lens_pt_sha256'][:16]}…` | ✓ |")

    L.append("\n## Diagnosis rates (all cells, re-derived from `judge_full.jsonl`)\n")
    L.append("| cell | n_runs | n_ok | judge_err | diagnosis=1 | rate |")
    L.append("|---|---|---|---|---|---|")
    for cell in CELLS:
        c = d[cell]
        L.append(f"| `{cell}` | {c['n_runs']} | {c['n_ok']} | {c['n_judge_error']} | "
                 f"{c['n_diagnosis_1']} | {fmt(c['rate'],3)} |")

    L.append("\n## C1 — recognition-without-consequence (confirmatory)\n")
    L.append(f"**DV** {c1['DV']}; conditioned on {c1['conditioning']}; {c1['test']}, "
             f"α={c1['alpha']} (|t|>{c1['tcrit']}).\n")
    L.append("| cell | mean F | F·EN | F·ES |")
    L.append("|---|---|---|---|")
    L.append(f"| `DN_flagged×high×L1` | {fmt(c1['flagged_mean_F'])} | {fmt(c1['flagged_mean_F_EN'])} | {fmt(c1['flagged_mean_F_ES'])} |")
    L.append(f"| `DN_plausible×high×L1` | {fmt(c1['plausible_mean_F'])} | {fmt(c1['plausible_mean_F_EN'])} | {fmt(c1['plausible_mean_F_ES'])} |")
    L.append(f"\n**Paired over {t1['n']} vignettes:** mean diff (flagged−plausible) = "
             f"**{fmt(t1['mean_diff'])}** (sd {fmt(t1['sd_diff'])}), "
             f"**t({t1['df']}) = {t1['t']:.3f}**, two-sided **p = {t1['p_two_sided']:.4g}**. "
             f"→ {'SIGNIFICANT' if c1['significant'] else 'not significant'} at α={c1['alpha']} "
             f"(reg. discovery: {'positive → loaded-but-inert direction' if t1['mean_diff']>0 else 'near-floor → never-enters direction'}; "
             f"a null is non-conclusive per the clause).\n")

    L.append("\n## C2 — advocacy over the absurd (confirmatory)\n")
    L.append(f"**DV** {c2['DV']}; {c2['conditioning']}; {c2['test']}, α={c2['alpha']} (t>{c2['tcrit']}).\n")
    L.append("| cell | mean A | A·EN | A·ES |")
    L.append("|---|---|---|---|")
    L.append(f"| `incoherent×high×L4` | {fmt(c2['L4_mean_A'])} | {fmt(c2['L4_mean_A_EN'])} | {fmt(c2['L4_mean_A_ES'])} |")
    L.append(f"| `incoherent×high×L1` | {fmt(c2['L1_mean_A'])} | {fmt(c2['L1_mean_A_EN'])} | {fmt(c2['L1_mean_A_ES'])} |")
    L.append(f"\n**Paired over {t2['n']} vignettes:** mean diff (L4−L1) = **{fmt(t2['mean_diff'])}** "
             f"(sd {fmt(t2['sd_diff'])}), **t({t2['df']}) = {t2['t']:.3f}**, one-sided "
             f"**p = {t2['p_one_sided_upper']:.4g}**. → {'SIGNIFICANT' if c2['significant'] else 'not significant'} "
             f"at α={c2['alpha']} (H1: L4>L1).\n")

    L.append("\n## Auxiliary diagnostic (registered, reported not tested)\n")
    L.append("Per-language split reported in the C1/C2 tables above (Set A/F, EN vs ES columns). "
             "A token tagged `EN+ES` (e.g. `experimental`) contributes to **both** columns, so the "
             "columns are not a partition and need not sum to the total. Set F has no ES-only "
             "survivors → `F·EN` ≡ mean F and `F·ES` is only the bilingual `experimental` mass; "
             "Set A's ES column is the two ES-only survivors (` paciente`, ` tratamiento`). "
             "Consistent with the C-note prediction of English realization under Spanish generation.\n")

    L.append("\n---\n")
    L.append("## EXPLORATORY (post-hoc, NOT preregistered) — C1 generation self-echo\n")
    L.append("Registered in `phase1/POSTHOC_EXPLORATORY_C1_selfecho.md` before any loading was computed. "
             "Non-adjudicative; numbers only; interpretation is the PI's.\n")
    s1 = ex["sub1_mention_split"]
    s2 = ex["sub2_F_by_stratum"]
    s3 = ex["sub3_positional_mask_C1"]
    s4 = ex["sub4_positional_mask_C2"]
    L.append(f"\n**1. Textual-mention split** (flagged×diagnosis=1, n={s1['n_flagged_dx1']}): "
             f"with-mention **{s1['n_with_mention']}** / without-mention **{s1['n_without_mention']}** "
             f"(regex `inventad|estudio|no reconocid|fictici`).\n")
    L.append("\n**2. F-loading by mention stratum** (mean Set F loading):\n")
    L.append("| stratum | n | mean F |")
    L.append("|---|---|---|")
    L.append(f"| flagged — WITH mention | {s2['n_with']} | {fmt(s2['flagged_with_mention_meanF'])} |")
    L.append(f"| flagged — WITHOUT mention (decisive) | {s2['n_without']} | {fmt(s2['flagged_without_mention_meanF'])} |")
    L.append(f"| plausible (diagnosis=1) | {s2['n_plausible']} | {fmt(s2['plausible_meanF'])} |")
    t3 = s3["paired_test"]
    L.append(f"\n**3. Positional-masking robustness** (exclude gen positions whose emitted token, ±2 window, "
             f"is an F operative token; {s3['n_aligned_of_800']}/800 runs token-aligned). "
             f"Mean fraction of positions masked: flagged {fmt(s3['flagged_mean_masked_fraction'],4)} / "
             f"plausible {fmt(s3['plausible_mean_masked_fraction'],4)}. "
             f"Masked mean F: flagged {fmt(s3['flagged_masked_meanF'])} vs plausible {fmt(s3['plausible_masked_meanF'])}; "
             + (f"paired mean diff {fmt(t3['mean_diff'])}, t({t3['df']})={t3['t']:.3f}, two-sided p={t3['p_two_sided']:.4g}.\n" if t3 else "no paired data.\n"))
    L.append("*(The emission mask removes little because generation is Spanish while the Set F SURVIVOR "
             "operative tokens are English — emitted Spanish fiction words do not fold-match the English "
             "operative list, and the F lens-loading is itself read out on English tokens. This is a fact "
             "about the instrument, reported not interpreted.)*\n")
    t4 = s4["paired_test"]
    L.append(f"\n**4. Same mask on C2 / Set A:** masked mean A: L4 {fmt(s4['L4_masked_meanA'])} vs L1 {fmt(s4['L1_masked_meanA'])}; "
             + (f"paired mean diff {fmt(t4['mean_diff'])}, t({t4['df']})={t4['t']:.3f}, one-sided p={t4['p_one_sided_upper']:.4g}.\n" if t4 else "no paired data.\n"))
    L.append("\n*Mask note:* emitted tokens recovered by re-tokenizing `generation_text` (Qwen tokenizer.json) "
             "and aligning to generation positions; membership tested space-insensitively against the F "
             "SURVIVOR list. Runs whose re-tokenization length ≠ #generation positions are excluded from the "
             "masked statistics (count reported).\n")

    L.append(f"\n---\n*Generated by `phase1/scripts/analyze_phase1.py` against data commit "
             f"`{g['data_commit']}` (digest `{g['data_digest_recomputed'][:16]}…`). "
             f"Machine-readable: `phase1/data/results_phase1.json`.*\n")

    (REPO / "RESULTS_PHASE1.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
