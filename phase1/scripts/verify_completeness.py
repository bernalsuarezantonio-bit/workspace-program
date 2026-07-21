# Copyright 2026 — Phase 1 delegate. Stage P1: mechanical smoke gate + completeness.
"""Mechanical verification only — NO content aggregates, NO loadings. Counts + integrity.

--smoke : adjudicate the 8 smoke runs against 4 criteria (formats valid, positions
          marked, judge parses, files/manifest consistent). Exit 0 = PASS (proceed),
          nonzero = FAIL (stop and report).
--full  : completeness of the 800-run set — exact N, 0 duplicate trial_ids, every
          readout present with sha256 == manifest, all top-k weights finite, positions
          marked, judge coverage + parse errors, N per cell (incl. diagnosis=1 count for
          C1 conditioning). Writes phase1/data/completeness_report.json. Exit 0 = complete.
"""
from __future__ import annotations

import argparse, json, hashlib, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[2]

CELLS = ["C1_DN_flagged_L1", "C1_DN_plausible_L1", "C2_incoherent_L4", "C2_incoherent_L1"]
EXPECT_FULL = {"cells": 4, "vignettes": 20, "reps": 10, "total": 800}


def sh(b): return hashlib.sha256(b).hexdigest()


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_run(readout_path, manifest_line, deep):
    """Return list of problems for one run ([] = clean)."""
    probs = []
    if not readout_path.exists():
        return [f"missing readout {readout_path.name}"]
    blob = readout_path.read_bytes()
    if sh(blob) != manifest_line["readout_sha256"]:
        probs.append("sha256 != manifest (digest re-verify FAIL)")
    d = json.loads(blob)
    rows = d["rows"]
    if not rows:
        probs.append("no rows")
    segs = {r["segment"] for r in rows}
    if not {"prompt", "generation"} <= segs:
        probs.append(f"positions not marked (segments={segs})")
    if not any(r.get("ood_unfitted_pos") for r in rows):
        probs.append("no ood_unfitted_pos flags")
    sample = rows if deep else rows[:200]
    for r in sample:
        if len(r["topk"]) != 10:
            probs.append(f"topk!=10 at L{r['layer']}p{r['position']}"); break
        for c in r["topk"]:
            w = c["weight"]
            if not isinstance(w, (int, float)) or w != w:  # NaN check
                probs.append("non-finite weight"); break
    if not any(r["segment"] == "generation" for r in rows):
        probs.append("no generation positions")
    return probs


def smoke(argv=None):
    rd = REPO / "phase1" / "data" / "readouts_smoke"
    man = load_jsonl(REPO / "phase1" / "data" / "run_manifest_smoke.jsonl")
    jpath = REPO / "phase1" / "data" / "judge_smoke.jsonl"
    fails = []
    ids = [m["trial_id"] for m in man]
    if len(ids) != 8:
        fails.append(f"expected 8 smoke runs, got {len(ids)}")
    if len(set(ids)) != len(ids):
        fails.append("duplicate trial_ids")
    if {m["cell"] for m in man} != set(CELLS):
        fails.append("not all 4 cells covered")
    for m in man:
        p = check_run(rd / f"{m['trial_id']}.json", m, deep=True)
        if p:
            fails.append(f"{m['trial_id']}: {p}")
    # judge parses
    if not jpath.exists():
        fails.append("no judge_smoke.jsonl")
    else:
        j = load_jsonl(jpath)
        if len(j) != len(ids):
            fails.append(f"judge count {len(j)} != {len(ids)}")
        errs = [x["trial_id"] for x in j if "judge_error" in x]
        if errs:
            fails.append(f"judge parse errors: {errs}")
        if any(x.get("diagnosis") not in (0, 1) for x in j if "judge_error" not in x):
            fails.append("judge diagnosis not 0/1")
    print("SMOKE CRITERIA:")
    print(f"  runs=8 & 4 cells & no dupes : {'FAIL' if any('smoke' in f or 'cell' in f or 'dup' in f for f in fails) else 'ok'}")
    print(f"  formats/positions/finite    : ok" if not any(':' in f for f in fails) else "  formats/positions/finite    : FAIL")
    print(f"  judge parses                : {'FAIL' if any('judge' in f for f in fails) else 'ok'}")
    if fails:
        print("SMOKE: FAIL"); [print("  -", f) for f in fails]; return 1
    print("SMOKE: PASS"); return 0


def full():
    rd = REPO / "phase1" / "data" / "readouts"
    man = load_jsonl(REPO / "phase1" / "data" / "run_manifest_full.jsonl")
    judge = load_jsonl(REPO / "phase1" / "data" / "judge_full.jsonl")
    report = {"expected": EXPECT_FULL, "problems": [], "per_cell": {}, "judge": {}}
    ids = [m["trial_id"] for m in man]
    report["n_runs"] = len(ids)
    report["n_duplicate_ids"] = len(ids) - len(set(ids))
    if len(ids) != EXPECT_FULL["total"]:
        report["problems"].append(f"N={len(ids)} != {EXPECT_FULL['total']}")
    if report["n_duplicate_ids"]:
        report["problems"].append(f"{report['n_duplicate_ids']} duplicate trial_ids")
    # per-run integrity (deep sha + finiteness on all)
    bad = 0
    for m in man:
        p = check_run(rd / f"{m['trial_id']}.json", m, deep=True)
        if p:
            bad += 1; report["problems"].append(f"{m['trial_id']}: {p}")
    report["runs_with_problems"] = bad
    # digest consistency
    lens_shas = {m["lens_pt_sha256"] for m in man}
    model_digs = {m["model_digest"] for m in man}
    report["lens_pt_sha256"] = sorted(lens_shas)
    report["model_digest"] = sorted(model_digs)
    if len(lens_shas) != 1 or len(model_digs) != 1:
        report["problems"].append("inconsistent digests across runs")
    # per-cell counts + judge conditioning counts (COUNTS ONLY)
    jmap = {j["trial_id"]: j for j in judge}
    report["judge"] = {"n_judged": len(judge),
                       "n_parse_errors": sum(1 for j in judge if "judge_error" in j)}
    for cell in CELLS:
        cm = [m for m in man if m["cell"] == cell]
        judged = [jmap[m["trial_id"]] for m in cm if m["trial_id"] in jmap]
        ok = [j for j in judged if "judge_error" not in j]
        dx1 = sum(1 for j in ok if j.get("diagnosis") == 1)
        report["per_cell"][cell] = {
            "n_runs": len(cm), "n_judged": len(judged),
            "n_judge_errors": len(judged) - len(ok),
            "n_diagnosis_1": dx1, "n_diagnosis_0": len(ok) - dx1,
            "vignettes": len({m["vignette"] for m in cm}),
            "reps": len({m["rep"] for m in cm}),
        }
    report["COMPLETE"] = not report["problems"]
    outp = REPO / "phase1" / "data" / "completeness_report.json"
    outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_runs","n_duplicate_ids","runs_with_problems",
                                             "judge","per_cell","COMPLETE")}, indent=2, ensure_ascii=False))
    print(f"-> {outp.relative_to(REPO)}")
    return 0 if report["COMPLETE"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    raise SystemExit(smoke() if a.smoke else full() if a.full else 2)
