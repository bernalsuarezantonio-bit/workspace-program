#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b / Stage J1 — mechanical smoke adjudication + completeness.

MECHANICAL ONLY. NO content aggregates, NO loadings, NO rates. Counts and
integrity. Nothing here computes a DV: diagnosis counts are reported as raw
integrity counts (judge coverage / parse errors), never as a rate, and the ES
mention DV is not touched at all -- both belong to the separate analysis session.

--smoke : adjudicate the 6 smoke runs against 5 pre-declared criteria.
          Exit 0 = PASS (proceed to confirmatory), nonzero = FAIL (stop, report).
--full  : completeness of the 600-run set -- exact N, 0 duplicate trial_ids, every
          readout present with sha256 == manifest, all top-k weights finite, rows
          confined to the band and the readout window, ablation-landing counts per
          arm, judge coverage + parse errors, N per arm. Writes
          phase2b/data/completeness_report.json. Exit 0 = complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[2]

ARMS = ["B0_none", "B1_full", "B3_rand"]
BAND = list(range(17, 27))
EXPECT_FULL = {"arms": 3, "vignettes": 20, "reps": 10, "total": 600}
BUDGET_HOURS = 3.0
SEC_PER_RUN_JUDGE_ASSUMED = 6.0


def sh(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_run(readout_path: Path, m: dict) -> list[str]:
    probs = []
    if not readout_path.exists():
        return [f"missing readout {readout_path.name}"]
    blob = readout_path.read_bytes()
    if sh(blob) != m["readout_sha256"]:
        probs.append("sha256 != manifest (digest re-verify FAIL)")
    d = json.loads(blob)
    rows = d["rows"]
    if not rows:
        probs.append("no rows")
        return probs
    if {r["layer"] for r in rows} - set(BAND):
        probs.append("rows outside band 17-26")
    if {r["segment"] for r in rows} != {"generation"}:
        probs.append("non-generation rows present")
    lo, hi = d["meta"]["readout_window"]
    pos = {r["position"] for r in rows}
    if min(pos) < lo or max(pos) >= hi:
        probs.append(f"positions outside readout window [{lo},{hi})")
    if not all(isinstance(c["weight"], float) and c["weight"] == c["weight"]
               for r in rows for c in r["topk"]):
        probs.append("non-finite top-k weight")
    # ablation landing: exact position counts per arm
    exp = 0 if m["arm"] == "B0_none" else hi - lo if lo == m["prompt_tokens"] else hi - m["prompt_tokens"]
    if m["arm"] == "B0_none":
        if m["touched_positions_generate"] != 0 or m["touched_positions_readout"] != 0:
            probs.append("B0_none touched positions (must be 0)")
    else:
        want = m["total_positions"] - 1 - m["prompt_tokens"]
        if m["touched_positions_generate"] != want:
            probs.append(f"generate touched {m['touched_positions_generate']} != {want}")
        if m["touched_positions_readout"] != want:
            probs.append(f"readout touched {m['touched_positions_readout']} != {want}")
    _ = exp
    if m["model_digest"] != d["meta"]["model"] or m["lens_pt_sha256"] != d["meta"]["lens_pt_sha256"]:
        probs.append("digest mismatch manifest vs readout")
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gen-minutes", type=float, default=None,
                    help="measured smoke generation minutes, for the budget projection")
    args = ap.parse_args()
    if not (args.smoke or args.full):
        sys.exit("choose --smoke or --full")
    tag = "smoke" if args.smoke else "full"

    D = REPO / "phase2b" / "data"
    man = load_jsonl(D / f"run_manifest_{tag}.jsonl")
    jud = load_jsonl(D / f"judge_{tag}.jsonl")
    ro_dir = D / ("readouts_smoke" if args.smoke else "readouts")

    ids = [m["trial_id"] for m in man]
    dups = len(ids) - len(set(ids))
    problems = {}
    for m in man:
        p = check_run(REPO / m["readout_file"], m)
        if p:
            problems[m["trial_id"]] = p

    jud_by = {j["trial_id"]: j for j in jud}
    n_judge_err = sum(1 for j in jud if "judge_error" in j)
    n_judge_ok = len(jud) - n_judge_err
    missing_judge = [i for i in ids if i not in jud_by]

    per_arm = {a: sum(1 for m in man if m["arm"] == a) for a in ARMS}
    landing = {a: {"runs": 0, "touched_gen_total": 0, "touched_ro_total": 0} for a in ARMS}
    for m in man:
        L = landing[m["arm"]]
        L["runs"] += 1
        L["touched_gen_total"] += m["touched_positions_generate"]
        L["touched_ro_total"] += m["touched_positions_readout"]

    if args.smoke:
        gen_min = args.gen_minutes
        crit = {
            "C1_all_runs_present_and_intact": len(problems) == 0 and dups == 0,
            "C2_expected_smoke_count": len(man) == len(ARMS) * 2,
            "C3_judge_parses_all": len(missing_judge) == 0 and n_judge_err == 0,
            "C4_ablation_landed_per_arm": (
                landing["B0_none"]["touched_gen_total"] == 0
                and landing["B1_full"]["touched_gen_total"] > 0
                and landing["B3_rand"]["touched_gen_total"] > 0
                and landing["B1_full"]["touched_gen_total"]
                == landing["B1_full"]["touched_ro_total"]
                and landing["B3_rand"]["touched_gen_total"]
                == landing["B3_rand"]["touched_ro_total"]),
        }
        proj = None
        if gen_min is not None:
            per_run_gen = gen_min * 60 / len(man)
            proj = (EXPECT_FULL["total"] * (per_run_gen + SEC_PER_RUN_JUDGE_ASSUMED)) / 3600.0
            crit["C5_projected_within_budget"] = proj <= BUDGET_HOURS
        ok = all(crit.values())
        rep = {"stage": "Phase 2b J1 smoke adjudication", "prereg_tag": "prereg-phase2b-v1",
               "n_runs": len(man), "duplicates": dups, "problems": problems,
               "judge_ok": n_judge_ok, "judge_errors": n_judge_err,
               "missing_judge": missing_judge, "per_arm": per_arm, "landing": landing,
               "projected_full_hours": proj, "budget_hours": BUDGET_HOURS,
               "criteria": crit, "PASS": bool(ok)}
        (D / "smoke_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=2))
        if problems:
            print("PROBLEMS:", json.dumps(problems, indent=2))
        print(f"\nSMOKE {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 3

    # ---- full ----
    complete = (len(man) == EXPECT_FULL["total"] and dups == 0 and not problems
                and not missing_judge
                and all(per_arm[a] == EXPECT_FULL["vignettes"] * EXPECT_FULL["reps"]
                        for a in ARMS))
    digest = sh((D / "run_manifest_full.jsonl").read_bytes()
                + (D / "judge_full.jsonl").read_bytes())
    rep = {
        "stage": "Phase 2b J1 completeness", "prereg_tag": "prereg-phase2b-v1",
        "prereg_commit": "aa77d66",
        "note": "mechanical integrity only -- no rates, no loadings, no DV computed",
        "expected": EXPECT_FULL, "n_runs": len(man), "duplicates": dups,
        "n_runs_with_problems": len(problems), "problems": problems,
        "per_arm": per_arm, "landing": landing,
        "judge_total": len(jud), "judge_ok": n_judge_ok, "judge_errors": n_judge_err,
        "judge_error_trials": [j["trial_id"] for j in jud if "judge_error" in j],
        "missing_judge": missing_judge,
        "n_malformed_flagged": sum(1 for m in man if m.get("malformed")),
        "malformed_trials": [m["trial_id"] for m in man if m.get("malformed")],
        "manifest_plus_judge_sha256": digest,
        "COMPLETE": bool(complete),
    }
    (D / "completeness_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=2))
    if problems:
        print("PROBLEMS:", json.dumps(problems, indent=2))
    print(f"\nCOMPLETE = {complete}")
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
