#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b analysis — Gate 0. Verification ONLY, before any computation.

Checks, all from disk / committed artifacts:
  G1 data commit contents: N=600, 3 arms x 200, 0 duplicate trial_ids
  G2 judge coverage: 600 scores, 0 parse errors
  G3 malformed flagged: 0
  G4 data content digest recomputes to aa56df8d...
  G5 touched-positions integrity per arm (B0 zero; B1/B3 generate == readout ==
     total-1-prompt for EVERY run, not just in aggregate)
  G6 model / lens digests consistent across all 600 manifest lines
  G7 prereg blob sha256 at the tagged commit

Computes NO dependent variable. Exit 0 = pass, nonzero = STOP.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "phase2b" / "data"
ARMS = ["B0_none", "B1_full", "B3_rand"]
EXPECT = {"total": 600, "per_arm": 200}
DIGEST_EXPECTED = "aa56df8d5c6cfa7acef1792721f3b156f00ad6568d2f73953139538f049b0592"
PREREG_SHA_AT_TAG = "f17ac3656177db8a35586201addfc6c3d29d470d66897f397ebb4ab97a3dd8c5"
LENS_SHA = "3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29"
MODEL_DIGEST = "Qwen/Qwen2.5-7B-Instruct@a09a3545"


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    man = jl(D / "run_manifest_full.jsonl")
    jud = jl(D / "judge_full.jsonl")
    checks: dict = {}

    ids = [m["trial_id"] for m in man]
    per_arm = {a: sum(1 for m in man if m["arm"] == a) for a in ARMS}
    checks["G1_n_runs"] = len(man)
    checks["G1_per_arm"] = per_arm
    checks["G1_duplicates"] = len(ids) - len(set(ids))
    checks["G1_pass"] = bool(len(man) == EXPECT["total"]
                             and all(per_arm[a] == EXPECT["per_arm"] for a in ARMS)
                             and len(ids) == len(set(ids)))

    n_err = sum(1 for j in jud if "judge_error" in j)
    missing = sorted(set(ids) - {j["trial_id"] for j in jud})
    checks["G2_judge_total"] = len(jud)
    checks["G2_judge_errors"] = n_err
    checks["G2_missing"] = missing
    checks["G2_pass"] = bool(len(jud) == EXPECT["total"] and n_err == 0 and not missing)

    n_mal = sum(1 for m in man if m.get("malformed"))
    checks["G3_malformed"] = n_mal
    checks["G3_pass"] = n_mal == 0

    h = hashlib.sha256()
    for f in ("run_manifest_full.jsonl", "judge_full.jsonl"):
        h.update((D / f).read_bytes())
    got = h.hexdigest()
    checks["G4_digest"] = got
    checks["G4_expected"] = DIGEST_EXPECTED
    checks["G4_pass"] = got == DIGEST_EXPECTED

    bad = []
    for m in man:
        want = 0 if m["arm"] == "B0_none" else m["total_positions"] - 1 - m["prompt_tokens"]
        if m["touched_positions_generate"] != want or m["touched_positions_readout"] != want:
            bad.append({"trial_id": m["trial_id"], "arm": m["arm"], "want": want,
                        "gen": m["touched_positions_generate"],
                        "ro": m["touched_positions_readout"]})
    checks["G5_runs_with_bad_touch"] = len(bad)
    checks["G5_examples"] = bad[:5]
    checks["G5_totals"] = {a: {"gen": sum(m["touched_positions_generate"]
                                          for m in man if m["arm"] == a),
                               "ro": sum(m["touched_positions_readout"]
                                         for m in man if m["arm"] == a)} for a in ARMS}
    checks["G5_pass"] = len(bad) == 0

    mds = {m["model_digest"] for m in man}
    lss = {m["lens_pt_sha256"] for m in man}
    checks["G6_model_digests"] = sorted(mds)
    checks["G6_lens_shas"] = sorted(lss)
    checks["G6_pass"] = bool(mds == {MODEL_DIGEST} and lss == {LENS_SHA})

    checks["G7_prereg_sha_at_tag"] = PREREG_SHA_AT_TAG
    checks["G7_note"] = ("verified out-of-band: git cat-file blob "
                         "d9f037f:PREREG_PHASE2B.md | sha256sum")

    ok = all(checks[k] for k in checks if k.endswith("_pass"))
    checks["GATE0_PASS"] = bool(ok)
    (D / "gate0_report.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print(json.dumps(checks, indent=2))
    print(f"\nGATE 0: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
