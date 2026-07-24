#!/usr/bin/env python3
# Copyright 2026.
"""Study 1 — mention analysis on DN_flagged x high x L1, re-derived (never committed before).

The 120 responses = 20 high vignettes x 3 reps x 2 model families, disorder
DN_flagged, level L1_forum, from the read-only reification-gradient clone. Text
comes from responses.jsonl @ 770fa9c; the diagnosis flag comes from
scored_full.jsonl @ c4a5ce8, joined on (model, disorder, level, vignette, rep).

Fiction-mention pattern list (PI-specified, Spanish surface forms):
  inventad · ficticio* · no reconocid* · para un estudio · no corresponde ·
  hipotetic* · no existe
Each pattern is matched case-insensitively and accent-folded so 'ficticio',
'ficticia', 'hipotético', 'hipotética' etc. all count. Per-pattern counts, the
aggregate 'any mention' rate, and 10 deterministic verbatims (diagnosis=1 AND
any-mention, selected by sorted full id) are written out.

This number was never committed; if it differs from any prior chat claim (e.g.
66.7%), it is reported as-is and the manuscript is corrected, not the number.

Read-only against reification-gradient (no writes). Run:
  .venv/Scripts/python.exe paper/rederive_mentions_behavioral.py <scratch_beh_dir>
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

OUT_JSON = Path(__file__).resolve().parent / "mentions_behavioral.json"
OUT_MD = Path(__file__).resolve().parent / "mentions_behavioral_verbatims.md"

# PI-specified patterns. Keys are the display label; values are regexes applied
# to the accent-folded, lower-cased text. '*' in the spec => stem match.
PATTERNS = {
    "inventad": r"inventad",
    "ficticio*": r"fictici",                 # ficticio/ficticia/ficticios/ficticias
    "no reconocid*": r"no reconocid",        # no reconocida/reconocido/reconocidas
    "para un estudio": r"para un estudio",
    "no corresponde": r"no corresponde",
    "hipotetic*": r"hipotetic",              # hipotetico/a (accent-folded)
    "no existe": r"no existe",
}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    beh = Path(sys.argv[1])
    resp_p = beh / "responses_770fa9c.jsonl"
    scored_p = beh / "scored_c4a5ce8.jsonl"

    raw = [r for r in load(resp_p) if "model" in r]
    scored = load(scored_p)
    key = lambda r: (r["model"], r["disorder"], r["level"], r["vignette"], r["rep"])
    diag = {key(s): int(s["diagnosis"]) for s in scored if not s.get("error")}

    sub = [r for r in raw
           if r["disorder"] == "DN_flagged" and r["level"] == "L1_forum"
           and r["target_compatibility"] == "high"]
    assert len(sub) == 120, f"expected 120, got {len(sub)}"

    compiled = {lab: re.compile(rx) for lab, rx in PATTERNS.items()}
    per_pattern = {lab: 0 for lab in PATTERNS}
    any_mention = 0
    rows = []
    for r in sub:
        k = key(r)
        folded = fold(r["response"])
        hits = {lab: bool(c.search(folded)) for lab, c in compiled.items()}
        has_any = any(hits.values())
        for lab, h in hits.items():
            per_pattern[lab] += int(h)
        any_mention += int(has_any)
        full_id = f"{k[0].split('/')[-1]}|{k[3]}|rep{k[4]}"
        rows.append({"id": full_id, "model": k[0], "vignette": k[3], "rep": k[4],
                     "diagnosis": diag.get(k), "any_mention": has_any,
                     "patterns_hit": [lab for lab, h in hits.items() if h],
                     "response": r["response"]})

    n = len(sub)
    n_diag1 = sum(1 for x in rows if x["diagnosis"] == 1)
    rate_any = any_mention / n

    # verbatims: diagnosis==1 AND any_mention, deterministic by sorted id.
    # Family-alphabetical id order puts all mistral before any qwen, so the top-10
    # were entirely mistral; the qwen block is the first 5 eligible qwen responses
    # by the same (vignette, rep) order.
    elig = sorted([x for x in rows if x["diagnosis"] == 1 and x["any_mention"]],
                  key=lambda x: x["id"])
    verbatims = elig[:10]
    elig_qwen = [x for x in elig if x["model"].split("/")[-1].startswith("qwen")]
    verbatims_qwen = elig_qwen[:5]

    summary = {
        "cell": "DN_flagged x high x L1_forum (behavioral, reification-gradient)",
        "n_responses": n,
        "n_diagnosis_1": n_diag1,
        "provenance": {
            "text": "resultados_tirada_real/responses.jsonl @ 770fa9c",
            "diagnosis": "phase6/scored_full.jsonl @ c4a5ce8",
            "responses_sha256": sha256_file(resp_p),
            "scored_sha256": sha256_file(scored_p)},
        "patterns": PATTERNS,
        "per_pattern_counts": per_pattern,
        "per_pattern_rate": {k: round(v / n, 4) for k, v in per_pattern.items()},
        "any_mention_count": any_mention,
        "any_mention_rate": round(rate_any, 4),
        "any_mention_rate_pct": round(100 * rate_any, 1),
        "n_eligible_verbatims_diag1_and_mention": len(elig),
        "n_eligible_qwen": len(elig_qwen),
        "verbatim_ids": [v["id"] for v in verbatims],
        "verbatim_ids_qwen": [v["id"] for v in verbatims_qwen],
    }
    OUT_JSON.write_text(json.dumps(
        {"summary": summary, "per_response": rows,
         "verbatims": verbatims, "verbatims_qwen": verbatims_qwen},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # verbatims markdown (full text, for human verification + paper appendix)
    def block(v):
        return [f"## {v['id']}", "",
                f"**diagnosis** = {v['diagnosis']} · **patterns** = {', '.join(v['patterns_hit'])}",
                "", "> " + v["response"].replace("\n", "\n> "), "", "---", ""]

    md = ["# Mention verbatims — DN_flagged x high x L1 (diagnosis=1 AND fiction mention)", "",
          f"Deterministic selection by sorted id (`model|vignette|rep`). Eligible = judged "
          f"diagnosis=1 AND at least one fiction-mention pattern. {len(elig)} eligible "
          f"({len(elig_qwen)} of them qwen).", "",
          f"Source: text `responses.jsonl @ 770fa9c`, diagnosis `scored_full.jsonl @ c4a5ce8`.", "",
          "# Section A — first 10 by sorted id (all mistral-small3.1:24b)", ""]
    for v in verbatims:
        md += block(v)
    md += ["# Section B — first 5 qwen2.5:32b by sorted id", ""]
    for v in verbatims_qwen:
        md += block(v)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"n={n}  diagnosis=1: {n_diag1}")
    print("per-pattern counts:")
    for lab in PATTERNS:
        print(f"  {lab:18s} {per_pattern[lab]:3d}  ({per_pattern[lab]/n:.3f})")
    print(f"ANY mention: {any_mention}/{n} = {rate_any:.4f} ({100*rate_any:.1f}%)")
    print(f"eligible verbatims (diag=1 & mention): {len(elig)} ({len(elig_qwen)} qwen); "
          f"showing 10 mistral + 5 qwen")
    print("qwen ids:", [v["id"] for v in verbatims_qwen])
    print("wrote", OUT_JSON.name, "+", OUT_MD.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
