# Copyright 2026 — Phase 0 delegate. Mechanical execution of SEALED rules R1-R3.
"""Deterministic screening of the PI's sealed Phase 1 concept sets against the
Stage 0.1b inventory. This is EXECUTION of sealed rules, not decision:

  R1  realize each concept (+ listed morphological variants, +leading-space form)
      to the tokenizer's single tokens; drop & record multi-token-only concepts.
  R2  echo exclusion, substring-strict: exclude a realized token if it appears in
      the stimulus corpus at token-id OR substring level. The corpus's substring-
      token set is exactly Stage 0.1b `present_tokens_substring` (3076 ids); the
      token-id set is `present_tokens` (927). The Phase-1 task instruction is
      included per R2 (16 ids + substring check against its verbatim text).
  R3  case/diacritic folding (NFKD -> strip combining -> lowercase) applied to the
      matching. Fully-faithful against the inventory's *token/piece* set; the
      folded-substring-against-RAW-corpus residual is NOT computable on this host
      (raw corpus phase0/data/stimuli_src/ is gitignored / iMac-only) and is
      flagged. Expected null for these English-dominant sets vs a Spanish corpus.

Tokenizer only; no model, no GPU. Output: JSON + a markdown appendix for the seal.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import transformers

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR = M["model_path"]
INV = json.loads((REPO / "phase0" / "reports" / "stimulus_token_inventory.json").read_text(encoding="utf-8"))


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


# ---- SEALED concept sets (verbatim from the sealed document) ----
# each concept is a list of its surface variants (the "/"-separated forms).
SETS = {
    "A_generic_nosological": {
        "role": "core confirmatory",
        "concepts": [["disorder"], ["diagnosis", "diagnostic"], ["syndrome"],
                     ["condition"], ["pathology", "pathological"], ["symptom"],
                     ["clinical"], ["illness"], ["disease"], ["patient"],
                     ["treatment"], ["therapy"], ["chronic"]],
    },
    "B1_seed_gloss": {
        "role": "echo stratum only, never confirmatory",
        "concepts": [["narrative"], ["coherence", "coherent"], ["self"],
                     ["memory"], ["identity"], ["past"], ["life"],
                     ["emotional", "emotion"], ["story"]],
    },
    "B2_name_etymology": {
        "role": "own stratum, reported separately",
        "concepts": [["attention", "attentional"], ["focus"],
                     ["distraction", "distracted"]],
    },
    "C_real_anchor_DPDR": {
        "role": "positive control (single-token neighbors only, per R1)",
        "concepts": [["unreal"], ["detached", "detachment"], ["dream"],
                     ["fog", "foggy"], ["distant"], ["numb"], ["observer"],
                     ["dissociation", "dissociative"]],
    },
    "D_negative_control": {
        "role": "floor",
        "concepts": [["hobby"], ["routine"], ["weekend"], ["neighbor"],
                     ["weather"], ["commute"]],
    },
    "F_disclosure_fictional": {
        "role": "flagged contrast (scope-restricted per R5)",
        "concepts": [["fiction", "fictional", "fictitious"], ["invented"],
                     ["study"], ["experiment", "experimental"], ["fabricated"],
                     ["real"]],
    },
}


def main() -> int:
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)

    # ---- R2 exclusion reference (from sealed Stage 0.1b inventory) ----
    present_ids = set(INV["present_tokens"]["ids"])
    substr_ids = set(INV["present_tokens_substring"]["ids"])
    instr = INV["stage03_instruction"]
    instr_ids = set(instr["ids"])
    instr_text = instr["text"]
    instr_text_folded = fold(instr_text)

    exclusion_ids = present_ids | substr_ids | instr_ids          # R2 exact (id-level)
    # R3: folded forms of every excluded token's piece (decoded consistently here)
    folded_excluded_pieces = {fold(tok.decode([i])) for i in exclusion_ids}

    def realize(variant: str):
        """Return single-token (id, piece) realizations for a surface variant,
        both bare and leading-space (R1)."""
        out = []
        for form in (variant, " " + variant):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                out.append((ids[0], form))
        return out

    def status_for(tid: int) -> dict:
        piece = tok.decode([tid])
        f = fold(piece)
        r2_exact = tid in exclusion_ids
        r3_folded = (f in folded_excluded_pieces) or (f.strip() and f.strip() in instr_text_folded)
        excluded = r2_exact or r3_folded
        return {"id": tid, "piece": piece, "folded": f,
                "r2_exact_echo": r2_exact, "r3_folded_echo": bool(r3_folded),
                "status": "ECHO_excluded" if excluded else "SURVIVES"}

    result = {}
    for set_name, spec in SETS.items():
        concept_rows = []
        dropped = []
        for variants in spec["concepts"]:
            realized = []
            seen = set()
            for v in variants:
                for tid, form in realize(v):
                    if tid not in seen:
                        seen.add(tid)
                        realized.append({**status_for(tid), "from_form": form})
            if not realized:
                dropped.append({"concept": "/".join(variants),
                                "reason": "multi-token-only (no single-token realization)"})
            else:
                concept_rows.append({"concept": "/".join(variants), "realized": realized})
        survivors = [r for c in concept_rows for r in c["realized"] if r["status"] == "SURVIVES"]
        echoes = [r for c in concept_rows for r in c["realized"] if r["status"] == "ECHO_excluded"]
        result[set_name] = {
            "role": spec["role"],
            "concepts": concept_rows,
            "dropped_multitoken": dropped,
            "n_survivors": len(survivors),
            "n_echo_excluded": len(echoes),
            "survivor_folded_keys": sorted({r["folded"] for r in survivors}),
        }

    out_json = REPO / "phase0" / "data" / "phase1_seal_screening.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- markdown appendix ----
    md = []
    md.append("## APPENDIX — Operative lists (mechanical R1–R3 execution at seal time)\n")
    md.append("**Produced:** 2026-07-20 by `phase0/scripts/phase1_seal_screening.py` "
              "(deterministic; tokenizer Qwen2.5-7B-Instruct @ a09a3545). "
              "This is rule execution, not decision.\n")
    md.append("**Exclusion reference (R2):** Stage 0.1b inventory — "
              f"`present_tokens` ({len(present_ids)}) ∪ `present_tokens_substring` "
              f"({len(substr_ids)}) ∪ task-instruction tokens ({len(instr_ids)}). "
              "R2 substring-strict is exact and complete against the inventory. "
              "**R3 folded-substring residual:** the raw stimulus corpus "
              "(`phase0/data/stimuli_src/`, gitignored) is not on this host, so folded "
              "matching is applied against the inventory's token/piece set (and the "
              "verbatim instruction text), not a re-folded raw corpus; additional folded "
              "matches existing only across corpus token boundaries are not captured — "
              "expected null for these English-dominant concept sets against a Spanish "
              "corpus, flagged for optional closure against the corpus.\n")
    md.append("**Legend:** SURVIVES = passes R2/R3 (operative). ECHO_excluded = present "
              "in corpus/instruction (echo stratum, reported separately, barred from "
              "confirmatory measures). Scoring matches generated tokens by **folded** "
              "form (R3), so case/leading-space/diacritic variants of a survivor are "
              "covered without separate listing.\n")
    for set_name, r in result.items():
        md.append(f"\n### {set_name}  ·  _{r['role']}_")
        md.append(f"survivors: **{r['n_survivors']}** · echo-excluded: {r['n_echo_excluded']} "
                  f"· dropped (multi-token-only): {len(r['dropped_multitoken'])}")
        if r["dropped_multitoken"]:
            md.append("- dropped: " + ", ".join(d["concept"] for d in r["dropped_multitoken"]))
        md.append("")
        md.append("| concept | token id | piece | folded | status |")
        md.append("|---|---|---|---|---|")
        for c in r["concepts"]:
            for rr in c["realized"]:
                md.append(f"| {c['concept']} | {rr['id']} | `{rr['piece']}` | "
                          f"`{rr['folded']}` | {rr['status']} |")
        md.append(f"\n**Operative folded scoring keys ({set_name}):** "
                  + (", ".join(f"`{k}`" for k in r["survivor_folded_keys"]) or "_(none)_"))
    md_text = "\n".join(md) + "\n"
    (REPO / "phase0" / "data" / "phase1_seal_appendix.md").write_text(md_text, encoding="utf-8")

    # console summary
    print("R1-R3 screening complete.")
    for set_name, r in result.items():
        print(f"  {set_name}: survivors={r['n_survivors']} echo={r['n_echo_excluded']} "
              f"dropped={len(r['dropped_multitoken'])}")
    print(f"\n-> {out_json}")
    print(f"-> {REPO / 'phase0' / 'data' / 'phase1_seal_appendix.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
