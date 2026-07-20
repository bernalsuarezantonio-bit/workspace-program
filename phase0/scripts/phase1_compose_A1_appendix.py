# Copyright 2026 — Phase 0 delegate. Compose + append A1 appendix to the sealed doc.
"""Reads the A1 bilingual screening result, builds the dated A1 amendment appendix,
appends it to phase1_token_sets_SEALED.md, normalizes to LF, and prints the pre-A1
(A0-final) and post-A1 sha256. Run as a FILE (never inline) so nothing parses the
markdown backticks."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[2]
d = json.loads((REPO / "phase0" / "data" / "phase1_seal_screening_A1.json").read_text(encoding="utf-8"))
BT = "`"

L = []
L.append("---\n")
L.append("# SEAL AMENDMENT A1 — Bilingual realization (dated 2026-07-20, pre-data)\n")
L.append("**Amends the seal per its own rule** (\"amendments only as dated, appended "
         "justifications — never silent edits\"; \"no changes once any condition-bearing "
         "readout exists\"). No condition-bearing readout exists at A1 time (only Stage 0.2 "
         "calibration + Stage 0.3 v12 pilot + the 20-rep technical calibration — no "
         "conditions/sets/counting). The problem was revealed by the mechanical A0 "
         "screening, not by any condition data.\n")
L.append("**Change:** every concept in every set (A–F) is now realized **bilingually "
         "(EN + ES)**. English realizations are **unchanged** from A0. Spanish "
         "realizations are the PI-signed list (2026-07-20). Added to R1: each operative "
         "token carries a **language tag**; R3 folding applies identically to both "
         "languages; loadings will be reported **per-language and aggregated** "
         "(aggregation itself remains open to the Phase 1 prereg).\n")
L.append("**PI-signed ES adjudications (2026-07-20):** (1) condition -> condición+afección; "
         "(2) illness/disease -> enfermedad+dolencia (collision benign — unit of measure is "
         "the SET, not the EN-ES pair; R1 note); (3) self -> **no ES realization** (drop "
         "recorded; \"yo\" = high-frequency pronoun noise, clean forms multi-token); "
         "(4) memory -> memoria+recuerdo; (5) story -> historia+relato; (6) focus -> "
         "concentración+concentrado/-a; (7) detached -> desconexión/desconectado/-a + "
         "distanciamiento/distanciado/-a; (8) numb -> embotado/embotamiento+insensible; "
         "(9) weather -> clima; (10) weekend/commute drops accepted per R1; "
         "(11) fabricated -> fabricado/-a (overlap w/ invented benign). Rest as proposed.\n")
L.append("**Exclusion reference (R2):** unchanged — Stage 0.1b inventory "
         "(present_tokens ∪ present_tokens_substring ∪ instruction tokens).\n")

L.append("## A1 mechanical finding (factual, for PI — not a decision)\n")
L.append("The Qwen2.5-7B tokenizer (English/Chinese-weighted) renders **most Spanish "
         "concept realizations multi-token**, so they are **dropped and recorded per R1** "
         "(e.g. trastorno->3 tok, síntoma->3, terapia->2, síndrome, irreal->2, sueño->3, "
         "disociación, clima->2, rutina->2, vecino->2). Spanish **operative** tokens are "
         "therefore the common-word subset that is single-token (usually via the "
         "leading-space form): e.g. " + BT + " paciente" + BT + ", " + BT + " tratamiento" + BT +
         ", " + BT + " memoria" + BT + ", " + BT + " pasado" + BT + ", " + BT + " atención" + BT +
         ", " + BT + "experimental" + BT + ". **The amendment nonetheless achieves its "
         "targeted goals:** B1 now has a real **echo stratum** in Spanish — " + BT + "vida" + BT +
         "/" + BT + " vida" + BT + "/" + BT + " historia" + BT + " are excluded by R2 (resolving "
         "A0's B1 echo=0 contradiction) — and F's " + BT + " estudio" + BT + " is now correctly "
         "echo-excluded (it survived in A0). **Set C (DPDR anchor):** its Spanish lexicon is "
         "entirely multi-token -> **0 ES operative tokens**; Set C remains English-only in "
         "practice. Pre-data mechanical observation; amendable pre-data (e.g. accept EN-only "
         "C explicitly, or a future A2 with alternative Spanish forms). Not decided by delegate.\n")

for sn, r in d.items():
    L.append(f"## {sn}  ·  _{r['role']}_")
    drops = ", ".join(f"{x['concept']}[{x['lang']}]" for x in r["dropped"]) or "_none_"
    L.append(f"survivors {r['survivors_by_lang']} · echo {r['echo_by_lang']} · drops: {drops}")
    L.append("")
    L.append("| concept | id | piece | lang | folded | status |")
    L.append("|---|---|---|---|---|---|")
    for cr in r["concepts"]:
        for rr in cr["realized"]:
            L.append(f"| {cr['concept']} | {rr['id']} | {BT}{rr['piece']}{BT} | "
                     f"{rr['langs']} | {BT}{rr['folded']}{BT} | {rr['status']} |")
    L.append("")

appendix = "\n".join(L) + "\n"

sealed = REPO / "phase1_token_sets_SEALED.md"
pre = sealed.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
pre_hash = hashlib.sha256(pre).hexdigest()
final = pre.decode("utf-8").rstrip() + "\n\n" + appendix
final_bytes = final.encode("utf-8")
sealed.write_bytes(final_bytes)
post_hash = hashlib.sha256(final_bytes).hexdigest()

print("PRE-A1  (A0-final) sha256:", pre_hash)
print("POST-A1 (with A1)  sha256:", post_hash)
print("final bytes:", len(final_bytes))
