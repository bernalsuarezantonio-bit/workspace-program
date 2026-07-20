# Copyright 2026 — Phase 0 delegate. SEAL AMENDMENT A1: bilingual R1-R3 execution.
"""Amendment A1 (dated 2026-07-20, pre-data): bilingual EN+ES realization of every
concept in every set. The English forms are IDENTICAL to the A0 seal (unchanged);
Spanish forms are added per the PI's signed adjudications (2026-07-20). Each
operative token carries a language tag. R1-R3 apply identically to both languages.
Loadings will be reported per-language and aggregated (aggregation itself open to
prereg). This is rule execution on PI-signed set content, not decision.

Signed ES adjudications applied (see NOTES in output):
 (1) condition -> condición + afección   (2) illness/disease -> enfermedad + dolencia
     (collision benign: unit of measure is the SET, not the EN-ES pair; R1 note)
 (3) self -> NO ES realization (drop recorded; 'yo' = high-freq pronoun noise)
 (4) memory -> memoria + recuerdo   (5) story -> historia + relato
 (6) focus -> concentración + concentrado/-a   (7) detached -> desconexión/
     desconectado/-a + distanciamiento/distanciado/-a   (8) numb -> embotado/
     embotamiento + insensible   (9) weather -> clima   (10) weekend/commute drops
     accepted per R1   (11) fabricated -> fabricado/-a (overlap w/ invented benign)
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


# Bilingual sealed concept sets. en = A0-sealed (unchanged); es = A1 (PI-signed).
SETS = {
    "A_generic_nosological": {"role": "core confirmatory", "concepts": [
        ("disorder", ["disorder"], ["trastorno"]),
        ("diagnosis/diagnostic", ["diagnosis", "diagnostic"], ["diagnóstico", "diagnóstica"]),
        ("syndrome", ["syndrome"], ["síndrome"]),
        ("condition", ["condition"], ["condición", "afección"]),
        ("pathology/pathological", ["pathology", "pathological"], ["patología", "patológico", "patológica"]),
        ("symptom", ["symptom"], ["síntoma"]),
        ("clinical", ["clinical"], ["clínico", "clínica"]),
        ("illness", ["illness"], ["enfermedad", "dolencia"]),
        ("disease", ["disease"], ["enfermedad", "dolencia"]),
        ("patient", ["patient"], ["paciente"]),
        ("treatment", ["treatment"], ["tratamiento"]),
        ("therapy", ["therapy"], ["terapia"]),
        ("chronic", ["chronic"], ["crónico", "crónica"]),
    ]},
    "B1_seed_gloss": {"role": "echo stratum only, never confirmatory", "concepts": [
        ("narrative", ["narrative"], ["narrativa", "narrativo"]),
        ("coherence/coherent", ["coherence", "coherent"], ["coherencia", "coherente"]),
        ("self", ["self"], []),  # ES dropped per PI adjudication (3)
        ("memory", ["memory"], ["memoria", "recuerdo"]),
        ("identity", ["identity"], ["identidad"]),
        ("past", ["past"], ["pasado"]),
        ("life", ["life"], ["vida"]),
        ("emotional/emotion", ["emotional", "emotion"], ["emocional", "emoción"]),
        ("story", ["story"], ["historia", "relato"]),
    ]},
    "B2_name_etymology": {"role": "own stratum, reported separately", "concepts": [
        ("attention/attentional", ["attention", "attentional"], ["atención", "atencional"]),
        ("focus", ["focus"], ["concentración", "concentrado", "concentrada"]),
        ("distraction/distracted", ["distraction", "distracted"], ["distracción", "distraído", "distraída"]),
    ]},
    "C_real_anchor_DPDR": {"role": "positive control (single-token neighbors, per R1)", "concepts": [
        ("unreal", ["unreal"], ["irreal"]),
        ("detached/detachment", ["detached", "detachment"],
         ["desconexión", "desconectado", "desconectada", "distanciamiento", "distanciado", "distanciada"]),
        ("dream", ["dream"], ["sueño"]),
        ("fog/foggy", ["fog", "foggy"], ["niebla", "nebuloso", "borroso"]),
        ("distant", ["distant"], ["distante", "lejano"]),
        ("numb", ["numb"], ["embotado", "embotamiento", "insensible"]),
        ("observer", ["observer"], ["observador", "observadora"]),
        ("dissociation/dissociative", ["dissociation", "dissociative"], ["disociación", "disociativo", "disociativa"]),
    ]},
    "D_negative_control": {"role": "floor", "concepts": [
        ("hobby", ["hobby"], ["afición", "pasatiempo"]),
        ("routine", ["routine"], ["rutina"]),
        ("weekend", ["weekend"], ["fin de semana"]),  # ES expected multi-token drop per R1
        ("neighbor", ["neighbor"], ["vecino", "vecina"]),
        ("weather", ["weather"], ["clima"]),
        ("commute", ["commute"], ["desplazamiento", "trayecto"]),
    ]},
    "F_disclosure_fictional": {"role": "flagged contrast (scope-restricted per R5)", "concepts": [
        ("fiction/fictional/fictitious", ["fiction", "fictional", "fictitious"], ["ficción", "ficticio", "ficticia"]),
        ("invented", ["invented"], ["inventado", "inventada"]),
        ("study", ["study"], ["estudio"]),
        ("experiment/experimental", ["experiment", "experimental"], ["experimento", "experimental"]),
        ("fabricated", ["fabricated"], ["fabricado", "fabricada"]),
        ("real", ["real"], ["real"]),
    ]},
}


def main() -> int:
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    present_ids = set(INV["present_tokens"]["ids"])
    substr_ids = set(INV["present_tokens_substring"]["ids"])
    instr = INV["stage03_instruction"]
    exclusion_ids = present_ids | substr_ids | set(instr["ids"])
    instr_text_folded = fold(instr["text"])
    folded_excluded = {fold(tok.decode([i])) for i in exclusion_ids}

    def realize(variant):
        out = []
        for form in (variant, " " + variant):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                out.append((ids[0], form))
        return out

    def status_for(tid):
        piece = tok.decode([tid]); f = fold(piece)
        r2 = tid in exclusion_ids
        r3 = (f in folded_excluded) or (f.strip() and f.strip() in instr_text_folded)
        return piece, f, r2, bool(r3), ("ECHO_excluded" if (r2 or r3) else "SURVIVES")

    result = {}
    for set_name, spec in SETS.items():
        concept_rows, dropped = [], []
        for concept, en_forms, es_forms in spec["concepts"]:
            realized = []
            seen = {}  # id -> langs
            for lang, forms in (("EN", en_forms), ("ES", es_forms)):
                lang_had_single = False
                for v in forms:
                    for tid, form in realize(v):
                        lang_had_single = True
                        if tid in seen:
                            seen[tid]["langs"].add(lang)
                        else:
                            piece, f, r2, r3, st = status_for(tid)
                            rowd = {"id": tid, "piece": piece, "folded": f, "langs": {lang},
                                    "from_form": form, "r2_exact_echo": r2,
                                    "r3_folded_echo": r3, "status": st}
                            seen[tid] = rowd
                            realized.append(rowd)
                # record language-specific multi-token-only drop
                if forms and not lang_had_single:
                    dropped.append({"concept": concept, "lang": lang,
                                    "forms": forms, "reason": "multi-token-only (R1)"})
                if lang == "ES" and not forms:
                    dropped.append({"concept": concept, "lang": "ES",
                                    "forms": [], "reason": "no ES realization (PI adjudication)"})
            for r in realized:
                r["langs"] = "+".join(sorted(r["langs"]))
            if realized:
                concept_rows.append({"concept": concept, "realized": realized})
        # tally by language
        def tally(status):
            c = {"EN": 0, "ES": 0, "EN+ES": 0}
            for cr in concept_rows:
                for r in cr["realized"]:
                    if r["status"] == status:
                        c[r["langs"]] = c.get(r["langs"], 0) + 1
            return c
        result[set_name] = {
            "role": spec["role"], "concepts": concept_rows, "dropped": dropped,
            "survivors_by_lang": tally("SURVIVES"), "echo_by_lang": tally("ECHO_excluded"),
            "n_survivors": sum(1 for cr in concept_rows for r in cr["realized"] if r["status"] == "SURVIVES"),
            "n_echo": sum(1 for cr in concept_rows for r in cr["realized"] if r["status"] == "ECHO_excluded"),
        }

    (REPO / "phase0" / "data" / "phase1_seal_screening_A1.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("A1 bilingual R1-R3 screening — per set: survivors / echo / drops (by lang)")
    for sn, r in result.items():
        print(f"  {sn}: surv {r['survivors_by_lang']} | echo {r['echo_by_lang']} | "
              f"drops {[(d['concept'],d['lang']) for d in r['dropped']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
