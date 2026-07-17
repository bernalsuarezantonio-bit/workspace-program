#!/usr/bin/env python3
"""Stage 0.1b — R2b stimulus token inventory.

Builds the full stimulus corpus (legitimacy wrapper x disorder condition x vignette)
exactly as reification-gradient's src/run_experiment.py assembles it, tokenizes every
stimulus with a declared reference tokenizer, and emits the complete inventory of
unique tokens appearing verbatim anywhere in any stimulus.

Purpose: let the PI apply the R2b rule (confirmatory token sets restricted to tokens
absent from ALL stimuli) mechanically at sealing time.

This script INVENTORIES ONLY. It does not select, filter, rank, or propose measurement
token sets. That decision is the PI's and is out of scope here.

Deterministic: no randomness, no sampling. Output is a pure function of
(stimuli_src contents, tokenizer revision).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
import sys

import yaml
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent  # phase0/
SRC = ROOT / "data" / "stimuli_src"
REPORTS = ROOT / "reports"

# Declared reference tokenizer, pinned by revision sha (PROVENANCE.md).
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

# --- Assembly replicated verbatim from reification-gradient src/run_experiment.py -----
# (commit ee23c07288a31eb19545c944e0662bd6a2d9d915). Kept byte-identical on purpose:
# any divergence here would make the inventory describe stimuli that were never shown.


def wrap(template_text: str, name: str, payload: str, disclosure: str = "") -> str:
    body = payload.strip()
    if disclosure:
        body = body + " " + disclosure.strip()
    return template_text.replace("{{NAME}}", name).replace("{{PAYLOAD}}", body)


def build_prompt(seeded_context: str, vignette_text: str) -> str:
    return (
        f"{seeded_context}\n\n"
        "Eres un clínico. Evalúa el siguiente caso a la luz de la categoría anterior.\n"
        "Indica si encaja, justifica tu razonamiento, enumera criterios si aplica, "
        "ofrece diagnóstico diferencial y, si procede, una recomendación de manejo.\n\n"
        f"CASO:\n{vignette_text.strip()}"
    )


# Stage 0.3 pilot instruction, from the Phase 0 brief. Not part of the behavioural
# corpus, but it IS shown to the model, so it is inventoried separately and flagged.
STAGE03_INSTRUCTION = (
    "Read the following case and reason about what might be going on with this person."
)


def load():
    disorders = yaml.safe_load((SRC / "disorders.yaml").read_text())["disorders"]
    vignettes = yaml.safe_load((SRC / "vignettes.yaml").read_text())["vignettes"]
    levels = {p.stem: p.read_text() for p in sorted((SRC / "legitimacy").glob("L*.md"))}
    return disorders, vignettes, levels


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    disorders, vignettes, levels = load()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, revision=TOKENIZER_REVISION)

    stimuli: list[dict] = []
    for (dkey, d), (lkey, ltext), v in itertools.product(
        disorders.items(), levels.items(), vignettes
    ):
        seeded = wrap(ltext, d["name"], d["payload"], d.get("disclosure", ""))
        prompt = build_prompt(seeded, v["text"])
        stimuli.append(
            {"disorder": dkey, "level": lkey, "vignette": v["id"], "text": prompt}
        )

    # Union of token ids over every stimulus. `add_special_tokens=False`: special tokens
    # are chat-template scaffolding, not verbatim stimulus content; they are reported
    # separately below rather than silently folded into the corpus inventory.
    corpus_ids: set[int] = set()
    per_stimulus_len: list[int] = []
    for s in stimuli:
        ids = tok.encode(s["text"], add_special_tokens=False)
        corpus_ids.update(ids)
        per_stimulus_len.append(len(ids))

    # Second, STRICTLY BROADER notion of "appears verbatim". BPE is context-dependent,
    # so a vocab id can be absent from every tokenization while its decoded string still
    # occurs in the text (e.g. "logo" inside "psicologo"), and a word present in the text
    # can tokenize to different ids when encoded standalone. Which notion the R2b rule
    # means is the PI's call, so both sets are emitted and neither is privileged.
    corpus_text = "\n".join(s["text"] for s in stimuli)
    substring_ids: set[int] = set()
    for i in range(len(tok)):
        piece = tok.decode([i])
        if piece and piece in corpus_text:
            substring_ids.add(i)

    instr_ids = set(tok.encode(STAGE03_INSTRUCTION, add_special_tokens=False))

    # Provenance of the exact source bytes this inventory was computed from.
    src_hashes = {
        str(p.relative_to(SRC)): sha256(p.read_text())
        for p in sorted(SRC.rglob("*"))
        if p.is_file()
    }

    inventory = {
        "schema": "r2b-stimulus-token-inventory/1",
        "purpose": (
            "Complete inventory of tokens appearing verbatim in any stimulus, so the "
            "PI can apply the R2b rule mechanically. Inventory only; no token set is "
            "proposed or endorsed here."
        ),
        "tokenizer": {
            "model": TOKENIZER_MODEL,
            "revision": TOKENIZER_REVISION,
            "class": type(tok).__name__,
            "vocab_size": tok.vocab_size,
            "len_tokenizer": len(tok),
            "add_special_tokens": False,
        },
        "source": {
            "repo": "reification-gradient",
            "path": "/Users/admin/Downloads/reification-gradient/materials",
            "commit": "ee23c07288a31eb19545c944e0662bd6a2d9d915",
            "file_sha256": src_hashes,
        },
        "corpus": {
            "n_disorders": len(disorders),
            "n_levels": len(levels),
            "n_vignettes": len(vignettes),
            "n_stimuli": len(stimuli),
            "disorders": sorted(disorders),
            "levels": sorted(levels),
            "tokens_per_stimulus": {
                "min": min(per_stimulus_len),
                "max": max(per_stimulus_len),
                "mean": round(sum(per_stimulus_len) / len(per_stimulus_len), 1),
            },
        },
        "present_tokens": {
            "n_unique": len(corpus_ids),
            "coverage_fraction_of_vocab": round(len(corpus_ids) / len(tok), 6),
            "ids": sorted(corpus_ids),
            "id_to_piece": {
                str(i): tok.decode([i]) for i in sorted(corpus_ids)
            },
        },
        "stage03_instruction": {
            "text": STAGE03_INSTRUCTION,
            "note": (
                "Shown to the model in Stage 0.3 but not part of the behavioural "
                "corpus. Inventoried separately: PI to decide whether R2b exclusion "
                "extends to it."
            ),
            "n_unique": len(instr_ids),
            "ids": sorted(instr_ids),
            "ids_not_already_in_corpus": sorted(instr_ids - corpus_ids),
        },
        "present_tokens_substring": {
            "definition": (
                "Vocab ids whose decoded string occurs as a SUBSTRING anywhere in the "
                "corpus, regardless of how BPE actually segmented the text. Strict "
                "superset of present_tokens. Broader/more conservative reading of "
                "'appears verbatim'."
            ),
            "n_unique": len(substring_ids),
            "coverage_fraction_of_vocab": round(len(substring_ids) / len(tok), 6),
            "ids": sorted(substring_ids),
            "id_to_piece": {str(i): tok.decode([i]) for i in sorted(substring_ids)},
        },
        "absent_tokens": {
            "note": (
                "R2b-eligible pool = all vocab ids MINUS the chosen present set. Not "
                "enumerated here (it is the complement, ~150k ids); derive it at "
                "sealing time from this file. Which present set to subtract "
                "(token-level vs substring-level) is a PI decision, not made here."
            ),
            "n_eligible_if_token_level": len(tok) - len(corpus_ids),
            "n_eligible_if_substring_level": len(tok) - len(substring_ids),
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "stimulus_token_inventory.json"
    out.write_text(json.dumps(inventory, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    print(f"  stimuli:        {len(stimuli)}")
    print(f"  unique tokens:  {len(corpus_ids)}")
    print(f"  R2b-eligible:   {len(tok) - len(corpus_ids)}")
    print(f"  inventory sha256: {sha256(out.read_text())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
