# Copyright 2026 — Phase 0 delegate. Append dated SEAL NOTE (Set C accepted EN-only).
"""Appends a dated, pre-data note to phase1_token_sets_SEALED.md recording the PI's
acceptance (2026-07-20) of Set C operating English-only (no A2). Documents an
acceptance + a registered prediction; changes NO set content or rule. Run as a file.
Prints pre (pre-note) and post (with note) LF sha256."""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BT = "`"

note = f"""---

# SEAL NOTE C-EN — Set C accepted English-only (dated 2026-07-20, pre-data)

Records the PI's decision (2026-07-20) to accept **Set C operating English-only**, following the A1 mechanical finding that the Spanish DPDR-anchor lexicon is entirely multi-token under Qwen2.5 and drops per R1. This note documents an **acceptance and a registered prediction**; it changes **no** set content and **no** rule. **No A2 is issued.** No condition-bearing readout exists at note time.

1. **Justification.** C's Spanish-lexicon drop is R1 behaving correctly on a real property of the substrate: Qwen2.5's single-token vocabulary is poor in clinical Spanish. Hunting alternative Spanish synonyms chosen because they "survive" R1 would **invert the concept->realization direction that R1 exists to protect** (realization follows from the tokenizer; it is never selected to hit a target). Accepting the drop keeps R1 intact.

2. **Registered prediction (turns the acceptance into a testable bet).** Given the documented cross-lingual phenomenon — the workspace paper, and our own Stage 0.2 Tier 2 ({BT}Italy{BT}/{BT}意大利{BT}, {BT}euros{BT}/{BT}欧元{BT} read out for Spanish/Italian-referent content) — we predict the workspace will realize the anchor concepts in **English tokens even under Spanish stimulus and Spanish generation**. Set C (EN-only) is therefore expected to load on anchor-compatible material regardless of the surface language. **Auxiliary diagnostic:** the A1 per-language breakdown in the sets that do have ES operative tokens (A, B1, B2) indexes how much Spanish-token load exists in general — a low ES load there corroborates the English-realization expectation; a substantial ES load there would qualify it. If the intermediate layers load English despite Spanish context, that is the paper's cross-lingual finding replicated in this domain; if they load Spanish, it is a qualification of the paper. Either is a registered outcome, not a forking path.

3. **Residual risk covered by a sealed rule.** Any under-capture from C being EN-only is covered by the already-sealed **asymmetric-informativeness rule (R5)**: a null in C is non-conclusive, like every null; only positive loadings inform. EN-only C therefore cannot produce a misleading false negative — it can only fail to detect, which the sealed rule already declares non-conclusive.

**Instrument status: CLOSED.** No further set/rule changes this session. Next: Phase 1 contrast design and preregistration (outside this prompt).
"""

sealed = REPO / "phase1_token_sets_SEALED.md"
pre = sealed.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
pre_hash = hashlib.sha256(pre).hexdigest()
final = pre.decode("utf-8").rstrip() + "\n\n" + note
final_bytes = final.encode("utf-8")
sealed.write_bytes(final_bytes)
post_hash = hashlib.sha256(final_bytes).hexdigest()
print("PRE-note  (A1 state) sha256:", pre_hash)
print("POST-note (with C-note) sha256:", post_hash)
print("final bytes:", len(final_bytes))
