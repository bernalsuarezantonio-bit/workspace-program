# Stage 0.1b — R2b Stimulus Token Inventory

**Date:** 2026-07-17
**Status:** Complete. Inventory only — no measurement token set is proposed, ranked, or endorsed here (brief rule 4).
**Machine-readable output:** [`stimulus_token_inventory.json`](stimulus_token_inventory.json) (sha256 `eadc1a00eff7918dd90633fcd3ef7b0d2b637084778e94677edc000dd43769d5`)
**Script:** [`../scripts/build_token_inventory.py`](../scripts/build_token_inventory.py)
**Logs:** `phase0/logs/20260717T*_stage01b_inventory*.log`

---

## 1. What was inventoried

The full stimulus corpus, assembled exactly as the behavioural study assembles it:

| | |
|---|---|
| Disorder conditions | 4 — `DN_plausible`, `DN_flagged` (incl. disclosure line), `real_anchor`, `incoherent` |
| Legitimacy wrappers | 5 — `L1_forum`, `L2_coach_blog`, `L3_wiki`, `L4_preprint`, `L5_pseudodsm` |
| Vignettes | 60 (`v01`–`v60`, post-Fase-1b) |
| **Total stimuli** | **1,200** (4 × 5 × 60) |
| Tokens per stimulus | min 141, max 205, mean 168.2 |

**Reference tokenizer (declared):** `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, class `Qwen2Tokenizer`, `vocab_size` 151,643, `len(tokenizer)` 151,665. Chosen to match the Stage 0.2/0.3 pilot model you designated. Encoding uses `add_special_tokens=False` — chat-template scaffolding is not verbatim stimulus content.

**Source provenance:** `reification-gradient` @ commit `ee23c07288a31eb19545c944e0662bd6a2d9d915`, copied read-only to `phase0/data/stimuli_src/` and `chmod a-w`. Per-file sha256 recorded in the JSON and in `PROVENANCE.md`. HEAD was read directly from `.git/HEAD` and `.git/refs/heads/main` rather than by running git against that repo (brief rule 3 — `git status` and friends can write index/lock files).

**Assembly fidelity.** `wrap()` and `build_prompt()` are copied byte-identically from `src/run_experiment.py` @ `ee23c07`. Two consequences worth knowing, both faithful to what the model actually saw:

- The wrappers are read with `p.read_text()` — **whole file, including the markdown header comments**. So `# L1 — Foro / red social (autoridad mínima)` and L1's `# {{NAME}} y {{PAYLOAD}} se inyectan en tiempo de ejecución.` are part of the prompt, with placeholders substituted. Their tokens are in the inventory because they are in the stimuli.
- The fixed instruction block (`Eres un clínico. Evalúa el siguiente caso…`) and the `CASO:` header appear verbatim in every one of the 1,200 stimuli, so their tokens are unavoidably present.

## 2. Result

| Quantity | Token-level | Substring-level |
|---|---|---|
| Unique tokens present in corpus | **927** | **3,076** |
| Fraction of vocabulary | 0.61% | 2.03% |
| **R2b-eligible pool (absent from all stimuli)** | **150,738** | **148,589** |

The eligible pool is the complement and is not enumerated in the JSON (it is ~150k ids); derive it at sealing time as `range(len(tok)) - present_ids`.

## 3. The one thing you need to decide: which notion of "appears verbatim"

This is the substantive finding of Stage 0.1b, and it is a fork in the R2b rule that I am flagging rather than resolving.

**BPE tokenization is context-dependent, so "token present" and "string present" are not the same set.** Two concrete demonstrations from our own corpus:

**(a) A word plainly in the text whose standalone encoding is "absent."** The word *narrativa* appears in every DN stimulus. But:

```
encode("narrativa")            -> [77 'n', 1118 'arr', 27852 'ativa']
encode(...in context...)       -> [..., 13408 ' narr', 27852 'ativa', 13 '.']
```

Ids 77 and 1118 **never occur** in any stimulus — the in-context segmentation uses ` narr` (13408) instead. A naive check of "is `encode('narrativa')` in the present set?" returns **False** for a word that is in the corpus 900 times.

**(b) A string plainly in the text whose token id is "absent."** Id 10129 decodes to `logo`. The string `logo` occurs in the corpus inside *psicólogo* (L1's "no soy psicólogo ni nada eh"), which segments as `no`/` soy`/` ps`/`ic`/`ólogo` — id 10129 never occurs as a token. I sampled ~4,000 vocab ids outside the token-level set and found 41 such cases.

So:

- **Token-level (927 ids)** = ids actually emitted by the tokenizer on some stimulus. This is the notion that matches issue #5's input-copying ceiling mechanically: the lens reads out *the token at that position*, and only an actually-emitted token can be copied.
- **Substring-level (3,076 ids)** = ids whose decoded string occurs anywhere in the corpus text. Strict superset (verified: token-level ⊆ substring-level). Costs 2,149 more ids of eligible pool — a rounding error against 150k — and forecloses the residual worry that a probe token's string is sitting inside a stimulus word.

I have **not** picked one. Both sets are in the JSON under `present_tokens.ids` and `present_tokens_substring.ids`, and `absent_tokens` reports the eligible count under each. My only observation is that the price of the stricter reading is negligible (1.4% of the eligible pool), which may make the choice easy — but it is yours.

There is a third notion I did **not** compute and want to name so it isn't silently missed: **case- and diacritic-insensitive** matching (*clínico* vs *clinico* vs *CLÍNICO*). All numbers here are exact-match. If R2b is meant to exclude near-variants, say so and I'll add it.

## 4. Stage 0.3 instruction — inventoried separately, flagged

The brief's pilot instruction — *"Read the following case and reason about what might be going on with this person."* — is shown to the model but is not part of the behavioural corpus. I inventoried it separately (`stage03_instruction` in the JSON) rather than folding it in, because whether R2b exclusion extends to it is a design decision. Note it is **English**, while the entire behavioural corpus is **Spanish**, so it contributes tokens the Spanish stimuli do not.

## 5. Verification performed

I adversarially checked my own output rather than trusting it:

1. **Soundness** — recomputed the union independently, from the read-only copies, with a separate reimplementation: 927 ids, set-identical to the reported set. ✓
2. **Completeness** — sampled ~4,000 vocab ids *not* in the token-level set and searched the concatenated corpus for their decoded strings. The 41 hits were all the substring/BPE artefact of §3(b), not omissions. This check is what surfaced §3. ✓
3. **Subset relation** — token-level ⊆ substring-level, verified programmatically. ✓
4. **Determinism** — no randomness in the script; output is a pure function of source bytes and tokenizer revision. Reruns reproduce the same sha256.

## 6. Caveat on reuse

This inventory is **tokenizer-specific**. All counts are void if the pilot model changes: a different tokenizer re-segments the same text into different ids. If the model decision is ever revisited, rerun the script with the new revision pinned — it takes ~2 minutes.
