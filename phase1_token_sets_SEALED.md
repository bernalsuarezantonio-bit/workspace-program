# Phase 1 Token Sets — SEALED

**Seal date (PI approval):** 2026-07-17
**Sealing procedure executed (git external mark):** 2026-07-20 on `DESKTOP-6NEU2TB`
**Status:** approved by PI in session (2026-07-17); SEALED upon completion of the sealing procedure below. No J-lens readout of any study vignette under any experimental condition existed at approval time (the only readouts in existence are the Stage 0.2 calibration examples and the single Stage 0.3 feasibility pilot on v12, run without token sets, without conditions, and without counting). This remains true at the git seal: the interim Stage 0.3 nightly calibration run (v12, no conditions, no sets, no counting) does not create condition-bearing data.

*Delegate note (factual, non-substantive):* the seal date recorded above is the PI's approval date; the immutable external timestamp is the git commit/push on 2026-07-20. Both are stated for transparency. The operative token lists in the appendix were produced by mechanical execution of the sealed rules R1–R3 against the Stage 0.1b inventory — rule execution, not decision.

---

## Purpose

This document pre-fixes the token sets and matching rules that operationalize "concept loading in the workspace" (J-lens readouts) for Phase 1 of the workspace program. Fixing sets and rules before any condition-bearing data exists is the program's defense against garden-of-forking-paths in the representational measure.

The seal covers the sets and the rules (R1–R5). Aggregation formulas (how loadings are summed across layers/positions, layer bands, normalization — including any length-normalization required by the flagged-length confound noted in R5) are sketched conceptually here but frozen at the Phase 1 preregistration, informed by pilot feasibility (format, volume) — never by condition contrasts.

## Sealing procedure

1. Rename to `phase1_token_sets_SEALED.md`; add seal date.
2. `sha256sum phase1_token_sets_SEALED.md` → record hash in PROVENANCE.md.
3. Commit to `workspace-program` with message `seal: phase1 token sets (pre-data)` and push — the GitHub timestamp is the external mark.
4. After the seal, no changes once any condition-bearing readout exists. Before that, amendments only as dated, appended justifications — never silent edits.

## Rules (part of the seal)

**R1 — Concept-to-token realization.** Concepts are defined at the concept level; their realization is whatever vocabulary tokens the pilot model's tokenizer (Qwen2.5-7B-Instruct @ a09a3545) assigns, including leading-space variants and listed morphological variants. Multi-token-only concepts are dropped and recorded — never substituted post hoc.

**R2 — Echo exclusion, substring-strict (PI decision #1).** A candidate token is excluded from all confirmatory sets if it appears in the stimulus corpus at either token-id level or substring level. Justification: BPE tokenization is context-dependent; a stimulus word can resurface in generation tokenized differently, and token-id matching alone would misclassify that echo as emergent. Cost: ~1.4% of the candidate pool (Stage 0.1b inventory: 3,076 substring-level vs 927 token-level exclusions). The stimulus corpus for this rule = all legitimacy wrappers (L1–L5) × all four disorder-condition texts (DN_flagged disclosure line included) × all 60 vignettes × the exact task instruction text used in Phase 1 runs (constant across conditions, but its tokens would otherwise load as uniform echo and inflate the floor).

**R3 — Case- and diacritic-insensitive matching (PI decision #2).** All matching in R2 and in readout scoring folds case and diacritics (Clínico/clinico/CLINICO are one). Justification: Spanish-language stimuli meet an English-biased verbalizable vocabulary; near-variants are the same concept. The folding function is fixed here: Unicode NFKD → strip combining marks → lowercase.

**R4 — Generation-only rule (PI decision #6; HARD RULE).** Confirmatory readouts are taken exclusively from positions the model generates — never from prompt positions, and in particular never from positions 0–15. Justification: (a) the lens is unfitted on the first 16 positions (fit/apply asymmetry, verified in source); (b) input-copying dominates prompt positions — a token present in the prompt reads out at ~rank 1 at its own position — confirmed three independent times (upstream issue #5; the ledger's arithmetic check; our own Stage 0.2 ascii-face observation). Prompt-position readouts may be dumped for exploratory description but are barred from every confirmatory measure.

**R5 — Set-scope restrictions.**
- Set F (disclosure) is used only in DN_flagged vs DN_plausible contrasts, never in any analysis involving the real anchor: its concepts (real/fiction/unreality) collide lexically with DPDR symptomatology — the collision documented in the behavioral study's recognition probe (19/19 false negatives) generalizes to any lexical instrument near this anchor.
- The DN_flagged stimulus is ~23 tokens longer than DN_plausible with byte-identical payload (disclosure text); any flagged-vs-plausible aggregation must address length/priming (normalization decided at prereg, flagged here as a known confound).
- Asymmetric informativeness (standing program rule, restated for the seal): the J-lens captures the workspace incompletely; null loadings are non-conclusive, only positive loadings inform. This goes verbatim into the Phase 1 preregistration.

## Sets

All sets pass R1–R3 mechanically before freezing their final token lists; the surviving lists (post-inventory screening) are appended to this document at seal time as the operative sets. Predictions are registered per set for the Phase 1 contrasts (final hypotheses at prereg).

**A. Generic nosological (core confirmatory).** Concepts: disorder, diagnosis/diagnostic, syndrome, condition, pathology/pathological, symptom, clinical, illness, disease, patient, treatment, therapy, chronic. Role: does the model load diagnostic framing while reasoning? Primary loading DV across conditions and levels.

**B. Fabricated-category constituents ("disprosexia narrativa").**
- **B1 — seed-gloss constituents** (narrative, coherence/coherent, self, memory, identity, past, life, emotional/emotion, story): appear in stimuli → echo stratum only, reported separately, never confirmatory (per R2).
- **B2 — name-etymology constituents** (attention/attentional, focus, distraction/distracted): own stratum, reported separately (PI decision #3, option c). If the model decomposes the pseudo-Greek name into attention-concepts, that is a nameable finding (compositional decomposition of a fabricated label) — neither contaminates the emergent measure nor gets lost.
- **B3 — emergent elaboration:** not pre-listable by definition; the confirmatory emergent load rests on Set A (post-R2); DN-specific emergent tokens observed in readouts go to a clearly-labeled exploratory appendix.

**C. Real-anchor constituents** (anchor: DPDR — confirmed, no longer provisional; sealed in the behavioral prereg with v2h justification). Concepts (single-token neighbors; the disorder names are multi-token): unreal, detached/detachment, dream, fog/foggy, distant, numb, observer, dissociation/dissociative (if single-token post-R1). Role: positive control — should load when reasoning about DPDR-compatible material regardless of condition manipulations. Note: this set contains unreality lexicon by necessity (it IS the symptomatology); per R5 it never co-analyzes with Set F.

**D. Negative control (PI decision #4: no formal frequency matching; declared limitation).** Concepts: hobby, routine, weekend, neighbor, weather, commute. Role: floor. Limitation declared at prereg: frequency comparability asserted informally, not matched — rigorous multilingual frequency matching is out of scope; the set's job is to be a floor, not a twin.

**F. Disclosure / fictional-status (PI decision #5; NEW — scope-restricted per R5).** Concepts: fiction/fictional/fictitious, invented, study, experiment/experimental, fabricated, real (and folded variants). Role: the flagged contrast — when the model diagnoses a category it was explicitly told is fictional (the behavioral study's recognition-without-consequence cell, 120/120), is the fictional status loaded in the workspace during the diagnostic reasoning, or present only in output text? Two pre-registered architectures of failure: (i) loaded-but-inert (F loads during generation yet diagnosis proceeds) vs (ii) never-enters (F absent from generation-epoch workspace). Distinguishing them is the core representational question of the flagged contrast. Restriction: R5 — never near the anchor; "study/experiment" also appear in the disclosure text itself, so F is measured at generation positions only (R4 handles this) and F tokens appearing in stimuli are governed by R2 like all others — the echo stratum for F is reported separately from its emergent stratum.

## Registered set-level predictions (directional, informal until prereg)

1. Set A loads above Set D in all clinical conditions (sanity floor/ceiling).
2. Set C loads specifically when reasoning about anchor-compatible material (positive control).
3. B2 loading, if present, tracks the fabricated name's presence (decomposition finding).
4. The flagged contrast: whether F loads during generation in flagged-diagnosing runs adjudicates loaded-but-inert vs never-enters. No directional bet registered — this is the discovery question.

## Open at prereg (explicitly NOT sealed here)

Aggregation formula, layer bands, epoch definition within generation, normalization (incl. flagged-length), final hypothesis set and α structure, cell/contrast selection, storage policy (full dumps vs pre-aggregated loadings given ~6.3 GB per full replicate).

---

<!-- OPERATIVE LISTS APPENDIX APPENDED AT SEAL TIME BELOW THIS LINE -->

## APPENDIX — Operative lists (mechanical R1–R3 execution at seal time)

**Produced:** 2026-07-20 by `phase0/scripts/phase1_seal_screening.py` (deterministic; tokenizer Qwen2.5-7B-Instruct @ a09a3545). This is rule execution, not decision.

**Exclusion reference (R2):** Stage 0.1b inventory — `present_tokens` (927) ∪ `present_tokens_substring` (3076) ∪ task-instruction tokens (16). R2 substring-strict is exact and complete against the inventory. **R3 folded-substring residual:** the raw stimulus corpus (`phase0/data/stimuli_src/`, gitignored) is not on this host, so folded matching is applied against the inventory's token/piece set (and the verbatim instruction text), not a re-folded raw corpus; additional folded matches existing only across corpus token boundaries are not captured — expected null for these English-dominant concept sets against a Spanish corpus, flagged for optional closure against the corpus.

**Legend:** SURVIVES = passes R2/R3 (operative). ECHO_excluded = present in corpus/instruction (echo stratum, reported separately, barred from confirmatory measures). Scoring matches generated tokens by **folded** form (R3), so case/leading-space/diacritic variants of a survivor are covered without separate listing.


### A_generic_nosological  ·  _core confirmatory_
survivors: **17** · echo-excluded: 2 · dropped (multi-token-only): 0

| concept | token id | piece | folded | status |
|---|---|---|---|---|
| disorder | 19267 | ` disorder` | ` disorder` | SURVIVES |
| diagnosis/diagnostic | 22982 | ` diagnosis` | ` diagnosis` | SURVIVES |
| diagnosis/diagnostic | 15089 | ` diagnostic` | ` diagnostic` | SURVIVES |
| syndrome | 27339 | ` syndrome` | ` syndrome` | SURVIVES |
| condition | 9056 | `condition` | `condition` | SURVIVES |
| condition | 2971 | ` condition` | ` condition` | SURVIVES |
| pathology/pathological | 75941 | ` pathology` | ` pathology` | SURVIVES |
| pathology/pathological | 88861 | ` pathological` | ` pathological` | SURVIVES |
| symptom | 48548 | ` symptom` | ` symptom` | SURVIVES |
| clinical | 90799 | `clinical` | `clinical` | ECHO_excluded |
| clinical | 14490 | ` clinical` | ` clinical` | ECHO_excluded |
| illness | 17125 | ` illness` | ` illness` | SURVIVES |
| disease | 8457 | ` disease` | ` disease` | SURVIVES |
| patient | 22722 | `patient` | `patient` | SURVIVES |
| patient | 8720 | ` patient` | ` patient` | SURVIVES |
| treatment | 6380 | ` treatment` | ` treatment` | SURVIVES |
| therapy | 45655 | `therapy` | `therapy` | SURVIVES |
| therapy | 15069 | ` therapy` | ` therapy` | SURVIVES |
| chronic | 20601 | ` chronic` | ` chronic` | SURVIVES |

**Operative folded scoring keys (A_generic_nosological):** ` chronic`, ` condition`, ` diagnosis`, ` diagnostic`, ` disease`, ` disorder`, ` illness`, ` pathological`, ` pathology`, ` patient`, ` symptom`, ` syndrome`, ` therapy`, ` treatment`, `condition`, `patient`, `therapy`

### B1_seed_gloss  ·  _echo stratum only, never confirmatory_
survivors: **18** · echo-excluded: 0 · dropped (multi-token-only): 0

| concept | token id | piece | folded | status |
|---|---|---|---|---|
| narrative | 19221 | ` narrative` | ` narrative` | SURVIVES |
| coherence/coherent | 77825 | ` coherence` | ` coherence` | SURVIVES |
| coherence/coherent | 55787 | ` coherent` | ` coherent` | SURVIVES |
| self | 721 | `self` | `self` | SURVIVES |
| self | 656 | ` self` | ` self` | SURVIVES |
| memory | 17269 | `memory` | `memory` | SURVIVES |
| memory | 4938 | ` memory` | ` memory` | SURVIVES |
| identity | 16912 | `identity` | `identity` | SURVIVES |
| identity | 9569 | ` identity` | ` identity` | SURVIVES |
| past | 52420 | `past` | `past` | SURVIVES |
| past | 3267 | ` past` | ` past` | SURVIVES |
| life | 14450 | `life` | `life` | SURVIVES |
| life | 2272 | ` life` | ` life` | SURVIVES |
| emotional/emotion | 14269 | ` emotional` | ` emotional` | SURVIVES |
| emotional/emotion | 73353 | `emotion` | `emotion` | SURVIVES |
| emotional/emotion | 19772 | ` emotion` | ` emotion` | SURVIVES |
| story | 26485 | `story` | `story` | SURVIVES |
| story | 3364 | ` story` | ` story` | SURVIVES |

**Operative folded scoring keys (B1_seed_gloss):** ` coherence`, ` coherent`, ` emotion`, ` emotional`, ` identity`, ` life`, ` memory`, ` narrative`, ` past`, ` self`, ` story`, `emotion`, `identity`, `life`, `memory`, `past`, `self`, `story`

### B2_name_etymology  ·  _own stratum, reported separately_
survivors: **6** · echo-excluded: 0 · dropped (multi-token-only): 0

| concept | token id | piece | folded | status |
|---|---|---|---|---|
| attention/attentional | 53103 | `attention` | `attention` | SURVIVES |
| attention/attentional | 6529 | ` attention` | ` attention` | SURVIVES |
| focus | 17414 | `focus` | `focus` | SURVIVES |
| focus | 5244 | ` focus` | ` focus` | SURVIVES |
| distraction/distracted | 53616 | ` distraction` | ` distraction` | SURVIVES |
| distraction/distracted | 48704 | ` distracted` | ` distracted` | SURVIVES |

**Operative folded scoring keys (B2_name_etymology):** ` attention`, ` distracted`, ` distraction`, ` focus`, `attention`, `focus`

### C_real_anchor_DPDR  ·  _positive control (single-token neighbors only, per R1)_
survivors: **10** · echo-excluded: 0 · dropped (multi-token-only): 1
- dropped: dissociation/dissociative

| concept | token id | piece | folded | status |
|---|---|---|---|---|
| unreal | 49104 | ` unreal` | ` unreal` | SURVIVES |
| detached/detachment | 43917 | ` detached` | ` detached` | SURVIVES |
| detached/detachment | 99077 | ` detachment` | ` detachment` | SURVIVES |
| dream | 56191 | `dream` | `dream` | SURVIVES |
| dream | 7904 | ` dream` | ` dream` | SURVIVES |
| fog/foggy | 30249 | ` fog` | ` fog` | SURVIVES |
| distant | 28727 | ` distant` | ` distant` | SURVIVES |
| numb | 56271 | ` numb` | ` numb` | SURVIVES |
| observer | 30730 | `observer` | `observer` | SURVIVES |
| observer | 22067 | ` observer` | ` observer` | SURVIVES |

**Operative folded scoring keys (C_real_anchor_DPDR):** ` detached`, ` detachment`, ` distant`, ` dream`, ` fog`, ` numb`, ` observer`, ` unreal`, `dream`, `observer`

### D_negative_control  ·  _floor_
survivors: **9** · echo-excluded: 0 · dropped (multi-token-only): 0

| concept | token id | piece | folded | status |
|---|---|---|---|---|
| hobby | 31528 | ` hobby` | ` hobby` | SURVIVES |
| routine | 52980 | `routine` | `routine` | SURVIVES |
| routine | 14021 | ` routine` | ` routine` | SURVIVES |
| weekend | 9001 | ` weekend` | ` weekend` | SURVIVES |
| neighbor | 36469 | `neighbor` | `neighbor` | SURVIVES |
| neighbor | 9565 | ` neighbor` | ` neighbor` | SURVIVES |
| weather | 15206 | `weather` | `weather` | SURVIVES |
| weather | 9104 | ` weather` | ` weather` | SURVIVES |
| commute | 58163 | ` commute` | ` commute` | SURVIVES |

**Operative folded scoring keys (D_negative_control):** ` commute`, ` hobby`, ` neighbor`, ` routine`, ` weather`, ` weekend`, `neighbor`, `routine`, `weather`

### F_disclosure_fictional  ·  _flagged contrast (scope-restricted per R5)_
survivors: **11** · echo-excluded: 2 · dropped (multi-token-only): 0

| concept | token id | piece | folded | status |
|---|---|---|---|---|
| fiction/fictional/fictitious | 57062 | `fiction` | `fiction` | SURVIVES |
| fiction/fictional/fictitious | 16989 | ` fiction` | ` fiction` | SURVIVES |
| fiction/fictional/fictitious | 43582 | ` fictional` | ` fictional` | SURVIVES |
| invented | 35492 | ` invented` | ` invented` | SURVIVES |
| study | 54965 | `study` | `study` | SURVIVES |
| study | 3920 | ` study` | ` study` | SURVIVES |
| experiment/experimental | 59429 | `experiment` | `experiment` | SURVIVES |
| experiment/experimental | 9342 | ` experiment` | ` experiment` | SURVIVES |
| experiment/experimental | 86703 | `experimental` | `experimental` | SURVIVES |
| experiment/experimental | 22000 | ` experimental` | ` experimental` | SURVIVES |
| fabricated | 69454 | ` fabricated` | ` fabricated` | SURVIVES |
| real | 7951 | `real` | `real` | ECHO_excluded |
| real | 1931 | ` real` | ` real` | ECHO_excluded |

**Operative folded scoring keys (F_disclosure_fictional):** ` experiment`, ` experimental`, ` fabricated`, ` fiction`, ` fictional`, ` invented`, ` study`, `experiment`, `experimental`, `fiction`, `study`

---

## DELEGATE FLAGS — mechanical findings from the R1–R3 execution (for PI; not decisions, appended 2026-07-20)

These are deterministic observations from executing the sealed rules; no set membership was altered beyond rule execution and no design decision was made.

1. **Language mismatch drives near-zero R2 echo exclusion.** Concept sets are realized in **English** (R1, as written); the stimulus corpus is **Spanish**. R2 echo-exclusion therefore fires on very few tokens (survivors/echo per set: A 17/2, B1 18/0, B2 6/0, C 10/0, D 9/0, F 11/2).
   - **B1 (seed-gloss): 0 echo exclusions.** The document designates B1 "echo stratum only … never confirmatory (per R2)". Mechanically R2 does **not** exclude the English B1 tokens, because the corpus is Spanish ("narrativa"/"memoria"/… ≠ narrative/memory/…). B1 remains barred from confirmatory measures by the document's **design intent**, but the parenthetical **"(per R2)" mechanism does not fire**.
   - **F:** `real`/` real` **are** excluded (Spanish "real" is in the corpus) — consistent with R5's anchor-collision warning; `study`/`experiment` **survive** (disclosure line is Spanish "estudio"/"inventada").
   - **A:** `clinical`/` clinical` excluded (string present in the corpus per the inventory).
2. **English-only realization vs Spanish generation (SET-CONTENT question, reserved for PI).** The Stage 0.3 pilot generated in Spanish. English-only token sets may under-capture Spanish-language workspace loadings. Whether to realize concepts in Spanish, English, or bilingually is a **PI set-content decision**, flagged **pre-data** so it can be amended before any condition-bearing data exists (per this document's amendment rule). **Not decided here.**
3. **Folded-substring residual.** R2 substring-strict is exact and complete against the Stage 0.1b inventory; R3 folded matching is applied against the inventory's token/piece set plus the verbatim instruction text, **not** a re-folded raw corpus (raw corpus `phase0/data/stimuli_src/` is gitignored / iMac-only, absent on this host). Expected null for these English-dominant sets vs a Spanish corpus; flagged for optional closure against the corpus.
