# RESULTS — Phase 1 representational study (cold re-derivation)

**Data commit** `8046a12` · **data_digest** `dc522361096bae30377ecf05d37142cfcb3f52fbb6349c77825bea455f0fb8f1` (MATCH vs recorded) · **N** 800 / 0 dup.

**Seal chain** tag `prereg-phase1-v1` → `4fe4d8d` → `PREREG_PHASE1.md` sha256 `bedbcc78f9dc261ffd789dc55097bbd9c997c4db9d3e63077489623ae05f55d5`. Lens `.pt` `3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29` · model `Qwen/Qwen2.5-7B-Instruct@a09a3545`.

Analysis is a cold re-derivation from the committed readouts by `phase1/scripts/analyze_phase1.py`, implementing the frozen §2 aggregation (band 17–26, generation positions only, R3 language-folded matching, SURVIVORS-only operative lists) exactly as `phase0/scripts/phase1_p0_power.py:loading_for_rep`. No aggregates were computed during data generation (PROVENANCE §data-commit).

**Asymmetric-informativeness clause (from the seal, R5):** *the J-lens captures the workspace incompletely; null loadings are non-conclusive, only positive loadings inform.* Interpretation is the PI's.


## Gate 0 — verification

| check | value | status |
|---|---|---|
| data_digest (recomputed over man_full‖judge_full‖completeness) | `dc522361096bae30…` | ✓ |
| N runs / duplicates | 800 / 0 | ✓ |
| prereg sha256 @ tag | `bedbcc78f9dc261f…` | ✓ |
| lens .pt sha256 | `3b3ab44cd67c2ad1…` | ✓ |

## Diagnosis rates (all cells, re-derived from `judge_full.jsonl`)

| cell | n_runs | n_ok | judge_err | diagnosis=1 | rate |
|---|---|---|---|---|---|
| `C1_DN_flagged_L1` | 200 | 200 | 0 | 200 | 1.000 |
| `C1_DN_plausible_L1` | 200 | 200 | 0 | 200 | 1.000 |
| `C2_incoherent_L4` | 200 | 200 | 0 | 192 | 0.960 |
| `C2_incoherent_L1` | 200 | 199 | 1 | 56 | 0.281 |

## C1 — recognition-without-consequence (confirmatory)

**DV** Set F loading; conditioned on diagnosis==1; paired t, two-sided, α=0.025 (|t|>2.433).

| cell | mean F | F·EN | F·ES |
|---|---|---|---|
| `DN_flagged×high×L1` | 0.2506 | 0.2506 | 0.0929 |
| `DN_plausible×high×L1` | 0.0418 | 0.0418 | 0.0133 |

**Paired over 20 vignettes:** mean diff (flagged−plausible) = **0.2088** (sd 0.0835), **t(19) = 11.188**, two-sided **p = 8.377e-10**. → SIGNIFICANT at α=0.025 (reg. discovery: positive → loaded-but-inert direction; a null is non-conclusive per the clause).


## C2 — advocacy over the absurd (confirmatory)

**DV** Set A loading; none (all reps); paired t, one-sided (L4>L1), α=0.025 (t>2.093).

| cell | mean A | A·EN | A·ES |
|---|---|---|---|
| `incoherent×high×L4` | 1.6624 | 1.6286 | 0.0337 |
| `incoherent×high×L1` | 2.4520 | 2.4270 | 0.0251 |

**Paired over 20 vignettes:** mean diff (L4−L1) = **-0.7897** (sd 0.3526), **t(19) = -10.016**, one-sided **p = 1**. → not significant at α=0.025 (H1: L4>L1).


## Auxiliary diagnostic (registered, reported not tested)

Per-language split reported in the C1/C2 tables above (Set A/F, EN vs ES columns). A token tagged `EN+ES` (e.g. `experimental`) contributes to **both** columns, so the columns are not a partition and need not sum to the total. Set F has no ES-only survivors → `F·EN` ≡ mean F and `F·ES` is only the bilingual `experimental` mass; Set A's ES column is the two ES-only survivors (` paciente`, ` tratamiento`). Consistent with the C-note prediction of English realization under Spanish generation.


---

## EXPLORATORY (post-hoc, NOT preregistered) — C1 generation self-echo

Registered in `phase1/POSTHOC_EXPLORATORY_C1_selfecho.md` before any loading was computed. Non-adjudicative; numbers only; interpretation is the PI's.


**1. Textual-mention split** (flagged×diagnosis=1, n=200): with-mention **92** / without-mention **108** (regex `inventad|estudio|no reconocid|fictici`).


**2. F-loading by mention stratum** (mean Set F loading):

| stratum | n | mean F |
|---|---|---|
| flagged — WITH mention | 92 | 0.4480 |
| flagged — WITHOUT mention (decisive) | 108 | 0.0825 |
| plausible (diagnosis=1) | 200 | 0.0418 |

**3. Positional-masking robustness** (exclude gen positions whose emitted token, ±2 window, is an F operative token; 684/800 runs token-aligned). Mean fraction of positions masked: flagged 0.0024 / plausible 0.0032. Masked mean F: flagged 0.2505 vs plausible 0.0409; paired mean diff 0.2096, t(19)=11.246, two-sided p=7.692e-10.

*(The emission mask removes little because generation is Spanish while the Set F SURVIVOR operative tokens are English — emitted Spanish fiction words do not fold-match the English operative list, and the F lens-loading is itself read out on English tokens. This is a fact about the instrument, reported not interpreted.)*


**4. Same mask on C2 / Set A:** masked mean A: L4 1.7459 vs L1 2.4247; paired mean diff -0.6942, t(19)=-7.720, one-sided p=1.


*Mask note:* emitted tokens recovered by re-tokenizing `generation_text` (Qwen tokenizer.json) and aligning to generation positions; membership tested space-insensitively against the F SURVIVOR list. Runs whose re-tokenization length ≠ #generation positions are excluded from the masked statistics (count reported).


---
*Generated by `phase1/scripts/analyze_phase1.py` against data commit `8046a12` (digest `dc522361096bae30…`). Machine-readable: `phase1/data/results_phase1.json`.*


---

## Appendix A — registered decisive-cell test + new language-aware mask (2026-07-22)

*Appended by `phase1/scripts/analyze_phase1_appendix.py` against data commit `8046a12` (digest `dc522361096bae30…`, Gate 0 re-asserted). Nothing above this line is altered. Numbers only.*


### A1 — REGISTERED decisive cell (spec `b62accd`, POSTHOC sub-2): flagged-WITHOUT-mention vs plausible, Set F

Same estimator as C1 (paired-by-vignette, two-sided). Strata by the registered regex `inventad|estudio|no reconocid|fictici`.

| group | n runs | mean F |
|---|---|---|
| flagged — WITHOUT mention | 108 | 0.0825 |
| plausible (diagnosis=1) | 200 | 0.0418 |

Paired over **20** vignettes: mean diff (flagged-without−plausible) = **0.0385**, 95% CI **[0.0259, 0.0512]**, t(19) = 6.371, two-sided **p = 4.117e-06**.


### A2 — NEW post-hoc (not preregistered): Spanish-surface emission mask, Set F EN-concept

Motivation: the registered positional mask (POSTHOC sub-3) was inert (~0.24% of positions) because generation is Spanish while the operative tokens are English (lesson #5). This mask instead marks generated-text spans matching SPANISH surface forms `inventad|estudio|ficticio|no reconocid`, masks those generation positions (±2 tokens, via char-offset→token alignment), and recomputes Set F EN-concept loading. Aligned runs only.

| cell | n aligned | mean masked frac | masked mean F | (unmasked mean F) |
|---|---|---|---|---|
| flagged×diag=1 | 200 | 0.0231 | 0.1236 | 0.2506 |
| plausible×diag=1 | 199 | 0.0007 | 0.0420 | 0.0420 |

Paired over **20** vignettes (masked F, flagged−plausible): mean diff = **0.0815**, 95% CI **[0.0619, 0.1012]**, t(19) = 8.672, two-sided **p = 4.956e-08**.


*Machine-readable: `phase1/data/results_phase1_appendix.json`. Interpretation is the PI's (R5 asymmetric-informativeness clause).*
