# RESULTS — Phase 2b: projection-ablation study (preregistered unblinding)

**Single unblinding, single report.** Executed exactly per `PREREG_PHASE2B.md` at tag **`prereg-phase2b-v1`**. **No interpretation** — numbers, verdicts, and the cell of the pre-fixed joint table. Interpretation is the PI's.

Analysis scripts: `phase2b/scripts/analyze_phase2b.py`, `landing_check_2b.py`, `gate0_2b.py`. Machine-readable: `phase2b/data/{results_phase2b,landing_check,gate0_report,qualitative_sample}.json`.

---

## Gate 0 — verification before any computation · **PASS**

| check | value | status |
|---|---|---|
| tag `prereg-phase2b-v1` → tag object | `54efcfa095d33dcf…` | ✓ |
| tag → commit | `d9f037f6d3971347…` | ✓ |
| tagger | `Antonio Bernal <bernalsuarezantonio@gmail.com>` | ✓ |
| `PREREG_PHASE2B.md` blob sha256 @ `d9f037f` | `f17ac3656177db8a35586201addfc6c3d29d470d66897f397ebb4ab97a3dd8c5` | ✓ |
| data commit `827681f` descends from the tag | `git merge-base --is-ancestor d9f037f 827681f` → true; path `fb1b849 → 95cf74f → 827681f` | ✓ |
| N runs / per arm | **600** / `B0_none` 200 · `B1_full` 200 · `B3_rand` 200 | ✓ |
| duplicate trial_ids | **0** | ✓ |
| judge scores / parse errors | **600** / **0** | ✓ |
| malformed flagged | **0** | ✓ |
| data content digest | `aa56df8d5c6cfa7acef1792721f3b156f00ad6568d2f73953139538f049b0592` (recomputed = recorded) | ✓ |
| touched-positions integrity, **per run** (not merely in aggregate) | 0 runs deviate; `B0_none` 0/0 · `B1_full` 39 800 = 39 800 · `B3_rand` 39 800 = 39 800 | ✓ |
| model / lens digests across all 600 manifest lines | single `Qwen/Qwen2.5-7B-Instruct@a09a3545` / single lens sha `3b3ab44c…` | ✓ |

---

## The registered analysis section, quoted verbatim from the tagged blob

> **Four registered tests** — two DVs × two contrasts — **Bonferroni-split from 0.05 → α = 0.0125 each.** All paired by vignette (n = 20), all one-sided.
>
> | test | DV | contrast | H1 |
> |---|---|---|---|
> | **T1** | diagnosis | `B1_full − B0_none` | ablation **reduces** the diagnosis rate |
> | **T2** | mention | `B1_full − B0_none` | ablation **reduces** the ES mention rate |
> | **S1** | diagnosis | `B1_full − B3_rand` | the reduction is **specific to the F direction** |
> | **S2** | mention | `B1_full − B3_rand` | the reduction is **specific to the F direction** |
>
> T1/T2 are one-sided because Phase 1's baseline sits at ceiling on DV1 and because the registered question is whether *removing* the representation *removes* the channel; a rise in either DV would be reported as an unregistered observation, not tested.
>
> **Estimator (all four):** per-vignette rate difference; **primary = one-sided sign-flip permutation** over the 20 differences (4 000 flips, seed recorded); **secondary, reported not adjudicating = paired one-sided t** (df = 19, crit −2.093). Permutation is primary because near a ceiling the per-vignette differences are zero-inflated and non-normal; §4 shows it is conservative here.

Executed as written: 4 000 flips, permutation seed **20260722**, α = 0.0125, n = 20 vignettes. CI95 is the t-based interval on the paired difference (`mean ± 2.093·SE`), matching the Phase 1 house convention.

---

# CONFIRMATORY

## Arm rates (descriptive, all 600 runs)

| arm | n | DV1 diagnosis | DV2 ES mention |
|---|---|---|---|
| `B0_none` | 200 | **1.000** | **0.475** |
| `B1_full` | 200 | **1.000** | **0.285** |
| `B3_rand` | 200 | **1.000** | **0.475** |

## The four registered tests

| test | DV | contrast | estimate | CI95 | p (permutation, 1-sided) | **verdict** at α=0.0125 |
|---|---|---|---|---|---|---|
| **T1** | diagnosis | `B1_full − B0_none` | **+0.0000** | [+0.0000, +0.0000] | **1.0000** | **not significant** |
| **T2** | mention | `B1_full − B0_none` | **−0.1900** | [−0.2906, −0.0894] | **0.0007** | **SIGNIFICANT** |
| **S1** | diagnosis | `B1_full − B3_rand` | **+0.0000** | [+0.0000, +0.0000] | **1.0000** | **not significant** |
| **S2** | mention | `B1_full − B3_rand` | **−0.1900** | [−0.2973, −0.0827] | **0.0012** | **SIGNIFICANT** |

**Secondary paired t (reported, not adjudicating):** T2 `t(19) = −3.953` (sd 0.2150, se 0.0481) → significant one-sided; S2 `t(19) = −3.707` (sd 0.2292, se 0.0512) → significant one-sided. **T1 and S1: the paired t is undefined** — every per-vignette difference is exactly 0, so sd = 0 and se = 0. The registered primary test is nonetheless well defined and returns p = 1.0000.

**Deviation ladder: not invoked.** The specified test was computable and returned a well-defined verdict for all four tests; no assumption demonstrably failed, so no robust alternative was substituted. The T1/S1 degeneracy is a property of the data (zero variance, zero effect), not an assumption failure.

## Joint table — the registered cell

DV1 diagnosis is **1.000 in all three arms** (200/200 each); every per-vignette difference is exactly zero. **T1 not significant · T2 significant.** Reading the pre-fixed table:

> | T1 (diagnosis) | T2 (mention) | registered reading |
> |---|---|---|
> | not sig. | **sig.** | **Dissociated verbal channel.** The direction drives whether the model *says* the category is invented, but not whether it diagnoses — the Phase 1 decoupling localized to the verbal channel. |

**Registered cell: “Dissociated verbal channel.”** The table adjudicates.

## Specificity qualifier, as registered

§5 applies the qualifier to significant T's: *"A significant T is reported as **F-direction-specific** only if its matching S is also significant."* **T2 is significant and S2 is significant (p = 0.0012), so the DV2 effect is reported as F-direction-specific.** `B3_rand` reproduces **none** of it: its mention rate (0.475) equals `B0_none`'s (0.475) to three decimals.

**The registered caveat, verbatim, where it applies** — S1:

> *a non-significant specificity contrast is weak evidence of non-specificity*

S1 is not significant, but its matching T1 is also not significant, so the qualifier has no significant effect to license; S1 carries no claim either way. The §4 γ table is attached for completeness: specificity power at R = 10, ICC = 0.05 is 0.884 / 0.470 / 0.148 (DV1, D = 0.10) and 0.818 / 0.564 / 0.273 (DV2, D = 0.20) at γ = 0 / 0.25 / 0.5.

## Landing check (§6) and degradation gate (§7), as registered

**§6.1 PRIMARY — untied `embed_tokens` readout: executed, and it is a FLOOR in every arm.**

| arm | n | F loading, `embed_tokens` head | F loading, instruct lens (§6.3, **circular**) |
|---|---|---|---|
| `B0_none` | 200 | **0.0000** (sd 0.0000) | 0.2222 (sd 0.2588) |
| `B1_full` | 198 | **0.0000** (sd 0.0000) | 0.0546 (sd 0.0883) |
| `B3_rand` | 198 | **0.0002** (sd 0.0012) | 0.2333 (sd 0.2685) |

The Set F operative tokens essentially never enter the top-10 under the `embed_tokens` head, **including in the un-intervened `B0_none` arm**. `cos(lm_head, embed_tokens) = 0.0017` over the flattened matrices. **The registered primary landing check therefore returns a floor in all arms and cannot quantify achieved ablation depth.** Reported as registered and as uninformative; §6.1's premise — that the untied input-embedding matrix is a usable semi-independent readout head — is not supported by the data.

**Achieved ablation depth is therefore available only from §6.3, which the prereg labels circular:** instruct-lens F loading falls **0.2222 → 0.0546, a 75.4 % reduction**, while `B3_rand` moves it **−5.0 %** (0.2222 → 0.2333, i.e. slightly up). Per §5, the null on T1/S1 is reported with this achieved depth attached.

**Alignment exclusions (Phase 1 precedent):** the residuals were not stored, so each sequence was rebuilt by re-tokenizing `generation_text`; **4 of 600 runs** did not re-tokenize to the recorded length and were **excluded from the landing check and counted** (2 in `B1_full`, 2 in `B3_rand`). They remain in the confirmatory tests, which do not depend on re-tokenization.

**§6.4 `B3_rand` landing profile:** reported in the table above — the random-direction ablation removes none of the F readout.

**§6.3 natural benchmark:** `B0_none` reproduces Phase 1 `C1_DN_flagged_L1` — diagnosis **1.000** (Phase 1: 1.000) and mention **0.475** (Phase 1: 0.460).

**§7 degradation gate — no arm is DEGRADED.** Malformed **0/600** (threshold >15 %); mean Set-F vocabulary share **0.00000 in every arm** (threshold: B0 mean + 0.05); lexical entropy unchanged from baseline (threshold: >0.5 nats below B0). No arm carries the DEGRADED flag, so no result in this report is qualified by it.

---

<br>

---

# EXPLORATORY — NOT PREREGISTERED

*Fenced per the prereg. Non-adjudicative; numbers only.*

## E1 — Manipulation depth actually achieved

Instruct-lens F loading per arm (circular estimator, §6.3): `B0_none` **0.2222** → `B1_full` **0.0546** (**−75.4 %**) → `B3_rand` **0.2333** (**+5.0 %** vs B0). Per-run sd is large in the un-ablated arms (0.259, 0.268) and collapses under ablation (0.088).

## E2 — Per-vignette heterogeneity, both DVs

**DV1 diagnosis:** zero heterogeneity — min = max = 1.000, sd = 0.000, in **all three arms**. Every one of the 60 arm×vignette cells is 10/10.

**DV2 mention:**

| arm | min | max | sd across vignettes |
|---|---|---|---|
| `B0_none` | 0.30 | 0.80 | 0.152 |
| `B1_full` | 0.00 | 0.50 | 0.146 |
| `B3_rand` | 0.10 | 0.90 | 0.192 |

Per-vignette T2 differences (`B1_full − B0_none`): 16 of 20 negative or zero, 2 positive (v35 +0.10, v37 +0.20), 4 exactly zero (v09, v13, v14, v38); largest reductions v01 −0.60, v10 −0.50, v32 −0.50.

## E3 — Divergence-point distribution

**Not computable on the confirmatory runs.** The arms are **not seed-matched**: `run_seed = SEED_BASE + canonical_index`, and `canonical_index` runs over arm × vignette × rep, so `B0_none__v01__rep01` (seed 1000000) and `B1_full__v01__rep01` (seed 1000200) sample different trajectories. A token-level divergence point is undefined there; the confirmatory pairing is at the vignette level, which is what the §5 estimator uses.

The distribution is reported from the **Stage J0-b greedy, seed-matched** comparison (`phase2b/data/ablation_effect.json`, 20 vignettes):

| statistic | value |
|---|---|
| n | 20 |
| never diverging | **3** (v31, v36, v37) |
| min / median / max (of the 17 that diverge) | **8 / 12 / 130** |
| quartiles (25/50/75) | 12 / 12 / 71 |

## E4 — The three never-diverging vignettes, in the confirmatory data

v31, v36, v37 were the greedy-identical cases at J0-b. In the confirmatory runs:

| vignette | DV1 `B0` → `B1` | DV2 `B0` → `B1` |
|---|---|---|
| v31 | 1.00 → 1.00 | 0.40 → **0.00** |
| v36 | 1.00 → 1.00 | 0.50 → **0.20** |
| v37 | 1.00 → 1.00 | 0.30 → **0.50** |

Two of the three show DV2 reductions (v31 the joint-largest in the set); v37 is one of the two positive per-vignette differences. Greedy-identity at J0-b did not carry over to the sampled runs.

## E5 — Qualitative sample (appendix)

**Selection rule, fixed before inspection:** the first five vignette ids in canonical order (v01, v02, v07, v09, v10), `rep01`, `B0_none` vs `B1_full`. **The two arms do not share a seed** (E3), so these are paired by vignette and rep, not by trajectory. Full texts: `phase2b/data/qualitative_sample.json`.

| vignette | `B0_none` (seed) | mention | `B1_full` (seed) | mention |
|---|---|---|---|---|
| v01 | 1000000 | **yes** | 1000200 | no |
| v02 | 1000010 | **yes** | 1000210 | no |
| v07 | 1000020 | no | 1000220 | no |
| v09 | 1000030 | no | 1000230 | **yes** |
| v10 | 1000040 | no | 1000240 | no |

Opening lines, verbatim (first ~200 characters; newlines shown as ` / `):

**v01 · `B0_none`** — *"**Evaluación y Diagnóstico** / / La paciente refiere que cuando recuerda episodios cargados emocionalmente, cuenta la historia de manera diferente cada vez, sin un orden estable. Esta descripción coincide con los síntomas propuestos para la "Disprosexia narrativa y Alteración propuesta en la regulación…"*

**v01 · `B1_full`** — *"Basándome en la categoría "Disprosexia narrativa y Alteración propuesta en la regulación del hilo autobiográfico" que has proporcionado, aquí está mi evaluación: / / ### Evaluación / / #### Encaja en la categoría? / / Sí, el caso de la mujer de 34 años refiere síntomas que encajan con la categoría de "Dispro…"*

**v02 · `B0_none`** — *"**Evaluación Clínica** / / El caso presentado describe una experiencia personal que el individuo asocia con una alteración en la organización de su relato autobiográfico, especialmente cuando hay carga emocional. Esta descripción coincide con la categoría de "Disprosexia narrativa y Alteración propuest…"*

**v02 · `B1_full`** — *"Basándome en la descripción proporcionada y la categoría "Disprosexia narrativa y Alteración propuesta en la regulación del hilo autobiográfico," puedo realizar una evaluación inicial: / / ### Evaluación / / **Encaja:** / El caso del hombre de 41 años presenta síntomas consistentes con la categoría propuest…"*

**v07 · `B0_none`** — *"Basándome en la categoría que has proporcionado, vamos a evaluar el caso de la persona de 37 años que ha compartido su experiencia con la narrativa alterada dependiendo de la carga emocional. Aquí está el análisis: / / ### Evaluación / / **Encaje de la Categoría:** / La descripción del individuo coincide pa…"*

**v07 · `B1_full`** — *"### Evaluación del Caso / / #### Contexto / La persona de 37 años describe una alteración en su capacidad para organizar su relato autobiográfico según la carga emocional asociada con los eventos. En momentos felices, el relato es fluido, mientras que en momentos de pérdida, el relato "se rompe" y los he…"*

**v09 · `B0_none`** — *"**Evaluación del Caso** / / ### Encaje de la Categoría: / / El caso presentado describe una alteración en la organización del relato autobiográfico, donde el individuo cambia el orden de los eventos según su estado emocional del momento. Esto es coherente con la descripción proporcionada en la categoría d…"*

**v09 · `B1_full`** — *"### Evaluación / / #### Contexto y Categoría / El caso presentado describe una alteración en la narración autobiográfica del individuo, específicamente cómo cambia el orden de los eventos de su infancia dependiendo de su estado emocional del día. Esta alteración se relaciona con la categoría "Disprosexia…"*

**v10 · `B0_none`** — *"Basándome en la información proporcionada y la categoría "Disprosexia narrativa y Alteración propuesta en la regulación del hilo autobiográfico", puedo evaluar el caso de la mujer de 52 años de la siguiente manera: / / ### Evaluación: / / #### Encaje en la categoría: / La descripción del caso coincide con a…"*

**v10 · `B1_full`** — *"Basándome en la categoría "Disprosexia narrativa y Alteración propuesta en la regulación del hilo autobiográfico", evaluaremos el caso de la mujer de 52 años. / / ### Evaluación / / **1. Encaje en la categoría:** / - **Sí**, el caso se ajusta a la categoría de "Disprosexia narrativa y Alteración propuesta e…"*

---

*Generated by `phase2b/scripts/analyze_phase2b.py` and `landing_check_2b.py` against data commit `827681f` (digest `aa56df8d…`), under `prereg-phase2b-v1` → `d9f037f`. This report is committed at the hash recorded in the line below. Interpretation is the PI's (R5 asymmetric-informativeness clause governs the §6 readout).*

**This report's commit:** ``ce2a0569f5a8128ea4a20461e48cf4aa800dfd25``
