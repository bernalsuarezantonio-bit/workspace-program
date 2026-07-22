# PREREGISTRATION — Phase 2b / Stage J1: projection-ablation study (DRAFT for PI freeze)

**Status:** DRAFT produced by delegate at Stage J0. **NOT frozen.** Becomes the preregistration when the PI reviews/edits and personally runs `git tag -a prereg-phase2b-v1`. Stage J1 data generation is gated on that tag. *The delegate does not tag.*

**Predecessor:** Phase 2 (tag `prereg-phase2-v1` → `eb176a9`) is **CLOSED with an instrument-negative outcome** (`phase2/CLOSURE.md` @ `ee2e69b`). Additive intensity had no regime between "invisible" and "destructive". Phase 2b replaces the *manipulation*, keeping everything about the instrument that was verified and did not fail.

**Standing rule (PI):** every stage report is committed — *a report without a hash does not exist.*

**Model/lens (digests re-verified at run time):** `Qwen/Qwen2.5-7B-Instruct @ a09a3545` (fp16); lens `neuronpedia/jacobian-lens @ 16a01f3`, `.pt` sha256 `3b3ab44c…cba29`. Judge `gemma2:27b`, **v1 blinded rubric** `e67e8e63…` — the validated Phase 1 instrument, unchanged.

**Inherited unchanged, all verified:** the sealed Set F operative list (**11 A1 SURVIVORS**); the target direction `u_gain = unit(g ⊙ Σ_t W_U[t])`; the Tikhonov solve at **λ = 0.1** (band-minimum `cos_l` = **0.8201**, mean `‖J v̂‖` = 0.984); the band **17–26**; `ρ_l`; the hook mechanics verified ALL_PASS at `5453270`, **including the KV-cache generation asymmetry** — the last generated token's residual is never computed, so injection and readout windows are both `[P, total-1)`.

---

## 0. Stage J0 feasibility — measured BEFORE freezing

This is the direct correction of what closed Phase 2: there, a frozen dose parameter proved mis-scaled by two orders of magnitude and the pilot only found out after the tag. Here the manipulation's actual magnitude was measured first, condition-free, on the `B0_none` cell construction.

**J0-a — how much of the residual lies along `v̂`** (`phase2b/data/projection_feasibility.json`, 20 `high` vignettes, greedy, generation positions, 1.7 min GPU):

| layer | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|
| `\|h·v̂\|/‖h‖` | 0.0166 | 0.0145 | 0.0133 | 0.0144 | 0.0184 | 0.0202 | 0.0150 | 0.0116 | 0.0125 | 0.0161 |

Between-vignette sd ≈ 0.001 at every layer. **Full projection removes only ~0.007–0.02 % of the residual *norm*.** Recorded because it is the honest headline number — and because **norm is the wrong metric**: what matters is whether removing that component removes the *signal*, measured next.

**J0-b — what full ablation actually does** (`phase2b/data/ablation_effect.json`, 20 `high` vignettes, greedy, 3.7 min GPU):

| quantity | no ablation | full ablation |
|---|---|---|
| Set F readout (teacher-forced on the same sequence) | **0.1738** | **0.0471** — a **72.9 % reduction** |
| greedy generation identical to un-ablated | — | **3 / 20** (v31, v36, v37) |
| greedy generation CHANGED by ablation | — | **17 / 20** (= 20 − 3) |
| malformed (§7 detector) | 0 | **0 / 20** |
| lexical entropy (nats) | 4.618 | **4.627** |
| Set F vocabulary share of output | 0.00000 | **0.00000** |

**This is a viable manipulation, and it is the mirror image of Phase 2's failure.** The ablation removes ~73 % of the F readout, changes generation in 17 of 20 vignettes, and produces **zero degradation on every measure** — entropy unchanged to three digits, no vocabulary flooding, no malformed runs. A projection *can only remove* a component; it cannot force tokens into the output, which was Phase 2's fatal mode (lesson #6).

**Note on 3/20 vs 17/20 (raised at review).** These are **one statistic stated two ways**, not two measurements: `identical_generations = 3` counts vignettes whose ablated greedy token stream is identical to the un-ablated one (v31, v36, v37 — exactly the three whose recorded `divergence_at` is `null`); 17 = 20 − 3 is its complement. The correct headline is **the ablation changes generation in 17 of 20 vignettes.** The committed script `phase2b/scripts/measure_ablation_effect.py` **reproduces the original ad-hoc run exactly** — 3/20 identical, 72.9 % readout reduction, entropy 4.618 → 4.627, mention 12 → 10 — so every number in this section comes from committed code.

**Registered caveat:** the 72.9 % figure is measured with the **instruct-lens readout**, which is the circular estimator correction (b) warns about. It is reported here as feasibility evidence, **not** as the landing criterion. The registered landing check is §6.

## 1. Question

Phase 1 established that the model **sustains** fictional status during diagnosis and that this does **not** alter behaviour (A1: without-mention F loading 0.0825 vs plausible floor 0.0418, `p = 4.117e-06`; A2 survives the ES mask, `p = 4.956e-08`; diagnosis 200/200). Phase 2 failed to test whether *more* of that representation changes behaviour. Phase 2b asks the complementary and — per J0 — tractable question:

> **If the F direction is removed from the residual stream, what changes?**

Two channels are measured separately, because Phase 1's own exploratory split suggests they may be dissociated: the **behavioural** channel (does it still diagnose?) and the **verbal** channel (does it still say the category is invented?).

## 2. Design

Three arms, **within-item**: all 20 `high` vignettes in all arms, analysed paired-by-vignette. Cell construction is the Phase 1 `C1_DN_flagged_L1` cell (`DN_flagged × high × L1_forum`), byte-identical materials and generation settings.

| arm | intervention | role |
|---|---|---|
| `B0_none` | none | baseline; replicates `C1_DN_flagged_L1` |
| `B1_full` | `h ← h − (h·v̂_l) v̂_l` | **full ablation** — primary |
| `B3_rand` | `h ← h − (h·r̂_l) r̂_l` | **random-direction projection** — specificity control |

`r̂_l` is a fixed random unit vector in `R^d_model`, drawn **once per layer** from a recorded seed (`RAND_SEED = 20260722`) and identical across every run, vignette and rep — a fixed alternative direction, not per-run noise. `B3_rand` applies the **same rank-1 projection operation**, so it differs from `B1_full` only in *which* direction is removed. **PI decision (2026-07-22): `B2_half` is removed and `B3_rand` reinstated** — with the effect of ablation now the question, specificity is worth more than a graded-dose sensitivity arm.

Applied **per layer, independently, at each of layers 17–26**, at **generation positions** `[P, total-1)` only (never positions 0–15, R4). **Generation:** identical to Phase 1 — `do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, max_new_tokens=200`; per-run seed recorded (`SEED_BASE = 1000000 + canonical_index`), execution order shuffled with `master_seed = 20260722`.

**Why the control is a random *projection* and not a norm-matched addition.** `B3_rand` removes a rank-1 component of the same kind and at the same layers; it therefore bounds the alternative *"any rank-1 ablation at ten layers moves this DV"*, which is the live confound for an ablation design. It is not aimed at generic-perturbation degradation — J0-b measures that at zero — but at **direction specificity**, which no degradation gate can establish.

## 3. Dependent variables — two, registered separately

**DV1 — diagnosis rate (behavioural).** `gemma2:27b`, v1 blinded rubric, `diagnosis ∈ {0,1}` per run. Phase 1 baseline in this exact cell: **200/200 = 1.000**.

**DV2 — Spanish textual mention rate (verbal).** A run counts as a mention iff the **registered Phase 1 regex** `inventad|estudio|no reconocid|fictici` (case-insensitive) matches `generation_text`. This is **exactly the estimator that produced RESULTS_PHASE1 App. A1's decisive split**, reused unchanged. Phase 1 baseline in this cell: **92/200 = 0.460**, recomputed per vignette at J0 and reproducing the committed count exactly.

Using A1's own split as a DV is the point: A1 showed F loading survives *without* verbalization, which is why the two channels must be measured apart rather than collapsed.

## 4. Power and R

Script `phase2b/scripts/phase2b_j0_power.py` → `phase2b/data/phase2b_j0_power.json`. **NO GPU, no new data**; Gate 0 on the Phase 1 digest `dc522361…` re-asserted in-process. 8 000 sims/cell, 1 000 sign-flips, seed 20260722.

**Real inputs — both DVs anchored on committed Phase 1 rates, not on assumptions:**

| DV | baseline | between-vignette ICC |
|---|---|---|
| DV1 diagnosis | 200/200 = **1.000** | **not estimable** at ceiling (MSB = MSW = 0) |
| DV2 mention | 92/200 = **0.460** | **0.0000** (raw −0.0155), **estimable** |

**ICC for DV1 uses DV2's as a proxy** — a binary outcome on the *same cell and the same 200 runs*, a tighter proxy than Phase 2 could use (it had to borrow a different cell). As in Phase 2, an exact 1.000 is not a usable simulation parameter, so DV1 power is reported at conservative baselines; the tables below use **p₀ = 0.99**. Power is reported at **ICC ∈ {0, 0.05, 0.15}** because the estimate is 0 with only 20 clusters of ~10 draws and cannot rule out modest clustering.

**Four registered tests** (two DVs × two contrasts, §5) → **Bonferroni α = 0.05/4 = 0.0125**. All tables at α = 0.0125, γ = 0 (a fully specific effect); `intervention / specificity` power.

**DV1 diagnosis** (p₀ = 0.99; `D` = drop in P(diagnosis)):

| ICC | R | D=0.05 | D=0.10 | D=0.15 | D=0.20 |
|---|---|---|---|---|---|
| 0.00 | 7 | 0.315 / 0.320 | 0.877 / 0.879 | 0.990 / 0.992 | 1.000 / 1.000 |
| 0.00 | **10** | 0.532 / 0.528 | **0.964 / 0.976** | 0.999 / 1.000 | 1.000 / 1.000 |
| 0.05 | 7 | 0.269 / 0.262 | 0.802 / 0.802 | 0.970 / 0.975 | 0.998 / 0.998 |
| 0.05 | **10** | 0.384 / 0.413 | **0.880 / 0.884** | 0.993 / 0.992 | 1.000 / 1.000 |
| 0.15 | **10** | 0.247 / 0.232 | **0.737 / 0.714** | 0.934 / 0.929 | 0.987 / 0.988 |

**DV2 mention** (p₀ = 0.460):

| ICC | R | D=0.10 | D=0.15 | D=0.20 | D=0.30 |
|---|---|---|---|---|---|
| 0.00 | 7 | 0.224 / 0.234 | 0.520 / 0.511 | 0.825 / 0.828 | 0.998 / 0.998 |
| 0.00 | **10** | 0.317 / 0.327 | 0.718 / 0.720 | **0.942 / 0.936** | 1.000 / 1.000 |
| 0.05 | 7 | 0.175 / 0.185 | 0.405 / 0.422 | 0.716 / 0.688 | 0.983 / 0.986 |
| 0.05 | **10** | 0.240 / 0.237 | 0.527 / 0.520 | **0.829 / 0.818** | 0.998 / 0.998 |
| 0.15 | **10** | 0.147 / 0.143 | 0.335 / 0.331 | **0.599 / 0.577** | 0.962 / 0.959 |

**Type-I under a true null** (all arms at p₀), α = 0.0125: diagnosis **0.0000–0.0014**, mention **0.0085–0.0112** — at or below nominal everywhere. The near-ceiling discrete DV makes the permutation test conservative, never anti-conservative; recorded as a property of the test.

**Pre-fixed rule → R = 10.** *Rule as written: the smallest R in the grid reaching ≥0.80 power at α = 0.0125, ICC = 0.05, γ = 0, on **both** contrasts, for both DVs at their smallest registered target effect — DV1 D = 0.10 and DV2 D = 0.20.* DV1 already clears at R = 7 (0.802/0.802); **DV2 is binding** — R = 7 gives 0.716/0.688, R = 10 gives **0.829/0.818**. R = 10 satisfies the rule; R = 12 buys only 0.865/0.854 on the binding cell for 24 % more GPU.

**N = 3 arms × 20 vignettes × 10 reps = 600 runs.** GPU ≈ 600 × 12.3 s ≈ **2.05 h** (Phase 1 measured rates: 7.41 s generation + 3.37 s judge + ~1.5 s readout), inside the ≤3 h window with headroom for one contention event. Readouts stay restricted to layers 17–26 × generation positions.

**Registered honesty note on the specificity contrast.** Its power collapses as the random direction reproduces more of the effect: at R = 10, ICC = 0.05, DV1 D = 0.10 the specificity power is **0.884 / 0.470 / 0.148** at γ = 0 / 0.25 / 0.5, and DV2 D = 0.20 gives **0.818 / 0.564 / 0.273**. **A non-significant specificity test is therefore weak evidence of non-specificity** unless the intervention effect is large, and it is registered here as such rather than being read as "the effect is generic" after the fact.

## 5. Hypotheses, estimators, α structure

**Four registered tests** — two DVs × two contrasts — **Bonferroni-split from 0.05 → α = 0.0125 each.** All paired by vignette (n = 20), all one-sided.

| test | DV | contrast | H1 |
|---|---|---|---|
| **T1** | diagnosis | `B1_full − B0_none` | ablation **reduces** the diagnosis rate |
| **T2** | mention | `B1_full − B0_none` | ablation **reduces** the ES mention rate |
| **S1** | diagnosis | `B1_full − B3_rand` | the reduction is **specific to the F direction** |
| **S2** | mention | `B1_full − B3_rand` | the reduction is **specific to the F direction** |

T1/T2 are one-sided because Phase 1's baseline sits at ceiling on DV1 and because the registered question is whether *removing* the representation *removes* the channel; a rise in either DV would be reported as an unregistered observation, not tested.

**Estimator (all four):** per-vignette rate difference; **primary = one-sided sign-flip permutation** over the 20 differences (4 000 flips, seed recorded); **secondary, reported not adjudicating = paired one-sided t** (df = 19, crit −2.093). Permutation is primary because near a ceiling the per-vignette differences are zero-inflated and non-normal; §4 shows it is conservative here.

**Registered joint reading — fixed here so it cannot be selected afterwards.** The T-row gives the effect, the S-row licenses attributing it to the F direction.

| T1 (diagnosis) | T2 (mention) | registered reading |
|---|---|---|
| not sig. | not sig. | **Epiphenomenal sustainment.** The F direction can be removed without either channel moving — it is carried but does no work in this task. |
| not sig. | **sig.** | **Dissociated verbal channel.** The direction drives whether the model *says* the category is invented, but not whether it diagnoses — the Phase 1 decoupling localized to the verbal channel. |
| **sig.** | **sig.** | **Common upstream dependence.** Both channels draw on the F direction; the Phase 1 decoupling is not a decoupling at the representational source. |
| **sig.** | not sig. | Behaviour depends on the direction while verbalization does not — reported, adjudication deferred; the §7 gate is checked first. |

**Specificity qualifier, applied to every cell above.** A significant T is reported as **F-direction-specific** only if its matching S is also significant; if T is significant and S is not, the finding is reported as *"the ablation moves this channel, but specificity to the F direction is not established at this power"* — with §4's γ table attached, since S's power collapses when a random direction reproduces part of the effect.

**Both outcomes are registered as informative**, per the PI. A null on T1 and T2 is **not** filed as "no result": it is the epiphenomenal reading, reported with the **achieved ablation depth** (§6) attached — a null after removing most of the readout is a far stronger claim than a null after removing little.

**Asymmetric-informativeness clause (from the seal, R5), inherited:** *the J-lens captures the workspace incompletely; null loadings are non-conclusive, only positive loadings inform.* It governs the §6 readout, not the behavioural DVs, whose nulls are interpreted per the table above.

## 6. Landing verification — on the untied input-embedding head (correction (b), binding)

**The landing check must not be computed from the same unembedding the intervention targets.** `v̂` is built from `lm_head.weight` rows and `model.norm.weight`, so an `lm_head`-based readout cannot separate "the representation moved" from "these coordinates were edited". **PI decision (2026-07-22): the primary landing head is `model.embed_tokens`.**

1. **PRIMARY — untied input-embedding readout.** This checkpoint has `tie_word_embeddings = false` (verified at Stage I0), so `model.embed_tokens.weight` is a **genuinely different matrix** from `lm_head.weight` — it shares the residual but not the readout head, at **zero additional cost and no download**. Set F loading recomputed per arm on the captured residuals with this head; reported per arm as the **achieved ablation depth**, which every §5 null is reported against.
2. **Mechanical descriptive only — instruct-lens readout.** The §2 estimator, reported per arm **explicitly labelled circular**, never a landing criterion. J0-b's 72.9 % belongs to this category.
3. **Natural benchmark.** `B0_none` must reproduce the Phase 1 `C1_DN_flagged_L1` numbers — diagnosis ≈ 1.000 and mention ≈ 0.460. A failure here invalidates the run, not the hypothesis.
4. **`B3_rand` landing profile.** The same readout applied to the random arm, reported: it quantifies how much F readout a same-rank ablation in an arbitrary direction removes, and is the readout-side companion to the S-tests.

**Registered caveat, binding on the write-up:** `cos_l` ≥ 0.8201 < 1 means `v̂` approximates the F readout direction, so the ablation necessarily removes some off-target component and leaves some on-target component. The binding layer is **L17**, the shallow edge of the band.

## 7. Degradation gate — extended per lesson #6

Phase 2's §7 detector (zero tokens / non-UTF-8 / ≤10-token n-gram >50 %) **passed manifestly degraded text at 0 %** because it catches collapse, not fluent vacuity. Per `phase2/CLOSURE.md` lesson #6, the gate is extended with two terms, both computed before any judging:

- **Set-vocabulary share** — fraction of generated tokens belonging to the driven set, versus the `B0_none` baseline (J0-b: **0.00000 in both arms**).
- **Lexical entropy** — Shannon entropy of the generated token distribution, versus baseline (J0-b: **4.618 → 4.627**, i.e. unchanged).

**Pre-declared rule.** An arm is **DEGRADED** if any of: malformed rate > 15 % (Phase 2 rule, retained); mean set-vocabulary share exceeds the `B0_none` mean by more than **0.05 absolute**; or mean lexical entropy falls more than **0.5 nats** below the `B0_none` mean. A degraded arm's point is **still reported and plotted, flagged**, and the flag is carried into every statement about it; a DV change in a degraded arm is registered as uninterpretable.

**Additional registered non-degradation check:** mean per-token negative log-likelihood of each arm's own generation under the **un-ablated** model, reported per arm. This is the perplexity comparison the PI asked for; it is reported, not gated, because a projection that changes the text necessarily changes its likelihood under the un-ablated model, and the quantity of interest is the magnitude.

## 8. Exclusions, storage, provenance

- **Exclusions** (inherited): judge parse failures excluded from DV1 and counted; any run failing digest re-verification discarded and re-run; all exclusions logged, N reported exactly per arm.
- **No optional stopping.** All runs are generated before any rate is computed. **No aggregates during generation** — analysis is a separate session against the data commit.
- **Storage:** readouts restricted to layers 17–26 × generation positions. Committed: run manifest (per-run seed, sha256, digests, achieved ablation depth), judge output, mention flags, degradation report, completeness report, and a data content digest.
- **Order of operations (gates, in order):** PI tag → `Projector` verification ALL_PASS (covering both `B1_full` and `B3_rand`) → smoke gate → confirmatory 600 → separate analysis session. **No model download is on the critical path** (§6.1 uses a matrix already in the checkpoint).

## 9. Open at prereg / not fixed here

- **Base-model robustness readout (optional).** Repeating §6's landing check with the **base `Qwen/Qwen2.5-7B`** unembedding would be a fully independent head. **PI decision: not now — no ~15 GB download.** Registered here as optional post-hoc robustness; if it is ever run it is reported as robustness, never as the registered landing criterion, which is §6.1.
- Final **R** confirmation (§4's rule gives **R = 10**; grid and budget tabulated).
- The `Projector`'s own verification run — mechanics are shared with the `Injector` verified ALL_PASS at `5453270`, but the projection path is new code and gets its own ALL_PASS gate before the smoke gate. The `B3_rand` arm is covered by the same gate.
- Any exploratory reporting: per-layer ablation profiles, per-language breakdown, the per-run relationship between mention and diagnosis (the run-level 2×2 that the two DVs make available for the first time).
- The analysis itself — a separate later session against the data commit.

**Resolved (PI, 2026-07-22):** `B2_half` removed, **`B3_rand` reinstated** as the specificity control (§2) · landing head = **untied `embed_tokens`**, base model deferred (§6) · power and budget refitted to **3 arms**, rule → **R = 10, 600 runs, ≈2.05 h** (§4) · the 3/20 vs 17/20 question answered in §0 and the ad-hoc measurement re-run from committed code.

---

*Generated at Stage J0. On freeze the PI edits as needed, then commits and runs `git tag -a prereg-phase2b-v1 -m "phase 2b preregistration freeze"` + push. **The delegate does not tag.***
