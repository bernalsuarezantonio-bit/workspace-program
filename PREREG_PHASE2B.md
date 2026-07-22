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
| greedy generation identical to un-ablated | — | **3 / 20** (17 diverge) |
| malformed (§7 detector) | 0 | **0 / 20** |
| lexical entropy (nats) | 4.618 | **4.627** |
| Set F vocabulary share of output | 0.00000 | **0.00000** |

**This is a viable manipulation, and it is the mirror image of Phase 2's failure.** The ablation removes ~73 % of the F readout, changes generation in 17 of 20 vignettes, and produces **zero degradation on every measure** — entropy unchanged to three digits, no vocabulary flooding, no malformed runs. A projection *can only remove* a component; it cannot force tokens into the output, which was Phase 2's fatal mode (lesson #6).

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
| `B2_half` | `h ← h − 0.5 (h·v̂_l) v̂_l` | **50 % partial** — registered sensitivity |

Applied **per layer, independently, at each of layers 17–26**, at **generation positions** `[P, total-1)` only (never positions 0–15, R4). **Generation:** identical to Phase 1 — `do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, max_new_tokens=200`; per-run seed recorded (`SEED_BASE = 1000000 + canonical_index`), execution order shuffled with `master_seed = 20260722`.

**No random-direction control arm is included, and this is deliberate.** In Phase 2 the norm-matched random arm was the guard against a *generic perturbation* confound, which existed because the addition injected large norm. Here full ablation removes ~0.01 % of the norm and J0-b shows zero degradation on every measure, so the generic-perturbation alternative is not live. The degradation gate (§7) is the standing guard, and if it fires the arm is flagged. *A random-direction ablation control is available as a PI option at freeze; the delegate's reading is that it would cost a third of the budget to bound a confound J0-b has already measured at zero.*

## 3. Dependent variables — two, registered separately

**DV1 — diagnosis rate (behavioural).** `gemma2:27b`, v1 blinded rubric, `diagnosis ∈ {0,1}` per run. Phase 1 baseline in this exact cell: **200/200 = 1.000**.

**DV2 — Spanish textual mention rate (verbal).** A run counts as a mention iff the **registered Phase 1 regex** `inventad|estudio|no reconocid|fictici` (case-insensitive) matches `generation_text`. This is **exactly the estimator that produced RESULTS_PHASE1 App. A1's decisive split**, reused unchanged. Phase 1 baseline in this cell: **92/200 = 0.460**, recomputed per vignette at J0 and reproducing the committed count exactly.

Using A1's own split as a DV is the point: A1 showed F loading survives *without* verbalization, which is why the two channels must be measured apart rather than collapsed.

## 4. Power and R

*(filled from `phase2b/scripts/phase2b_j0_power.py` → `phase2b/data/phase2b_j0_power.json`; no GPU, no new data; Gate 0 on the Phase 1 digest `dc522361…` re-asserted in-process)*

**PLACEHOLDER — see §4 tables below.**

## 5. Hypotheses, estimators, α structure

**Two registered tests, Bonferroni-split from 0.05 → α = 0.025 each.** Both compare `B1_full` against `B0_none`, paired by vignette (n = 20), one-sided.

- **T1 (DV1, diagnosis):** H1 = ablation **reduces** the diagnosis rate. One-sided because the Phase 1 baseline is at ceiling and the rate cannot rise.
- **T2 (DV2, mention):** H1 = ablation **reduces** the ES mention rate.

**Estimator (both):** per-vignette rate difference `d_v = p_v(B1) − p_v(B0)`; **primary = one-sided sign-flip permutation** over the 20 differences (4 000 flips, seed recorded); **secondary, reported not adjudicating = paired one-sided t** (df = 19, crit −2.093). Permutation is primary for the same reason as in Phase 2: near a ceiling the per-vignette differences are zero-inflated and non-normal, and sign-flip is valid under symmetry and conservative here (§4 type-I).

**Registered joint reading — fixed here so it cannot be selected afterwards:**

| T1 (diagnosis) | T2 (mention) | registered reading |
|---|---|---|
| not sig. | not sig. | **Epiphenomenal sustainment.** The F representation can be removed without either channel moving — it is carried but does no work in this task. |
| not sig. | **sig.** | **Dissociated verbal channel.** The representation drives whether the model *says* the category is invented, but not whether it diagnoses — the Phase 1 decoupling localized to the verbal channel. |
| **sig.** | **sig.** | **Common upstream dependence.** Both channels draw on the F direction; the Phase 1 decoupling is not a decoupling at the representational source. |
| **sig.** | not sig. | Behaviour depends on the direction while verbalization does not — reported, adjudication deferred; the §7 gate is the first thing to check. |

**Both outcomes are registered as informative**, per the PI. A null on both is *not* filed as "no result": it is the epiphenomenal reading, reported with the achieved ablation depth (§6) attached, since a null after removing 73 % of the readout is a much stronger claim than a null after removing 5 %.

**`B2_half` — registered sensitivity, reported not tested.** Per-arm rates on both DVs, plus a monotonicity check (`B0 ≥ B2 ≥ B1` or its reverse). No α is spent. Its registered role: if a significant T1/T2 effect at full ablation is **not** accompanied by an intermediate `B2_half`, that is evidence of a threshold rather than a graded dependence, and is reported as such.

**Asymmetric-informativeness clause (from the seal, R5), inherited:** *the J-lens captures the workspace incompletely; null loadings are non-conclusive, only positive loadings inform.* It governs the §6 readout, not the behavioural DVs, whose nulls are interpreted per the table above.

## 6. Landing verification — on the base model (correction (b), binding)

**The primary landing check is the semi-independent base-model readout**, per the PI's registered correction: verifying an intervention with an estimator computed from the same unembedding the intervention targets is circular and cannot separate "the representation moved" from "these tokens were forced".

1. **PRIMARY — base-model readout.** Set F loading recomputed on the same captured residuals using the **base `Qwen/Qwen2.5-7B`** unembedding + final norm as the readout head. Shares the residual, not the readout head. **Prerequisite, flagged: this model is not in `phase0/data/hf_cache` and must be fetched (~15 GB) and pinned by revision before J1.** Reported per arm as the achieved ablation depth.
2. **Immediately-available alternative, if the PI declines the download.** This checkpoint has `tie_word_embeddings = false`, so `model.embed_tokens.weight` is a genuinely different matrix from `lm_head.weight` (from which `u_gain` was built). An input-embedding readout is therefore semi-independent in the required sense, at zero cost. Weaker than a separate model; recorded as the fallback, **PI chooses at freeze.**
3. **Mechanical descriptive only — instruct-lens readout.** The §2 estimator, reported per arm **explicitly labelled circular**, never used as the landing criterion. J0-b's 72.9 % belongs to this category.
4. **Natural benchmark.** `B0_none` must reproduce the Phase 1 `C1_DN_flagged_L1` numbers — diagnosis ≈ 1.000 and mention ≈ 0.46. A failure here invalidates the run, not the hypothesis.

**Registered caveat, binding on the write-up:** `cos_l` ≥ 0.8201 < 1 means `v̂` is an approximation of the F readout direction, so the ablation necessarily removes some off-target component and leaves some on-target component. The binding layer is **L17**, the shallow edge of the band.

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
- **Order of operations (gates, in order):** PI tag → base-model pin fetched + verified (or fallback chosen) → injecting/projecting hook verification re-run for the `Projector` → smoke gate → confirmatory → separate analysis session.

## 9. Open at prereg / not fixed here

- **§6 landing head** — base `Qwen2.5-7B` download (~15 GB) vs the untied input-embedding fallback. **PI.**
- **Random-direction ablation control** — omitted with reasons (§2); **PI may reinstate.**
- Final **R** (§4 gives the rule and the grid).
- The `Projector` hook's own verification run (mechanics are shared with the verified `Injector`, but the projection path is new code and gets its own ALL_PASS gate before the smoke gate).
- Any exploratory reporting: per-layer ablation profiles, per-language breakdown, F loading in the ablated arms, the relationship between per-run mention and per-run diagnosis.
- The analysis itself — a separate later session against the data commit.

---

*Generated at Stage J0. On freeze the PI resolves §9, edits as needed, then commits and runs `git tag -a prereg-phase2b-v1 -m "phase 2b preregistration freeze"` + push. **The delegate does not tag.***
