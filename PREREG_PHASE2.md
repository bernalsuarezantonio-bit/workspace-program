# PREREGISTRATION — Phase 2 / Stage I1: causal activation-addition study (DRAFT for PI freeze)

**Status:** DRAFT **v3** — **§3 COMPLETE, no open slots, no open decisions.** **NOT frozen.** Becomes the preregistration when the PI reviews/edits and personally runs `git tag -a prereg-phase2-v1`. Stage I1 data generation is gated on that tag. *The delegate does not tag* (the Phase 1 waiver was a one-time PI-instructed act of signature and is not carried forward).

**✅ RESOLVED — the v1 freeze blocker.** Stage I0 was re-derived in cold, committed and pushed: **`phase2/I0_RECON.md` @ `52c6a17`** (script `phase2/scripts/i0_recon.py`, machine-readable `phase2/data/i0_recon.json`). No GPU, no conditions, no counting. The prior chat PDF was treated as hypothesis, not source; every I0 number is script-produced from pinned artifacts.

**✅ RESOLVED — both amendments (PI, 2026-07-22, §0.3).** **A-1 accepted: λ = 0.1.** **A-2 accepted: `u_gain`.** Both are dated pre-data amendments; §3 is filled with the accepted values throughout.

**✅ RESOLVED — `ρ_l` measured** (`phase2/data/rho_layers.json`, §3.3): condition-free greedy pass over the 20 `high` vignettes in the `A0_base` construction, 1.5 min GPU, no token sets scored, no counting.

**Standing rule (PI, 2026-07-22), applied from here on:** every stage report is committed — *a report without a hash does not exist.* This is the discipline **Incident #3** made necessary: *re-derive in cold; commit the artifact before citing it; do not reconcile.*

**Model/lens (digests re-verified at run time):** `Qwen/Qwen2.5-7B-Instruct @ a09a3545` (fp16); lens `neuronpedia/jacobian-lens @ 16a01f3` (`qwen2.5-7b-it`, 27 fitted layers), `.pt` sha256 `3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29`. Judge: `gemma2:27b` (this host's Ollama, digest re-verified), **v1 blinded rubric** `phase1/materials_canonical/scoring/judge_prompt.md` sha256 `e67e8e63…` — the same validated instrument as Phase 1, unchanged.

**Instrument inheritance.** The Phase 1 token-set seal is CLOSED and is inherited verbatim: A0 `3689ac85…` → A1 bilingual `9530aceb…` → note C-EN `cfce4742…`. Set F operative list = the sealed **A1 SURVIVORS, 11 tokens** (9 EN + 2 EN+ES incl. `experimental`); R2 echoes remain barred. **No set content is changed by Phase 2**, and no new sealing act is performed here.

**Materials.** The same 20 `high` vignettes and the same L1 wrapper / `DN_flagged` disclosure as the Phase 1 `C1_DN_flagged_L1` cell, regenerated from the tracked byte-exact canonicals at `phase1/materials_canonical/` (`vignettes.yaml 59f37915…`, `disorders.yaml 91d0ccb7…`, `L1_forum.md 1f9bb56c…`). Stimulus assembly, task instruction, and generation parameters are **byte- and setting-identical to Phase 1** (PREREG_PHASE1 §1). The only manipulated variable is the activation addition.

---

## 0. Stage I0 results (`phase2/I0_RECON.md` @ `52c6a17`)

### 0.1 Gate G1 — **PASS**

Threshold fixed by the PI on 2026-07-22 **before any `cos_l` existed**: **≥0.80 passes; 0.65–0.80 executable-with-flag (degraded fidelity declared); <0.65 gate closed, construction reopened pre-freeze.**

| λ | `min_l cos_l` (`u_raw`) | `min_l cos_l` (`u_gain`) | verdict |
|---|---|---|---|
| 1e-6 | 0.9984 | 0.9968 | **PASS** |
| 1e-4 | 0.9870 | 0.9815 | **PASS** |
| 1e-2 | 0.9301 | 0.9184 | **PASS** |
| **1e-1** | **0.8292** | **0.8201** | **PASS** |
| 1.0 | 0.6568 | 0.6550 | executable-with-flag |
| 10.0 | 0.4374 | 0.4578 | gate closed |

**The band 17–26 survives intact — no layer dropped, construction not reopened.** The binding layer is **L17** at every λ; `cos_l` rises monotonically with depth. Per-layer values at the accepted λ=0.1 / `u_gain` are in **§0.4**. (Both target columns are retained above because the sweep predates the A-2 decision; the accepted branch is `u_gain`.)

### 0.2 Capability of `jlens`, verified in code — all GREEN

`J_l` exposed as `[3584,3584]` and `transport(h,l) == h @ J_lᵀ` ✓ · readout is `lm_head(final_norm(h @ J_lᵀ))`, **final norm applied** ✓ · a forward hook **replaces** a block's output ✓ · `Qwen2DecoderLayer` returns a **bare Tensor** (transformers 5.14.1) ✓. **Consequence for I1:** `ActivationRecorder` is *record-only*; the intervention needs its own injecting hook, to be written and committed at I1 execution.

### 0.3 Two amendments — **BOTH ACCEPTED (PI, 2026-07-22, pre-data)**

Dated amendments in the sense the seal permits: appended, dated, and **pre-data** — no condition-bearing Phase 2 readout exists at acceptance time (the only Phase 2 measurement in existence is the condition-free `ρ_l` pass, §3.3, which scores no token set). Both are **instrument-motivated only**: neither was informed by any Phase 2 result, because none exists.

**A-1 — ACCEPTED: λ = 0.1.** Structural justification, recorded verbatim as the ground of acceptance: (i) the original rule is **unfalsifiable** — `cos_l` is monotone decreasing in λ, so "maximize `min_l cos_l`" returns the ladder's smallest rung for *any* ladder and can never select an interior λ; (ii) the replacement uses **only the G1 threshold the PI had already fixed** (≥0.80), set before any `cos_l` existed, and introduces no new free parameter; (iii) the motivation is **purely instrumental** — landing efficiency `‖J v̂‖` (0.976 vs 0.034, ~29×) — and no result datum was available to bias it.

**A-2 — ACCEPTED: `u_gain`.** Ground: coherence with the DV's estimator. The readout is `W_U·(g ⊙ x/rms(x))`, so `u_gain = unit(g ⊙ Σ_t W_U[t])` is the direction the registered loading estimator actually responds to; matching the estimator outweighs the ~0.01 lower `cos_l`.

**Neither amendment touches sealed set content, the band, the arms, the DV, R, or the α structure.** The two I0 findings recorded as-is and approved by the PI: the analytic reachability ceiling (§3.4) and the saturation asymmetry (§6).

<details><summary>Original statement of the two problems (retained for the record)</summary>

**A-1 — the §3.2(c) λ rule is degenerate.** `cos_l` is monotonically decreasing in λ at every layer, so *"maximize `min_l cos_l`"* **always selects the ladder's smallest rung** and can never select an interior λ. At its pick (λ=1e-6) the solution sits in `J_l`'s near-null space: mean `‖J v̂‖` = **0.0341** vs **0.9758** at λ=0.1 — **~97 % of injected norm lands where the lens cannot see it**, while still perturbing the model. The rule selects for exactly the pathology Tikhonov prevents, and it degrades the `A4_rand` comparison and inflates the off-target load §6 warns about.

> **Proposed replacement (uses only the threshold the PI already fixed):** *select the **largest** λ whose band-minimum `cos_l` still meets G1 (≥0.80); ties → the larger λ.* → **λ = 0.1** under both targets. Alternatives: keep the rule as written and accept λ=1e-6 with the cost documented; or gate on a `‖J v̂‖` floor instead.

**A-2 — §3.1 omits the final-norm gain.** The readout is `W_U·(g ⊙ x/rms(x))`, so the correct target is `u_gain = unit(g ⊙ Σ_t W_U[t])`, not `u_raw`. The gain is far from uniform (min −0.174 / max 10.75 / mean 3.839 / sd 0.678) and `cos(u_raw, u_gain) = 0.9709`. G1 passes either way; `cos_l` is ~0.01 lower under `u_gain`.

</details>

### 0.4 Frozen instrument values

At **λ = 0.1** with target **`u_gain`** (the accepted branch), per layer L17→L26:

| | L17 | L18 | L19 | L20 | L21 | L22 | L23 | L24 | L25 | L26 |
|---|---|---|---|---|---|---|---|---|---|---|
| `cos_l` | **0.820** | 0.867 | 0.907 | 0.929 | 0.941 | 0.948 | 0.951 | 0.952 | 0.956 | 0.962 |
| `‖J v̂_l‖` | 0.856 | 0.822 | 0.841 | 0.907 | 1.001 | 1.071 | 1.112 | 1.100 | 1.074 | 1.059 |
| `ρ_l` | 76.09 | 80.91 | 97.29 | 112.15 | 137.52 | 171.18 | 216.27 | 279.57 | 355.33 | 463.97 |

**G1: PASS at every layer** (band-minimum 0.820 at L17 ≥ 0.80). Nothing left open.

---

## 1. Premise (redrafted from `RESULTS_PHASE1.md` @ `f39df1f`)

The Phase 2 premise carried in earlier chat narrative — *"Phase 1 found recognition-as-echo, no sustained holding"* — is **retired**; PROVENANCE §Incident #3 records that it had no committed numerical basis. The premise below is drafted **only** from committed numbers.

**What Phase 1 established.** Natural sustainment of fictional status **exists** and is measurable in the workspace during diagnosis, and it does **not** alter behaviour:

- **A1, registered decisive cell** (`RESULTS_PHASE1.md` App. A): among flagged runs that **never verbalize** the fiction, Set F loading is **0.0825** vs a plausible floor of **0.0418** — paired over 20 vignettes, diff **0.0385**, 95% CI **[0.0259, 0.0512]**, **t(19)=6.371, p=4.117e-06**. Roughly **2× the floor**, without emission.
- **A2, Spanish-surface emission mask** (post-hoc): the effect **survives** removal of Spanish fiction-verbalizing spans — masked diff **0.0815**, 95% CI **[0.0619, 0.1012]**, **t(19)=8.672, p=4.956e-08**.
- **Behaviour is unmoved.** In that same cell the model diagnoses the invented disorder at **200/200 (1.000)**. Set F is loaded and inert: the registered *loaded-but-inert* architecture, not *never-enters*.

**The Phase 2 question, stated as the residual.** Phase 1 shows a *know→behave* decoupling **at the naturally occurring intensity of the F representation**. It cannot say whether that decoupling is a property of the architecture or a property of the intensity. Stage I1 asks exactly that, causally:

> **If the F direction is driven well above its natural level, does diagnosis behaviour change?**

**R5 asymmetric-informativeness clause (verbatim from the seal), inherited and binding:** *"the J-lens captures the workspace incompletely; null loadings are non-conclusive, only positive loadings inform."* In Phase 2 this clause governs the **readout** verification (§6), not the behavioural DV; a behavioural null is interpreted per §5, and interpretation of both remains the PI's.

## 2. Design

Five arms, **within-item**: all 20 `high` vignettes appear in all five arms, analyzed paired-by-vignette.

| arm | addition | purpose |
|---|---|---|
| `A0_base` | none | baseline; replicates the Phase 1 `C1_DN_flagged_L1` cell exactly |
| `A1_low` | `+ k₁·ρ_l · v̂_l` | dose 1 |
| `A2_mid` | `+ k₂·ρ_l · v̂_l` | dose 2 |
| `A3_high` | `+ k₃·ρ_l · v̂_l` | dose 3 |
| `A4_rand` | `+ c_l · r_l`, `‖c_l r_l‖ = ‖k₃ ρ_l v̂_l‖` | **norm-matched random control** at top dose |

`r_l` is a fixed random unit vector in `R^d_model`, drawn once per layer from a **recorded seed** (`RAND_SEED = 20260722`) and identical across all runs, vignettes, and reps — so the control arm is a *fixed alternative direction*, not per-run noise. Norm matching is per layer.

**Dose scores** for the trend test: `A0_base=0, A1_low=1, A2_mid=2, A3_high=3`. `A4_rand` is **excluded** from the trend test and enters only the control contrast (§5).

**Generation** — identical to Phase 1: `do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, max_new_tokens=200`. Per-run seed recorded (`SEED_BASE = 800000 + canonical_index`); execution order shuffled with `master_seed = 20260722`.

## 3. The intervention (specification)

### 3.1 Target direction in the final-layer basis

The lens maps a residual at layer `l` into the final-layer basis by `transport(h,l) = h @ J̄_lᵀ` (`vendor/jacobian-lens/jlens/lens.py:135`), and the readout is the unembedding of that transported vector. The Set F target is therefore built **in the final-layer basis**:

I0 verified in code that the readout applies the final RMSNorm before the LM head: `logits = W_U · (g ⊙ x/rms(x))`, with `g = model.norm.weight`. **FIXED (A-2 accepted, §0.3):**

    u_F  =  u_gain  =  unit( g ⊙ Σ_{t ∈ F_survivors} W_U[t] )

The sum is over the **11 sealed Set F SURVIVOR tokens**, unweighted (each operative token counts once, matching the §2 loading estimator, which sums operative-token weights unweighted). `W_U` is read from the same checkpoint whose digest is re-verified per run; I0 confirms this model does **not** tie embeddings (`lm_head.weight`, `[152064, 3584]`). The 11 ids are `57062, 16989, 43582, 35492, 54965, 3920, 59429, 9342, 86703, 22000, 69454` — verified identical to seal A1.

### 3.2 Per-layer additive vector by Tikhonov inverse of the lens

For each layer `l ∈ {17,…,26}` (the Phase 1 primary band, inherited unchanged):

    v̂_l = (J̄_lᵀ J̄_l + λ I)^{-1} J̄_lᵀ u_F ,   then rescaled to ‖v̂_l‖ = 1

(a) `J̄_l` is the lens transport matrix at `l`, in fp32 for the solve, from the pinned `.pt` (sha re-verified).
(b) `v̂_l` is unit-normalized **after** the solve, so all magnitude information lives in `α` (§3.3–3.4) and none in `λ`.
(c) **`λ` — FIXED at 0.1** (A-1 accepted, §0.3), scaled by the mean eigenvalue of `J̄_lᵀJ̄_l`, **one λ shared across the whole band** so the arms differ only in intensity. Selection rule of record: *the largest λ on the ladder `{1e-6 … 10}` whose band-minimum `cos_l` still meets G1 (≥0.80); ties → the larger λ.* Result: **λ = 0.1**, band-minimum `cos_l` = **0.8201** at L17, mean `‖J v̂‖` = **0.984**.

(d) Per-layer `cos_l` and `‖J v̂_l‖` at the fixed λ are tabulated in **§0.4**; the full `(λ, l)` sweep is in `phase2/data/i0_recon.json` @ `52c6a17`. **Gate G1 PASSES at every layer in the band.**

### 3.3 `ρ_l` — the per-layer residual scale (condition-free, fully specified)

`ρ_l` is measured in a **fresh, condition-free forward pass** that **precedes** the calibration pilot and touches no experimental condition:

- Stimulus: the 20 `high` vignettes in the **`A0_base` cell construction** (flagged × L1), the same prompts the study uses, with **greedy decoding and `max_new_tokens = 200`**, one pass per vignette, no reps, no conditions varied.
- Capture: `ActivationRecorder(model.layers, at=range(17,27))` (`vendor/jacobian-lens/jlens/hooks.py`), which stores each block's output residual.
- **`ρ_l` = the mean over vignettes of the mean over GENERATION positions of the L2 norm `‖h_{l,pos}‖₂`** — generation positions only, never positions 0–15 (R4, inherited). Mean, not sum (R5 length-confound rule, inherited).
- Output: `phase2/data/rho_layers.json` (10 numbers + the per-vignette table + seeds + digests), committed. **No token sets are scored, no loadings computed, no counting** — this is a scale measurement, exactly analogous to the Phase 0 nightly technical calibration.

**MEASURED (2026-07-22).** Script `phase2/scripts/measure_rho.py`; output `phase2/data/rho_layers.json`. Greedy pass over the 20 `high` vignettes in the `A0_base` construction; all 20 ran to the 200-token cap (P ≈ 331–351, G = 200); **1.5 min GPU**. Prompt assembly byte-faithful to `run_confirmatory.py`, so `ρ_l` is measured on exactly the prompts `A0_base` will use. Lens `.pt` sha re-verified in-process.

| layer | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|
| **`ρ_l`** | **76.09** | 80.91 | 97.29 | 112.15 | 137.52 | 171.18 | 216.27 | 279.57 | 355.33 | **463.97** |
| sd (between vignettes) | 0.74 | 0.73 | 0.53 | 0.80 | 0.82 | 1.13 | 1.54 | 1.90 | 3.01 | 5.88 |

**Two mechanical observations, recorded not interpreted.** (i) Between-vignette variability is ~1 % of the mean at every layer, so `ρ_l` is essentially a property of the layer, not of the item — the per-layer dose is stable across the 20 vignettes. (ii) `ρ_l` grows **6.1×** across the band (76 → 464). This is exactly why the dose is specified as `k·ρ_l` and not as a fixed absolute norm: a single absolute α would be ~6× more aggressive at L17 than at L26, confounding depth with intensity.

**Condition-free, as specified:** no token sets scored, no loadings computed, no counting, no judge — a scale measurement analogous to the Phase 0 nightly technical calibration. It creates no condition-bearing data and does not touch the freeze.

### 3.4 `α` and the calibration pilot (empirical units)

The addition applied at layer `l`, at every **generation** position (re-added per layer, i.e. the vector is added to the residual at each of the 10 layers independently, not propagated once), is:

    h_{l,pos} ← h_{l,pos} + k · ρ_l · v̂_l          for pos ∈ generation positions

α is expressed in **empirical readout units**, not raw norm units. `k₁, k₂, k₃` are the multipliers whose **measured Set F readout at the added positions** reaches approximately **2×, 10×, and 50×** the natural without-mention level **0.0825** (`RESULTS_PHASE1.md` App. A1) — i.e. targets ≈ **0.165, 0.825, 4.125** in the §2 loading estimator, restricted to positions that received the addition.

**I0 reachability check (registered, from `52c6a17`).** The dumped readout `"weight"` is a **raw logit** (`logits.topk(10)`, no softmax — this is why Phase 1's Set A loading of 2.45 exceeds 1 and is not a probability). Because `final_norm` fixes `‖x/rms(x)‖ = √d`, the summed F logit at one position is hard-bounded by `√d·‖g ⊙ Σ_t W_U[t]‖ = **755.1**`, i.e. **~9153× the natural level**. **The 50× target (4.125) is therefore not blocked by the estimator's arithmetic.** This is a *necessary, not sufficient* condition — the ceiling assumes perfect alignment at every band position with all 11 F tokens inside top-10. Real achievability is what the pilot measures, and `k_max` (malformed <10 %) may still bind first; registered so that a pilot failure to reach 50× is read as an empirical limit, not an arithmetic impossibility found late.

**Calibration pilot — rule written before it is run:**
- **PI DECISION (2026-07-22): 2 `neutral` vignettes, drawn from OUTSIDE the confirmatory set**, by a recorded seed. **The 20 `high` vignettes stay wholly intact for the confirmatory block** — no study item is touched by calibration. **Registered caveat — content transfer:** `k` is calibrated on `neutral` material and applied to `high` material, so the calibration transfers across a content difference. The transfer is *assumed, not verified*: `ρ_l` (§3.3) is measured on the `high` construction, so the norm-based part of the dose is on-target, but the readout-target part (the 2×/10×/50× rungs) is calibrated off-target. Registered consequence: the **achieved** readout in the confirmatory arms is reported per arm (§6 check 1) against the intended rungs, and any shortfall is reported as a calibration-transfer gap, not silently absorbed.
- **Ladder, fixed here (pre-declared, before the pilot runs): `k ∈ {0.05, 0.1, 0.2, 0.4, 0.8, 1.6}`** — six geometric rungs, ratio 2, spanning an addition of **5 % to 160 % of the natural per-layer residual norm**. With `ρ_l` from §3.3 this is a fully determined set of absolute magnitudes, e.g. at L17 `α ∈ {3.80, 7.61, 15.22, 30.43, 60.87, 121.7}` and at L26 `α ∈ {23.2, 46.4, 92.8, 185.6, 371.2, 742.3}`. The ladder is fixed **before** the pilot; the pilot only reads off which rungs hit which targets.
- 5 reps per rung, measuring (i) the Set F readout at added positions, against the **natural benchmark 0.0825** and the 2×/10×/50× targets (0.165 / 0.825 / 4.125), and (ii) the **malformed rate** (§7).
- **`k_max` rule (pre-declared): the largest rung with malformed rate < 10 %.** `k₃` = min(the rung hitting the 50× target, `k_max`). `k₂`, `k₁` = the rungs nearest the 10× and 2× targets, subject to `k₁ < k₂ < k₃`.
- If `k₃` is capped by `k_max` below the 10× target, the design **collapses to fewer than three distinct doses** — pre-declared response: report it, run the doses that exist, and record the trend test on the reduced score set (which remains valid; power drops, §4).
- Pilot outputs, ladder, and the chosen `k` are committed **before** any confirmatory run.

## 4. Power and R

Script `phase2/scripts/phase2_i1_power.py`; output `phase2/data/phase2_i1_power.json`; Gate 0 (Phase 1 data digest `dc522361…`) re-asserted in-process. **No GPU, no new data** — it reads only the committed Phase 1 judge outcomes.

**Real inputs (two, and only two):**

| input | value | source |
|---|---|---|
| baseline diagnosis rate | **200 / 200 = 1.000** (Jeffreys posterior mean 0.9975) | `C1_DN_flagged_L1`, `judge_full.jsonl` |
| between-vignette ICC of the binary judge outcome | **0.000** (raw moment estimate −0.0035) | `C2_incoherent_L1` (56/199) — the **only** non-ceiling confirmatory cell; a ceiling cell carries no clustering information |

**Registered caveat on the ICC.** It is estimated from k=20 clusters of n₀≈10 binary draws and **cannot distinguish "no clustering" from "modest clustering"** (per-vignette rates in that cell span 0.10–0.60, a spread consistent with binomial noise at n≈10). Power is therefore reported at **ICC ∈ {0, 0.05, 0.15}** and R is chosen against the inflated values.

**Estimators simulated** (identical to §5): PRIMARY = per-vignette OLS slope of run-level diagnosis on dose score, then a one-sided **sign-flip permutation test** over the 20 vignette slopes; SECONDARY = paired one-sided t (df=19, crit −2.093), top dose vs baseline. 20 000 sims (8 000 for sensitivity/type-I), seed 20260722. Because the baseline sits at ceiling, exact 1.0 is not a usable simulation parameter; power is reported at three conservative baselines.

**Power, primary trend test, α = 0.025** (`D` = the drop in P(diagnosis) at the top dose; lower doses on a 0.25/0.55/1.0 ramp):

| baseline p₀ | D=0.05 | D=0.10 | D=0.15 | D=0.20 |
|---|---|---|---|---|
| 0.9975, R=5 | 0.535 | 0.946 | 0.997 | 1.000 |
| 0.9975, **R=7** | 0.767 | **0.993** | 1.000 | 1.000 |
| 0.99, R=5 | 0.453 | 0.892 | 0.989 | 1.000 |
| 0.99, **R=7** | 0.638 | **0.970** | 0.999 | 1.000 |
| 0.97, R=5 | 0.315 | 0.757 | 0.950 | 0.994 |
| 0.97, **R=7** | 0.437 | **0.888** | 0.992 | 0.999 |

**ICC sensitivity** (p₀ = 0.99, primary trend test):

| ICC | D=0.05, R=7 | D=0.10, R=7 | D=0.15, R=7 | D=0.10, R=5 |
|---|---|---|---|---|
| 0.00 | 0.650 | 0.972 | 0.999 | 0.888 |
| 0.05 | 0.544 | 0.931 | 0.995 | 0.835 |
| 0.15 | 0.413 | 0.826 | 0.966 | 0.741 |

**Type-I rate under a true null** (all arms at p₀): **0.000–0.021 across the grid**, i.e. at or below the nominal 0.025 everywhere — the discreteness of a near-ceiling binary DV makes the permutation test *conservative*, never anti-conservative. Recorded as a property of the test.

**Pre-fixed rule → R = 7** (5 arms × 20 vignettes × 7 reps = **700 runs**). Justification: ≥0.93 power to detect a 10-point drop from ceiling even at ICC = 0.05, and 0.83 at the pessimistic ICC = 0.15; R = 5 falls to 0.84/0.74 on the same cells. The design is explicitly powered **for a drop from ceiling**, which is the only direction the baseline permits.

**GPU budget** (Phase 1 measured rates: 7.41 s/run generation, 3.37 s/run judge):

| item | estimate |
|---|---|
| `ρ_l` fresh pass (20 greedy passes) | ~3 min |
| calibration pilot (2 vignettes × ~5 rungs × 5 reps ≈ 50 runs) | ~10 min |
| smoke gate (8 runs, as Phase 1) | ~3 min |
| confirmatory 700 runs × (7.41 gen + 3.37 judge + ~1.5 readout) | **~143 min** |
| **total** | **~2.65 h** |

This fits ≤3 h **only with a clean GPU window.** Phase 1 lost ~30 % of run speed to `gemma2:27b` spilling to CPU under Ollama contention; at that rate 700 runs would overrun. Two pre-declared mitigations:
1. **Readout dumps are restricted to layers 17–26 × generation positions** (Phase 1 dumped all 27 layers × all positions) — ~10× less I/O and disk, sufficient for the §6 verification. Est. ≈ 1 GB total.
2. **Pre-declared fallback, decided by the smoke gate and not by results:** if the smoke gate's measured per-run time projects the confirmatory block beyond 3 h, **R drops to 5 (500 runs)** and the reduction is recorded before the confirmatory block starts. Power at R=5 is tabulated above.

## 5. Hypotheses, estimators, α structure

**Two registered tests, Bonferroni-split from 0.05 → α = 0.025 each.** Both one-sided; both use vignette as the unit (n = 20).

**T1 — PRIMARY, dose-response.** Per vignette `v`, let `p_{v,a}` be the diagnosis rate over that vignette's R runs in arm `a ∈ {A0,A1,A2,A3}`, and `s_a ∈ {0,1,2,3}` the dose score. The per-vignette slope is the OLS slope `β_v = Σ_a (s_a − s̄) p_{v,a} / Σ_a (s_a − s̄)²`.

> **H1(T1):** mean `β_v` < 0 — P(diagnosis) decreases monotonically with F-vector intensity.

Test: **one-sided sign-flip permutation** on the 20 slopes (4 000 flips, seed recorded; exact enumeration 2²⁰ optional). *Rationale for permutation over t:* near a ceiling the per-vignette slopes are zero-inflated and non-normal; sign-flip is valid under the null of symmetry and, per §4, conservative here. **Reported alongside (not adjudicating):** the paired one-sided t on the same slopes, and the paired one-sided t on `A3 − A0`.

**T2 — the control contrast (direction-specificity).** Per vignette, `d^F_v = p_{v,A3} − p_{v,A0}` and `d^R_v = p_{v,A4} − p_{v,A0}`.

> **H1(T2):** mean (`d^F_v − d^R_v`) < 0 — the F direction drops diagnosis *more* than a norm-matched random direction does.

Test: one-sided sign-flip permutation on the 20 paired differences, α = 0.025.

**Registered joint reading (fixed here, so it cannot be chosen later):**

| T1 | T2 | registered reading |
|---|---|---|
| sig. | sig. | **Dose-dependent, direction-specific coupling.** The know→behave link exists and is a matter of intensity. |
| sig. | not sig. | Behaviour moves with intensity but **not specifically with the F direction** — consistent with generic perturbation/degradation. Adjudication deferred; §6 and the degradation diagnostics are the evidence. |
| not sig. | — | **Decoupling robust to intensity** across the doses actually achieved. Informative, with the §7 limitation named. |

**Both outcomes are registered as informative.** A null in T1 is *not* filed as "no result": it says the decoupling survived driving the F direction to the achieved intensity, and it is reported with the achieved intensity (§6) attached, since a null at 2× is a far weaker claim than a null at 50×.

**Registered discriminating diagnostic (reported, not tested).** The v1 judge rubric already emits six keys. Per arm we report `epistemic_flag`, `criteria_invented`, `differential_offered`, `treatment_recommended`, `confidence`, plus the malformed rate (§7). The registered discrimination: a diagnosis drop accompanied by **rising `epistemic_flag`** at flat malformed rate reads as fiction-status holding; a drop accompanied by **rising malformed rate** and flat `epistemic_flag` reads as degradation. This is stated in advance precisely because it is the design's central inferential threat, and it is *reported, not tested* — no α is spent on it.

## 6. Landing verification (registered as verification, NOT as evidence of natural representation)

Three checks that the addition arrived. **All three are verification of the manipulation.** None is evidence that the model naturally represents fictional status at the driven level — that claim belongs to Phase 1 (§1) and is not re-litigated here.

1. **Lens readout (mechanical).** Set F loading by the §2 estimator, restricted to layers 17–26 and to positions that received the addition, per arm. Expected to rise monotonically with dose *by construction*. Reported against the natural reference 0.0825 and the 2×/10×/50× targets, so the **achieved** intensity is on the record next to whatever the DV did. This check is **not independent** of the intervention and is labelled so.
2. **Base-model readout (semi-independent).** The same loading recomputed with the **base** `Qwen2.5-7B` (non-instruct) unembedding as the readout head, on the same captured residuals. Semi-independent: it shares the residual but not the readout head, so it detects a v̂ that satisfies the instruct lens' inverse without carrying the concept.
3. **Natural benchmark.** The `A0_base` arm must reproduce the Phase 1 `C1_DN_flagged_L1` numbers — diagnosis rate ≈ 1.000 and Set F loading in the Phase 1 range. **Registered as a replication check on the whole pipeline**; a failure here invalidates the run, not the hypothesis.

**Registered caveat, binding on the write-up:** `cos_l < 1` (§0.4) means the injected direction is an *approximation* of the F readout direction and necessarily carries off-target components. At the fixed λ=0.1 the band-minimum is 0.820 (L17), so the off-target fraction is **largest at the shallow edge of the band**; the magnitude of that off-target load is bounded only by the `A4_rand` control, not by check 1.

**Registered saturation asymmetry (I0, `52c6a17`).** The lens readout is **scale-invariant in `J_l h`** (the final norm divides the magnitude out), so raising α moves the readout only by *rotating* `J_l h` toward `u_F` — which **saturates**. The causal effect on the model does **not** saturate: magnitude enters the real forward pass directly. Consequence, registered in advance: at high α the readout verification is expected to flatten while behaviour may still be moving. **A flat check-1 curve at the top dose is therefore not evidence that the dose stopped increasing**, and must not be written up as such.

## 7. Exclusions, degradation, storage, provenance

**Malformedness (pre-declared, mechanical, before any judging):** a run is *malformed* if it has zero generated tokens, is non-UTF-8-decodable, or is mechanically detected as non-terminating repetition (a single ≤10-token n-gram occupying >50 % of the generation). **Rule: if an arm's malformed rate exceeds 15 %, that arm is declared DEGRADED — its point is still reported and plotted, flagged as degraded, and the flag is carried into every statement about it.** A degraded arm is *not* silently dropped and *not* silently kept: the registered reading is that a diagnosis drop in a degraded arm is uninterpretable as fiction-holding.

**Other exclusions** (inherited from PREREG_PHASE1 §6): judge parse failures excluded from the rate and counted; any run whose model/lens/tokenizer digest fails re-verification is discarded and re-run. All exclusions logged; N reported exactly per arm.

**No optional stopping.** The 700 (or pre-declared 500) runs are generated in full before any diagnosis rate is computed. **No aggregates are computed during data generation** — analysis is a separate session against the data commit, exactly as Phase 1 (PROVENANCE §Stage P1).

**Storage:** per-run readouts restricted to layers 17–26 × generation positions (§4), gitignored; committed are the run manifest (per-run seed, readout sha256, digests, achieved-α record), the judge output, the completeness report, `rho_layers.json`, the λ/`cos_l` table from I0, the calibration-pilot ladder, and a data content digest over the manifests.

**Order of operations (gates, in order):** I0 committed → **G1** → PI freeze + tag → `ρ_l` pass → calibration pilot + `k` committed → smoke gate (+ R fallback decision) → confirmatory 700 → separate analysis session.

## 8. Open at prereg / not fixed here

**No open decisions and no open slots remain.** What is left is execution work, each item committed before the step it gates:

- The **I1 injecting hook** (§0.2) — to be written and committed at execution; `ActivationRecorder` is record-only and cannot be reused.
- The **calibration pilot's outcome** — which rungs of the fixed ladder (§3.4) hit the 2×/10×/50× targets, and `k_max`. Committed before any confirmatory run.
- The **`A4_rand` vectors** — drawn once per layer from `RAND_SEED = 20260722` at execution and recorded.
- Any **exploratory** reporting: all-layer profiles, per-position addition profiles, per-language Set F breakdown, the `A4_rand` arm's own readout profile.
- The **analysis** itself — a separate later session against the data commit; no aggregates during generation.

**Resolved (PI, 2026-07-22):** Stage I0 re-derived, committed, pushed at `52c6a17` · G1 threshold ≥0.80 / 0.65–0.80 flagged / <0.65 closed, fixed *before any `cos_l` existed* (§0.1) · **A-1 accepted → λ = 0.1** and **A-2 accepted → `u_gain`** (§0.3), both dated pre-data · reachability ceiling and saturation asymmetry approved as recorded (§3.4, §6) · `ρ_l` measured (§3.3) · pilot on 2 `neutral` vignettes outside the confirmatory set, content-transfer caveat registered (§3.4) · R = 7 with the smoke-gate R→5 fallback and the I/O mitigations (§4) · the three design additions approved — T1×T2 joint-reading table, sign-flip permutation as primary, drop-vs-degradation rubric (§5) · standing rule: *a report without a hash does not exist.*

**Order of operations from here:** PI reads + tags → I1 injecting hook committed → calibration pilot + `k` committed → smoke gate (+ R fallback decision) → confirmatory 700 → separate analysis session.

---

*Generated at Stage I1 (draft v3, post-I0, post-amendments, post-`ρ_l`). §3 is complete. On freeze the PI edits as needed, then commits and runs `git tag -a prereg-phase2-v1 -m "phase 2 preregistration freeze"` + push. **The delegate does not tag.***
