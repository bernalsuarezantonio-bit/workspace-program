# Stage I0 — reconnaissance for the Phase 2 activation-addition instrument

**Recorded:** 2026-07-22, `[host]`, delegate session. **Cold re-derivation.** **NO GPU** (CPU linear algebra only), **no generation, no conditions, no token-set scoring of any run.** Nothing here is condition-bearing; the Phase 1 data commit `8046a12` / digest `dc522361…` is untouched.

**Script** `phase2/scripts/i0_recon.py` · **machine-readable** `phase2/data/i0_recon.json` · run against `PREREG_PHASE2.md` (DRAFT) at `9809ea7`.

**Standing rule applied (PI, 2026-07-22): every stage report is committed — a report without a hash does not exist.** The prior chat PDF is **hypothesis, not source**; nothing in it is cited here. Every number below is produced by the script above from pinned artifacts.

**Pins re-verified in-process:**

| artifact | value | check |
|---|---|---|
| lens `.pt` sha256 | `3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29` | == Stage 0.2 pin ✓ |
| model | `Qwen/Qwen2.5-7B-Instruct @ a09a3545` (fp16 safetensors, `lm_head.weight` untied) | ✓ |
| lens geometry | `d_model` 3584, 27 fitted layers, band 17–26 ⊂ `source_layers` | ✓ |
| Set F operative list | **11 SURVIVORS**, ids from `phase0/data/phase1_seal_screening_A1.json` | == seal A1 ✓ |
| env | torch 2.11.0+cu128, transformers 5.14.1 | recorded |

Set F survivors used (unchanged from the seal, no set content touched): `fiction`(57062), ` fiction`(16989), ` fictional`(43582), ` invented`(35492), `study`(54965), ` study`(3920), `experiment`(59429), ` experiment`(9342), `experimental`(86703, EN+ES), ` experimental`(22000, EN+ES), ` fabricated`(69454).

---

## 1. Capability of `jlens`, verified in code — ALL GREEN

Each claim is asserted mechanically by the script, not read off documentation or prose.

| # | claim | how verified | result |
|---|---|---|---|
| **C1** | the lens exposes per-layer `J_l` as `[d_model, d_model]` | shape assert on all 10 band layers | ✓ `[3584, 3584]` |
| **C1b** | `transport(h,l) == h @ J_lᵀ` (i.e. `J_l h` in column convention) | numeric `allclose` against a random `h` | ✓ |
| **C2** | the readout is `lm_head(final_norm(h @ J_lᵀ))` | source assert on `jlens/hf.py:HFLensModel.unembed` | ✓ **final norm is applied** |
| **C3** | a `register_forward_hook` returning a value **replaces** the block output | toy `nn.Linear` round-trip: hooked output == expected, un-hooked output restored | ✓ |
| **C4** | the pinned `Qwen2DecoderLayer.forward` returns a **bare Tensor** (not a tuple) | `inspect.getsource` → `return hidden_states` | ✓ |

**C3 is the load-bearing one for I1.** `jlens`'s `ActivationRecorder` (`jlens/hooks.py`) is **record-only** — it stores activations and returns `None`. The intervention therefore needs its **own** injecting hook, written for I1; the recorder cannot be reused for addition. C4 confirms the injecting hook receives and must return a plain Tensor for this model and this `transformers` version (a tuple-returning block would need unpacking; it does not here).

## 2. Target direction `u_F` — and a correction to PREREG §3.1

The readout is `logits = W_U · (g ⊙ x/rms(x))` where `g = model.norm.weight` is the final RMSNorm gain. The F-token logits are therefore driven by `⟨ g ⊙ Σ_t W_U[t] , x ⟩`, so the direction that maximizes them in the space `J_l` lands in is the **gain-corrected** one:

    u_gain = unit( g ⊙ Σ_{t∈F} W_U[t] )      vs.      u_raw = unit( Σ_{t∈F} W_U[t] )

**PREREG_PHASE2 §3.1 as drafted specifies `u_raw`, which omits the gain.** The gain is not near-uniform — `min −0.1738 / max 10.75 / mean 3.839 / sd 0.678` — and the two targets differ measurably: **`cos(u_raw, u_gain) = 0.9709`**.

**Both were carried through the entire analysis below.** The delegate does not pick; see Amendment **A-2**, §6.

## 3. Tikhonov sweep and `cos_l` — the band is healthy

`v̂_l = (J_lᵀJ_l + λ·mean_eig(J_lᵀJ_l)·I)^{-1} J_lᵀ u_F`, unit-normalized after the solve (§3.2b: magnitude lives in α, not λ). `cos_l = cos(J_l v̂_l , u_F)`.

**Band-minimum `cos_l` over the ladder** (the G1-relevant statistic):

| λ | `min_l cos_l` (u_raw) | `min_l cos_l` (u_gain) | **G1 verdict** (PI, fixed 2026-07-22, before any cos existed) |
|---|---|---|---|
| 1e-6 | 0.9984 | 0.9968 | ≥0.80 **PASS** |
| 1e-5 | 0.9956 | 0.9935 | **PASS** |
| 1e-4 | 0.9870 | 0.9815 | **PASS** |
| 1e-3 | 0.9707 | 0.9611 | **PASS** |
| 1e-2 | 0.9301 | 0.9184 | **PASS** |
| **1e-1** | **0.8292** | **0.8201** | **PASS** |
| 1.0 | 0.6568 | 0.6550 | 0.65–0.80 → executable-with-flag (degraded fidelity) |
| 10.0 | 0.4374 | 0.4578 | <0.65 → **gate closed** |

**Per-layer `cos_l` at λ=0.1** (L17→L26): u_raw `0.829 0.879 0.921 0.944 0.958 0.965 0.968 0.969 0.972 0.977`; u_gain `0.820 0.867 0.907 0.929 0.941 0.948 0.951 0.952 0.956 0.962`. The band minimum sits at **L17** in every case and rises monotonically with depth — the shallow edge of the band is the binding constraint.

**Verdict: G1 PASSES, with wide margin, for every λ ≤ 0.1 under both targets.** The band 17–26 does not need to be narrowed and the construction does not need to be abandoned. Layer conditioning is poor but not fatal — `cond(J_l)` ranges 5.1e4 (L23) to 1.55e6 (L17), effective rank at 99 % spectral energy 3079–3437 of 3584.

## 4. FINDING 1 (material) — the pre-declared λ rule is degenerate

**PREREG §3.2(c) as written:** *"select the single λ that maximizes the minimum over l of `cos_l`."*

`cos_l` is **monotonically decreasing in λ at every layer** (table §3). The rule therefore **always selects the smallest rung on whatever ladder it is given** — here λ=1e-6 — and would select 1e-9 had the ladder gone lower. It cannot select an interior λ. **The rule is unfalsifiable by construction and makes the Tikhonov regularization vacuous:** it degenerates to a raw pseudo-inverse.

**Why that is harmful, not merely inelegant.** With `v̂_l` unit-normalized, `‖J_l v̂_l‖` measures how much of an injected unit of residual norm actually **lands in the readout**:

| λ | mean `‖J v̂‖` over band (u_raw) | min over band | landing efficiency vs λ=0.1 |
|---|---|---|---|
| **1e-6** (rule's pick) | **0.0341** | 0.0255 | **~29× worse** |
| 1e-2 | 0.6903 | 0.4132 | ~1.4× worse |
| **1e-1** | **0.9758** | 0.7992 | — |
| 1.0 | 1.4599 | 1.1991 | 1.5× better, but G1-degraded |

At λ=1e-6 the solution is dominated by `J_l`'s near-null directions (σ_min ≈ 3.1e-5 at L17): **~97 % of the injected norm goes into directions the lens cannot see.** Those directions still perturb the model's real forward computation. This is precisely the pathology Tikhonov exists to prevent, and the rule as written *selects for it*. Concretely it would (i) force much larger α to reach any readout target, (ii) make the norm-matched `A4_rand` control a poor comparator — the F arm would carry a near-null-space perturbation while the random arm carries a generic one, and (iii) maximize the off-target load that §6 of the prereg already flags as bounded only by that control.

**The delegate has executed the rule as written** (λ=1e-6 is reported above as the rule's output) and **has not substituted its own.** Choosing a different λ after seeing this table is exactly the pre-registration inversion §0 warns about. It goes to the PI as Amendment **A-1**, §6.

## 5. FINDING 2 — the readout is raw logits; the 50× target is not analytically blocked

`run_confirmatory.py` dumps `logits.topk(10)` with **no softmax** — the `"weight"` field in every readout, and therefore the §2 loading estimator, is in **raw logit units** (this is why Phase 1's Set A loading of 2.45 exceeds 1 and is not a probability).

Because `final_norm` rescales any residual so that `‖x/rms(x)‖ = √d`, the summed F logit at a single position is **hard-bounded**:

    Σ_t W_U[t]·(g ⊙ y)  ≤  √d · ‖ g ⊙ Σ_t W_U[t] ‖  =  **755.1**

Against the §3.4 targets (2×/10×/50× the natural 0.0825 = 0.165 / 0.825 / **4.125**), the ceiling is **~9153× the natural level**. **The 50× target is therefore not blocked by the estimator's arithmetic** — a useful negative result, since a bound below 4.125 would have made §3.4 unsatisfiable on paper.

**Stated as a necessary, not sufficient, condition.** The ceiling assumes perfect alignment at *every* band position *and* that all 11 F tokens stay inside top-10 there. Real achievability is exactly what the calibration pilot measures, and the pilot's `k_max` rule (malformed <10 %) may still bind first. Recorded so that a pilot failure to reach 50× is read as an empirical limit, not as an arithmetic impossibility discovered late.

**Second-order note, recorded not acted on:** because the readout is scale-invariant in `J_l h` (the norm divides out), driving α higher moves the readout only by *rotating* `J_l h` toward `u_F`, which **saturates**. The causal effect on the model does **not** saturate — magnitude enters the real forward pass directly. So α is expected to keep changing behaviour after the readout verification has flattened. This asymmetry belongs in the §6 write-up caveat.

## 6. Amendments put to the PI (pre-data, no condition-bearing Phase 2 data exists)

Both are changes to a **rule**, not fillings of a slot, so neither is taken by the delegate.

**A-1 — replace the degenerate §3.2(c) λ rule.** Proposed replacement, which reuses *only* the threshold the PI already fixed today, before any `cos_l` existed:

> *Select the **largest** λ on the ladder whose band-minimum `cos_l` still meets the G1 pass threshold (≥0.80). Ties → the larger λ.*

This maximizes landing efficiency subject to the fidelity gate, is monotone in the opposite direction from the failed rule (so it selects an interior point), and introduces no new free parameter. **Under it, λ = 0.1** for both targets (`min_l cos_l` = 0.8292 raw / 0.8201 gain; mean `‖J v̂‖` 0.976 / 0.984). Alternatives the PI may prefer: keep the rule as written and accept λ=1e-6 with the near-null-space cost documented; or gate on a `‖J v̂‖` floor instead.

**A-2 — fix §3.1 to the gain-corrected target.** `u_gain` is the mathematically correct target for the registered readout (§2). Cost: `cos_l` is uniformly ~0.01 lower than with `u_raw`; G1 passes either way. Recommendation: adopt `u_gain`, since matching the estimator matters more than a first-decimal cos.

**Neither amendment changes any sealed set content, the band, the arms, the DV, R, or the α structure.**

## 7. Status

- **Gate G1: PASS** (band 17–26 intact; no layer dropped; construction not reopened).
- **§0 slots now closed by measurement:** `cos_l` per layer (§3), target conditioning (§2–3), λ **as the rule computes it** (λ=1e-6) — with A-1 outstanding.
- **Not I0-dependent, unchanged:** `ρ_l` (§3.3), `k₁,k₂,k₃` (§3.4).
- **Open for the PI:** A-1, A-2.

*Generated by `phase2/scripts/i0_recon.py`; machine-readable `phase2/data/i0_recon.json`. No GPU, no conditions, no counting. Interpretation of the amendments is the PI's.*
