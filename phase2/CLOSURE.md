# Phase 2 — CLOSURE: instrument-negative outcome under the preregistration as written

**Recorded:** 2026-07-22. **PI adjudication of the Stage I1 block** (`phase2/PILOT_CALIBRATION.md` @ `aeb2cb2`). This closes the preregistered Phase 2. **No amendment was made; the preregistration is closed exactly as tagged.**

**Freeze of record:** tag `prereg-phase2-v1` (tag obj `f0586319`) → commit `eb176a9` → `PREREG_PHASE2.md`. **Data generated under it:** the condition-free `ρ_l` pass, the injecting-hook verification, and the calibration pilot on held-out `neutral` material. **No confirmatory data was ever generated.** The 20 confirmatory `high` vignettes were never run under any arm and remain unjudged.

---

## 1. Outcome

**Phase 2 closes with an instrument-negative result.** On `Qwen2.5-7B-Instruct` with a Tikhonov-inverted Jacobian-lens direction for the sealed Set F, **additive intensity has no regime in which the F representation is driven materially above its natural level without the model's output degrading.** The registered dose axis does not exist on this instrument.

This is a finding about the **method**, not about either registered hypothesis. Both Phase 2 outcomes — coupling-at-higher-intensity and intensity-robust decoupling — remain **untested**. Phase 1's result stands unchanged and unqualified by this: natural sustainment of fictional status exists (A1 `p = 4.117e-06`; A2 survives the ES mask, `p = 4.956e-08`) and does not alter behaviour (diagnosis 200/200).

## 2. The registered gate operated as designed

This is the part worth stating plainly: **nothing here required a judgement call.** §3.4's `k_max` rule was written before the pilot and executed mechanically:

| step | pre-declared rule | result |
|---|---|---|
| `k_max` | largest rung with malformed rate < 10 % | **0.05** (rungs 0.1–1.6 are **100 %** malformed) |
| `k₃` | min(rung reaching 50×, `k_max`) | 0.05 |
| `k₂`, `k₁` | rungs nearest 10× and 2×, subject to `k₁<k₂<k₃` | 0.05, 0.05 |
| **outcome** | three distinct doses | **FALSE — one dose** |

With one dose level the primary test **T1** (per-vignette slope of diagnosis on dose score) is **undefined**, not merely underpowered. §3.4 had anticipated a collapse to *fewer than three* doses; the actual collapse was to one. **The pre-registration caught its own design failure before any confirmatory run** — which is what it was for. ~2.5 h of GPU and 700 runs were not spent.

Two further facts, each independently sufficient:

- **Scale.** The registered targets (2×/10×/50× of 0.0825 = 0.165 / 0.825 / 4.125) lie entirely **below the ladder floor**: the lowest rung already reads 27.4 = **332× natural**. The ladder was mis-scaled by ~two orders of magnitude.
- **Anchor.** The α = 0 reference on the pilot's `neutral` material is **0.1595 ≈ 1.9× the 0.0825 anchor**. The registered 2× target *is* the untreated baseline of the calibration material. The content-transfer caveat the PI registered in §3.4 was **falsified as an assumption**, exactly where it was flagged.

## 3. Why the surviving dose could not be used either

Even taking the one dose the rule returned, it is unusable — and this is the substantive lesson. Fixed-seed generations on v48 (`phase2/data/pilot_sample_texts.json`):

| k | output |
|---|---|
| 0 | *"…analizar cuidadosamente los síntomas presentados y compararlos con los criterios sugeridos…"* — normal |
| 0.02 | *"Aunque esta categoría es ficticia para el estudio, podemos usarla como un marco conceptual…"* — coherent, fiction acknowledged in ordinary language |
| **0.05** | *"es importante experimentar con la experimentación teórica… no ha sido reconocida como una condición clínica experimentada o experimentando experimentación…"* — **lexical intrusion**: fluent, syntactically well-formed, semantically empty |
| 0.1 | `"fictionfictionfiction…"` ×183 — collapse |

**The addition does not modulate a semantic state; past a threshold it forces Set F tokens into the output distribution.** The readout climb (332× → 3616× natural) is therefore substantially *emission forcing*, and it is measured by an estimator that reads the very token logits the intervention drives — **the manipulation check and the artefact are the same quantity**. Between "indistinguishable from noise" (k ≤ 0.02, non-monotone across a 100× span of α) and "100 % malformed" (k ≥ 0.1) lies **less than one decade of α**, and the single rung inside it already exhibits the intrusion.

## 4. Why option (a) was rejected (PI)

Re-scaling into the cliff and re-anchoring was considered and **rejected**: the usable dynamic range is under one decade; the manipulation-check estimator is circular with respect to the intervention; and the only candidate rung already sits inside the degradation confound. Its best case converges on this same closure at additional cost. **(d) accepted; no amendment made.**

## 5. Lessons registered for any future attempt

**Lesson #5 — the EN-operative / ES-emission mismatch** *(carried forward, first recorded in `RESULTS_PHASE1.md` App. A2)*. The sealed operative token lists are realized in **English** while generation is **Spanish**. Any instrument that masks, matches, or drives *emission* will therefore misfire: the Phase 1 registered positional mask was inert (~0.24 % of positions maskable) and had to be replaced post-hoc by a Spanish-surface mask. **Rule for future work:** never assume a token-level operation on the operative list touches the generated surface; verify the mask/driver actually bites, and report the fraction it affects, before relying on it.

**Lesson #6 — malformedness detectors are blind to vocabulary flooding** *(new, from this pilot)*. The §7 detector (zero tokens / non-UTF-8 / ≤10-token n-gram >50 %) returned **0 % malformed at k = 0.05**, on text that is manifestly degraded — *"experimentada o experimentando experimentación"*. Repetition detectors catch *collapse*; they do not catch **fluent, well-formed, semantically vacuous output saturated with a target set's vocabulary**, which is precisely the failure mode a Set-directed activation addition produces. **Rule for future work:** any degradation gate on a set-directed intervention must include a **set-vocabulary share** term (fraction of generated tokens belonging to the driven set, versus the un-intervened baseline) and a **lexical-entropy** term. A gate built only on repetition will pass exactly the runs that most need excluding.

**Correction (b) — the manipulation check must not read the driven logits** *(registered, PI)*. Verifying an intervention with an estimator computed from the same unembedding the intervention targets is circular: it cannot distinguish "the representation moved" from "these tokens were forced". **Rule for future work:** the primary landing check runs on the **semi-independent base-model readout** (§6 check 2 of the closed prereg) — same residual, different readout head — and the instruct-lens readout is reported as a mechanical descriptive only, never as the landing criterion.

*(Lesson numbering is the PI's running project ledger; #5 was already in use in `RESULTS_PHASE1.md`. Distinct from the upstream `jacobian-lens` issue numbering — issue #5 Pitfall 1/2, issue #6 `isfinite` — recorded in `phase0/reports/stage01_recon.md`.)*

## 6. What is preserved and reusable

Nothing in the instrument itself failed; the failure is in the *additive* manipulation. The following are verified, committed, and carry forward unchanged to Phase 2b:

- **`phase2/scripts/intervene.py`** — target construction (`u_gain`), the Tikhonov solve at λ = 0.1, per-layer `k·ρ_l` dosing, the norm-matched control, the §2 F-loading estimator, the §7 detector.
- **The injecting hook**, verified ALL_PASS (`5453270`) — including the **KV-cache generation asymmetry** (the last generated token's residual is never computed, so injection and readout windows must both be `[P, total-1)`).
- **`ρ_l`** (`rho_layers.json`) — condition-free, layer-stable (sd ~1 % of mean), 6.1× growth across the band.
- **Landing geometry** — `cos_l` ≥ 0.8201 across the band at λ = 0.1, `‖J v̂‖` ≈ 0.98.
- **Stage I0's structural findings** — the degenerate-λ-rule analysis, the raw-logit readout units, the 755.1 reachability ceiling, and the saturation asymmetry.

---

*Phase 2 (preregistered, tag `prereg-phase2-v1`) is CLOSED. Next: Phase 2b, Stage J0 — projection-ablation design, `PREREG_PHASE2B.md`, PI review and tag. Interpretation of this closure is the PI's.*
