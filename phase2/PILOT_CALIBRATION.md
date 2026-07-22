# Stage I1 — calibration pilot: **the frozen dose ladder is not executable**

**Recorded:** 2026-07-22, `[host]`. Against **`PREREG_PHASE2.md` @ tag `prereg-phase2-v1`** (commit `eb176a9`). Script `phase2/scripts/run_pilot.py`; data `phase2/data/pilot_calibration.json`; sample generations `phase2/data/pilot_sample_texts.json`. **98 runs, 8.4 min GPU. No judge was run. No confirmatory data was generated.**

**Held-out material as the PI directed:** 2 `neutral` vignettes — **v48, v43** — drawn by `PILOT_SEED = 20260722` from the 20 `neutral`, **outside the confirmatory set**. The 20 `high` vignettes are untouched.

**Instrument as frozen:** λ = 0.1, target `u_gain`, band-minimum `cos_l` = 0.8201, `ρ_l` from `rho_layers.json`, injecting hook verified ALL_PASS at `5453270`.

> **Bottom line: the pre-declared k-selection rule returns `rule_satisfiable = False` and `three_distinct_doses = False`. The design's dose axis does not exist on this instrument. The confirmatory block was NOT started.** Reasons below, in the order they bind.

---

## 1. The frozen ladder: five of six rungs are 100 % malformed

| k | mean Set F loading | × natural (0.0825) | **malformed rate** |
|---|---|---|---|
| *α = 0 reference* | **0.1595** | 1.9× | 0 % |
| **0.05** | 27.372 | 331.8× | **0 %** |
| 0.1 | 98.163 | 1189.9× | **100 %** |
| 0.2 | 165.229 | 2002.8× | **100 %** |
| 0.4 | 238.622 | 2892.4× | **100 %** |
| 0.8 | 272.980 | 3308.8× | **100 %** |
| 1.6 | 298.345 | 3616.3× | **100 %** |

Applying §3.4 mechanically: `k_max` = **0.05** (the only rung under the 10 % malformed bar); the rung first reaching the 50× target is also 0.05; therefore **k₁ = k₂ = k₃ = 0.05** — *one* dose, not three. §3.4's registered fallback anticipated a collapse to *fewer than three* doses; this is a collapse to **one**, which leaves the primary test **T1 undefined** (a per-vignette slope on dose score needs ≥3 dose levels to be a trend rather than a two-point contrast).

## 2. The registered targets do not live on the ladder at all

Targets were 2× / 10× / 50× of the natural without-mention level 0.0825 → **0.165 / 0.825 / 4.125**. The **lowest frozen rung already sits at 27.4 = 332× natural**, i.e. **all three targets lie below the ladder's floor**. The ladder was fixed pre-pilot (correctly, to stop the pilot choosing it) and was mis-scaled by roughly **two orders of magnitude**.

**An extended downward sweep** (exploratory, 2 reps/rung, labelled as such in the JSON) locates the problem but does not solve it:

| k | 0.0002 | 0.0005 | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|---|---|---|
| mean F | 0.583 | 0.245 | 0.491 | 0.532 | 0.744 | 0.593 | 1.416 |
| × natural | 7.1× | 3.0× | 6.0× | 6.4× | 9.0× | 7.2× | 17.2× |
| malformed | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |

Two things to read off this. **(a) It is not monotone.** Across a 100× span of α (k = 0.0002 → 0.02) the readout moves only ~2.4×, non-monotonically — that band is dominated by run-to-run noise, not by dose. **(b) There is a cliff, not a curve.** From k = 0.02 to k = 0.05 (2.5× in α) the readout jumps **1.42 → 27.4 (19×)**, and by k = 0.1 the model is 100 % malformed. The whole transition from "indistinguishable from noise" to "totally degraded" occupies **less than one decade of α**.

## 3. The content-transfer caveat has materialized, quantitatively

The α = 0 reference **on the pilot's `neutral` material is 0.1595** — already **1.9× the 0.0825 anchor**, which was measured on `high`/`flagged` material (`RESULTS_PHASE1.md` App. A1). So the registered **2× target (0.165) is, to two decimal places, the untreated baseline of the calibration material.** The §3.4 content-transfer caveat the PI registered ("calibration transfers across a content difference; the transfer is assumed, not verified") is now **falsified as an assumption**: the target scale does not transfer, and the two lowest registered targets are not distinguishable from doing nothing.

## 4. The decisive finding: the intervention forces tokens, it does not modulate a state

Sample generations at fixed seed on v48 (`phase2/data/pilot_sample_texts.json`, verbatim excerpts):

- **k = 0 —** *"…es importante analizar cuidadosamente los síntomas presentados y compararlos con los criterios sugeridos para esta categoría."* Normal clinical evaluation.
- **k = 0.02 —** *"Aunque esta categoría es ficticia para el estudio, podemos usarla como un marco conceptual…"* Still fully coherent; fiction acknowledged in ordinary language.
- **k = 0.05 —** *"es importante experimentar con la experimentación teórica y conceptual… no ha sido reconocida como una condición clínica experimentada o experimentando experimentación…"* **Lexical intrusion.** Set F surface forms are being forced into the output; the sentence has stopped meaning anything.
- **k = 0.1 —** *"fictionfictionfictionfiction…"* ×183. Total collapse.

**This is the finding that governs.** The addition is not turning up a semantic representation of fictional status; past a threshold it **forces Set F tokens into the output distribution**. The readout climb from 27× to 3616× natural is therefore substantially *emission forcing* measured by an estimator that reads out over token logits — the readout and the artefact are the same thing.

Crucially, **k = 0.05 passes the mechanical malformed test at 0 %** while already showing clear intrusion. The §7 detector catches empty output, non-UTF-8, and ≤10-token n-gram repetition >50 %; it does **not** catch fluent-but-vacuous vocabulary flooding. **The one dose the rule selects is already inside the confound the study exists to avoid** — any diagnosis drop at k = 0.05 would be attributable to the model being derailed into F-vocabulary, not to holding fictional status. That is precisely the degradation alternative §5's drop-vs-degradation rubric was registered to discriminate, arriving *before* any usable dose.

## 5. Why the confirmatory block was not started

The PI's instruction was to proceed pilot → smoke gate → confirmatory 700 → STOP. The pilot is the registered gate on exactly that step, and it has failed on its own pre-declared rule. Starting 700 runs (~2.5 h GPU) would mean:

1. **T1, the primary test, is undefined** — one dose level, no trend.
2. **The only surviving dose is confounded** by lexical intrusion (§4), so even the T2 contrast at that dose could not distinguish direction-specific behaviour change from vocabulary derailment.
3. **The dose scale is not anchored** — the registered targets sit at or below the untreated baseline of the calibration material (§3).

Spending the budget would produce a data commit that cannot answer the registered question. **Per the standing rule, the delegate does not re-scale a frozen, tagged parameter and proceed;** the ladder, the targets and their 0.0825 anchor are all prereg-frozen text. This goes back to the PI.

## 6. What the PI now has to decide

The instrument works — the hook is verified, the landing is good (`cos_l` ≥ 0.82), the dose is well-scaled per layer. What fails is the **assumption that a norm-scaled additive dose has a usable regime** between "invisible" and "destructive" for this direction on this model. Options, none taken by the delegate:

- **(a) Re-scale and re-anchor, as a dated pre-data amendment.** New ladder inside the cliff, e.g. `k ∈ {0.02, 0.03, 0.04, 0.05}`, with targets re-anchored to the **`high`-material α=0 baseline** rather than to 0.0825, and the malformed detector extended to catch vocabulary flooding (e.g. Set F token share of the generation). Note the honest cost: the usable dynamic range is under one decade, so a "dose-response" over it is a weak instrument.
- **(b) Change the DV for the manipulation check** so the readout is not measured on the same token logits the intervention forces — the semi-independent base-model readout (§6 check 2) is the natural candidate, since it shares the residual but not the readout head.
- **(c) Change the intervention** — e.g. clamp/project rather than add, or add at a single layer rather than re-adding across ten, to buy a wider usable regime.
- **(d) Report Phase 2 as an instrument-negative result**: on this model, with this lens-inverse direction, additive intensity has no regime in which the F representation is driven well above baseline without the output degrading. Given Phase 1's decoupling result, that is itself a publishable limit on what activation addition can test here.

**No option is executable without the PI, because each changes prereg-frozen text.** Both possible outcomes of the study remain registered as informative (§5); this pilot result is about the *instrument*, not about either outcome.

---

*Generated by `phase2/scripts/run_pilot.py` against tag `prereg-phase2-v1` / commit `eb176a9`. Pilot material is held-out `neutral`; the 20 confirmatory `high` vignettes remain untouched and unjudged. No confirmatory data exists. Interpretation is the PI's.*
