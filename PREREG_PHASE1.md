# PREREGISTRATION — Phase 1 representational study (DRAFT for PI freeze)

**Status:** DRAFT produced by delegate at Stage P0. **Not frozen.** Becomes the preregistration when the PI reviews/edits and personally runs `git tag -a prereg-phase1-v1`. Stages P1/P2 are gated on that tag.
**Builds on the CLOSED token-set instrument** (seal chain, each byte-verified against its committed blob): A0 `3689ac85d4e61500357881569de0cf1feeb74c43d5712c17d25baa1d8539634f` → A1 bilingual `9530aceb8a982a2af931c4e513abb0bd3b2d5a62f6329bdcdec70394f2778f73` → note C-EN `cfce47427c24eeaf90bdc420191f35a7d54771de06bc5ecc6104c643e62a058f`.
**No condition-bearing readout exists at prereg time.** Only Stage 0.2 calibration, the Stage 0.3 v12 pilot, and the 20-rep condition-free technical calibration — none with conditions, token-set scoring, or counting. The power analysis below uses only that condition-free calibration data.

**Model/lens (digests re-verified at run time):** `Qwen/Qwen2.5-7B-Instruct @ a09a3545`; lens `neuronpedia/jacobian-lens @ 16a01f3` (`qwen2.5-7b-it`, 27 fitted layers), lens `.pt` sha256 `3b3ab44c…cba29`. Judge: `gemma2:27b` (this host's Ollama; digest re-verified).

---

## 1. Design (two primary contrasts; cells inherited from the behavioral study)

Both contrasts are **within-item** (the same 20 `high` vignettes appear in both cells of a contrast), analyzed paired-by-vignette.

- **C1 — recognition-without-consequence.** Cells `DN_flagged × high × L1` vs `DN_plausible × high × L1`. **DV: Set F loading** (generation positions). **Registered discovery question, no directional bet:** *loaded-but-inert* (F loads in flagged > plausible while the model diagnoses) vs *never-enters* (F ≈ plausible floor). Both architectures named; neither favored.
- **C2 — advocacy over the absurd.** Cells `incoherent × high × L4` vs `incoherent × high × L1`. **DV: Set A loading.** **Directional hypothesis:** advocacy (L4, high-authority) *increases* clinical-framing load (Set A) in the workspace relative to L1. (Motivated by the behavioral study's largest delta on these cells, 0.92 vs 0.16 — a behavioral fact, cited for direction only; the representational effect size is not assumed.)

**Vignettes.** The 20 `high` (`target_compatibility: high`) vignettes from the sealed 60-vignette set. **Not on this host** (the reification-gradient repo is absent); the PI must provide the 20 vignette texts read-only, with recorded provenance exactly as v12 was (source string + sha256 per vignette, copied to gitignored `phase0/data/`).

**Stimulus assembly.** Built exactly as the behavioral runner did: `wrapper (L1/L4) × condition text (DN_flagged / DN_plausible / incoherent) × vignette`, then the fixed task instruction. **Byte-identity of payloads across the two cells of each contrast except the single manipulated element** will be verified and documented per contrast (C1: only the DN_flagged disclosure line differs — the sealed ~23-token length asymmetry, see §5; C2: only the wrapper L4↔L1 differs).

**Generation.** Temperature 0.7, `num_predict = 200`, greedy-off (sampling); **all seeds recorded**; identical instruction wording across all cells. Same instruction template as the pilot/behavioral task.

## 2. DV and aggregation (preregistered)

**Loading of set S in a run** = for each layer `L` in the **primary band 17–26**, the **mean over GENERATION positions** (R4: generation-only, and never positions 0–15) of the **summed readout weights of S's operative tokens** present in the top-k dump at `(L, position)`, matched by **folded form (R3)**, using **SURVIVORS only** (R2 echoes are barred from confirmatory sets); then the **mean over the band**. Operative lists are the sealed A1 survivors: **Set A = 19 tokens** (17 EN + 2 ES: ` paciente`, ` tratamiento`), **Set F = 11 tokens** (9 EN + 2 EN+ES: `experimental`).

- **Length-confound mitigation (R5):** generation-only epoch + **mean (not sum)** over positions. Stated here per the seal.
- **Layer bands:** 17–26 primary (justification: Stage 0.2 Tier 2 currency/Italy action and the Stage 0.3 pilot depth-variety both live there); **all-layer profiles exploratory.**
- **Language-tagged breakdown reported alongside every aggregate** (per A1 and the C-note auxiliary diagnostic).
- **Measure property (documented):** dumps are top-k=10, so a run's loading captures an operative token's weight only when it ranks in the top-10 at that `(L, position)`. This is a fixed, pre-declared property of the instrument, identical across all cells.

## 3. Diagnosis conditioning (C1)

After the J-lens run, `gemma2:27b` (this host's Ollama, digest re-verified) judges `diagnosis ∈ {0,1}` on each generated text per the **validated v1 rubric**. **C1's confirmatory analysis conditions on `diagnosis = 1` runs** (expected ≈ all in flagged-high, per the behavioral 120/120 recognition-without-consequence cell); the **diagnosis rate is reported** for every cell. C2 does not condition on diagnosis (its DV is clinical-framing load, not a diagnosis gate), but the rate is reported.

## 4. Power and R (from real between-rep variance)

Estimated from the **20-rep condition-free nightly calibration** (v12) under the exact §2 aggregation (`phase0/scripts/phase1_p0_power.py`, `phase0/data/phase1_p0_power.json`; RNG seed 0):

| Set | mean loading | sd (between-rep) | CV | lang split |
|---|---|---|---|---|
| A | 0.1069 | 0.0698 | 0.65 | **EN 0.1066 / ES 0.0003** |
| F | 0.0233 | 0.0182 | 0.78 | (near-floor on non-disclosure v12, expected) |

**Power model:** within-item paired t-test over 20 vignettes; R reps averaged per (vignette, cell); per-vignette paired difference ~ N(effect, 2σ²_rep/R) assuming negligible vignette×cell interaction; df=19; α=0.025 (Bonferroni/2 of 0.05). **C2 one-sided** (directional), **C1 two-sided** (discovery). Power is scale-free (depends on standardized effect δ = effect/σ_rep and R); Monte-Carlo, 40 000 sims/cell.

Power at δ=0.5 (medium): **C1 two-sided** R=3/5/8/10/15 = 0.63 / **0.86** / 0.97 / 0.99 / 1.00; **C2 one-sided** = 0.74 / **0.92** / 0.99 / 1.00 / 1.00. (Full grid over δ∈{0.1…1.0} in the JSON.)

**Pre-fixed rule → R = 5** (smallest R with power ≥ 0.80 at the medium effect in BOTH contrasts, within ≤3h GPU: ~42 min lens + ~30 min judge ≈ 1.2 h).

> **⚠ Delegate caveat for the freeze (not a decision):** the calibration has **one** vignette, so **vignette×cell interaction variance is unestimated** — the R from the rule is a **FLOOR**. The confirmatory test itself remains valid (it uses the empirical SD of the 20 real per-vignette differences, which captures whatever variance exists), but the a-priori R could be under-provisioned if interaction is non-trivial. The 3h budget comfortably fits **R=10 (~1.9 h)** or **R=15 (~2.6 h)**; the PI may elect a higher R at freeze for robustness. Delegate recommendation: **R=10**. The PI sets the final R.

## 5. Hypotheses and α structure

- **C2 (directional):** H1 = loading(Set A | incoherent×high×L4) > loading(Set A | incoherent×high×L1). One-sided, α=0.025.
- **C1 (registered discovery, no directional bet):** two-sided test of loading(Set F | flagged) − loading(Set F | plausible) among diagnosis=1 runs. Interpretation registered as *loaded-but-inert* (positive, F loads during diagnosis) vs *never-enters* (≈0, F at floor). α=0.025 two-sided.
- Two tests, Bonferroni-split from 0.05 → **α = 0.025 each**.

**Asymmetric-informativeness clause (verbatim from the seal, R5):** *"the J-lens captures the workspace incompletely; null loadings are non-conclusive, only positive loadings inform."* A null in either contrast (including anything involving Set C's EN-only realization) is non-conclusive; only positive loadings inform.

**Auxiliary diagnostic (registered in the C-note, reported not tested):** per-language loading breakdown. Pre-data, condition-free calibration already shows Set A load is ~99.7% English (EN 0.1066 vs ES 0.0003) despite Spanish generation — consistent with the C-note prediction that the workspace realizes these concepts in English tokens under Spanish context. This is measurement characterization, not a confirmatory result.

## 6. Storage, exclusions, provenance

- **Storage:** full top-k=10 dumps (as in the pilot: all layers × all positions, prompt/generation marked, first-16 OOD flagged) for the **4 confirmatory cells only** — ~4–8 GB acceptable. Nothing else is run.
- **Exclusion rules (pre-declared):** (a) malformed/empty/degenerate generations (e.g., zero generated tokens, or non-terminating repetition detected mechanically) excluded and counted; (b) judge failures (unparseable JSON / rubric error) excluded from diagnosis conditioning and counted; (c) any run whose recorded model/lens/tokenizer digest fails re-verification is discarded and re-run. All exclusions logged; N reported exactly.
- **Determinism/reproducibility:** seeds recorded per run; digests (model, lens sha, gemma) re-verified and recorded; dump integrity (row counts) checked.

## 7. Open at prereg / not fixed here

Final R (PI at freeze; rule gives 5, delegate recommends 10); vignette provision + provenance; any exploratory (non-confirmatory) all-layer or non-primary-band reporting; the analysis itself (a separate later prompt — no aggregates or loadings are computed during the runs).

---

*Generated at Stage P0. On freeze, the PI edits as needed, then commits and runs `git tag -a prereg-phase1-v1 -m "phase 1 preregistration freeze"` + push. The delegate does not tag.*
