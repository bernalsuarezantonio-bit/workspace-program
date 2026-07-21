# Post-hoc EXPLORATORY analysis — C1 generation self-echo confound

**Status: EXPLORATORY, POST-HOC. NOT part of the sealed preregistration** (`prereg-phase1-v1`,
`PREREG_PHASE1.md` sha256 `bedbcc78…`). Specified by the PI on **2026-07-21**, recorded here
**before any loading was computed** (the 800-run data generation had just started; no unblinded
result existed at registration time). It will run **only after** the confirmatory A1–A4 pass, in the
report's fenced EXPLORATORY section, and it changes **nothing** in the confirmatory adjudication
(C1 adjudicates on the sealed mean-loading test).

**Motivation (PI, verbatim framing):** generation self-echo confound in C1 — the worry that Set F
loading in `flagged` runs is driven by the model *emitting* fiction-status tokens in its own text
(the disclosure being echoed/paraphrased), rather than fiction-status being *sustained in the
workspace* independent of emission. This analysis probes whether the C1 signal survives removing
the emission channel.

All loadings use the **sealed aggregation** (band 17–26, generation positions only per R4, R3
language-folded matching, operative token lists from `phase1_token_sets_SEALED.md` only). Set F and
Set A operative lists are the sealed A1 survivors.

## Registered exploratory sub-analyses (verbatim intent)

1. **Textual-mention split.** Among `flagged × diagnosis=1` runs, classify each by whether its
   generated text explicitly mentions fictional status — regex over `inventad`, `estudio`,
   `no reconocid*`, `fictici*` (case-insensitive, on the generated text only). Report the split
   (n with-mention / n without-mention).

2. **F-loading by mention stratum.** Compare Set F loading (same sealed aggregation) across three
   groups: `flagged`-WITH-mention, `flagged`-WITHOUT-mention, and `plausible`. **Decisive cell =
   `flagged`-WITHOUT-mention:** if F loads above `plausible` there too, fiction-status is sustained
   without being said (reinforces *loaded-but-inert*); if F-without-mention ≈ `plausible`, the trace
   is of the *saying*. Numbers only.

3. **Positional-masking robustness.** Recompute Set F loading EXCLUDING generation positions whose
   emitted token (or a ±2 window around it) is itself an F operative token. If the
   `flagged > plausible` difference survives the mask, the sustainment is independent of emission.

4. **Same mask on C2 / Set A** as a check that the confound does not also inflate that contrast:
   recompute the C2 Set A `incoherent×L4` vs `incoherent×L1` difference under the identical
   positional mask.

**Output:** numbers and the report only. Interpretation is the PI's.
