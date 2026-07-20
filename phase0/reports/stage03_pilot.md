# Stage 0.3 — Feasibility Pilot (vignette v12)

**Date:** 2026-07-20 · **Machine:** `[host]` (RTX 5090) · **Author:** Claude Code (Phase 0 delegate)
**Status:** ✅ Complete — **STOP POINT 3, END OF PHASE 0.**
**Scope:** *Feasibility only.* No interpretation of readouts, no diagnostic-token counting, no comparisons — those are reserved for the PI / Phase 1.
**Script:** `phase0/scripts/stage03_pilot.py` · **Log:** `phase0/logs/20260720T092551_stage03_pilot.log`
**Raw readouts (gitignored):** `phase0/data/pilot_readouts/v12_readouts.json` (+ `v12_meta.json`)

---

## 0. Stimulus & run configuration

| Item | Value |
|---|---|
| Vignette | **v12** — `phase0/data/stimuli_pilot/v12.txt`, sha256 `32c78f5f…19d0e`, 185 chars |
| Origin | *provided by PI in chat, copied verbatim from reification-gradient `materials/vignettes.yaml` (post-1b set of 60), id v12, target_compatibility: high.* The reification-gradient repo is **not** on this machine; text supplied by PI directly. Only the `text` field was copied (no metadata). |
| Instruction (exact, per brief) | `Read the following case and reason about what might be going on with this person.` |
| Prompt assembly | Qwen chat template, single user turn = instruction + `\n\n` + vignette, `add_generation_prompt=True` |
| Model / lens | `Qwen/Qwen2.5-7B-Instruct@a09a3545` / `neuronpedia/jacobian-lens@16a01f3` (`qwen2.5-7b-it`, 27 fitted layers) |
| Decoding | greedy (`do_sample=False`), deterministic; seed 0 recorded; `max_new_tokens=200` |

The instruction is **English** while the vignette is **Spanish** — noted here as a fact of the run (Stage 0.1b flagged this); the model generated in Spanish.

---

## 1. Did extraction work end to end? — **Yes**

Full path executed without intervention: chat prompt → `model.generate` (200 new tokens) → single forward pass over the **full prompt+generation sequence** under `ActivationRecorder` → per-layer `J_l` transport → unembed → top-k dump. Readouts were extracted at **every fitted layer (0–26) and every absolute position** of the 298-token sequence, plus the model's own final-layer output row (`kind="model_output"`) as reference.

The dump marks, per row: absolute `position`, `segment` (`prompt` vs `generation`), and `ood_unfitted_pos` (positions < 16, the unfitted region per issue #5 Pitfall 2) — so the downstream prompt/generation split the brief requires is mechanically available.

*(Two measurement-neutral harness fixes were needed on Windows and are noted for the record: forcing UTF-8 stdout for printing non-cp1252 vocab tokens, and assembling the chat prompt via `tokenize=False` + a second tokenize call because transformers 5.14's `apply_chat_template(return_tensors="pt")` returns a `BatchEncoding`, not a bare tensor. Neither touches the measurement.)*

## 2. Data volume & runtime (per vignette)

| Metric | Value |
|---|---|
| Prompt / generation / total positions | 98 / 200 / **298** |
| Rows dumped | **8,344** (27 lens layers × 298 + 1 model_output layer × 298) |
| Readout file size | **5.25 MB** JSON (top-k=10, token + weight per cell) |
| Generation time | 4.7 s |
| Readout extraction time | 1.6 s |
| **Compute per vignette** | **≈ 6.3 s** (+ one-time ~6 s model load, amortized across a batch) |
| VRAM peak | **15.63 GB** |

## 3. Are readouts non-degenerate? — **Yes** (varied tokens, not noise)

Unique top-1 tokens across the 298 positions, by layer (a constant/degenerate readout would be near 1):

| Layer | L2 | L8 | L15 | L20 | L26 |
|---|---|---|---|---|---|
| unique top-1 / 298 | 90 | 112 | 162 | 188 | **202** |

Variety rises with depth, as expected for a working lens. **Three verbatim snippets** (raw `(token, weight)`, top-6), presented purely as evidence of non-degeneracy — **not interpreted**:

```
L20 pos40  [prompt]:      [(':".$', 11.09), (":'.$", 10.86), ('%X', 9.98), ('并不意味', 9.87), ('datable', 9.16), ('nika', 9.12)]
L15 pos150 [generation]:  [(' psychological', 14.02), (' emotional', 13.91), (' significant', 13.13), (' possibly', 12.95), (' potentially', 12.94), (' feeling', 12.41)]
L24 pos150 [generation]:  [(' Emotional', 15.54), (' nostalgia', 13.16), (' Adjustment', 12.80), (' Anxiety', 12.53), (' Psychological', 12.44), (' Transition', 12.38)]
```

Readouts are well-formed vocabulary tokens with a spread of weights, and differ across layer/position/segment — i.e. non-degenerate. Prompt-region readouts (e.g. L20 pos40) are more diffuse/code-like; generation-region readouts are sharper and more topical. **No claim is made here about what any token means for any hypothesis.**

The generation itself was coherent on-topic Spanish reasoning (200 tokens; full text stored in the dump), confirming the model ran normally under the pilot instruction.

## 4. Technical obstacles for scaling (to a multi-condition, multi-run design)

Engineering notes only — **not design or hypothesis proposals**:

1. **VRAM windows are the binding constraint.** Peak 15.63 GB coexists with light desktop apps in 32 GB but **cannot** share the card with a ≥24 GB Ollama model. During this session `mistral-sim` (24 GB, `keep_alive=Forever`) **auto-reloaded on its own** (residual process/health-check hitting Ollama re-pins it), repeatedly closing the window. A batch run needs either exclusive GPU time or a way to hold that keep-alive off; ad-hoc windows will not survive a 1000+-run design.
2. **Storage scales linearly and is non-trivial.** ~5.25 MB/run at 200 gen tokens. A fully-crossed design (e.g. 4 conditions × 5 wrappers × 60 vignettes = 1,200 cells) × R runs ≈ **6.3 GB × R** of raw readouts. Fine on this disk (154 GB free) but needs a retention/compaction plan (parquet + top-k pruning would shrink it).
3. **Compute is tractable.** ~6.3 s/run compute → ~2.1 h per full 1,200-run replicate (excluding load). Model load (~6 s) should be amortized by keeping the model resident across a batch rather than per-run.
4. **Sequence length drives both volume and the OOD fraction.** `max_new_tokens` was fixed at 200 here; variable generation length will vary row counts and the prompt/first-16 OOD proportion per run — the dump already records the split so this is trackable.
5. **Determinism holds** (greedy, seed recorded), so runs are reproducible; any move to sampling would need seed control per cell.
6. **Windows harness gotchas** (UTF-8 stdout; `apply_chat_template` return type) are now handled in the pilot script and should be carried into any batch harness.

---

## 5. What this pilot does NOT do

No interpretation of any readout with respect to any hypothesis; no counting of diagnostic tokens; no comparison of conditions, positions, or layers beyond the non-degeneracy variety check above; no token-set selection (rule 4). The vignette content is treated as opaque. Phase 1 design, token-set freezing, and preregistration are out of scope and reserved for the PI.

**END OF PHASE 0.**
