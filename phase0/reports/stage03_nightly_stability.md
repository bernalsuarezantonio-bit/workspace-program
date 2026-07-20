# Stage 0.3 — Nightly Technical Calibration (pipeline variance)

**Date:** 2026-07-20 (run 12:10–12:12 UTC) · **Machine:** `[host]` (RTX 5090)
**Scope:** **Technical variance of the pipeline only** — no experimental conditions, no token sets, no counting, no comparison of readout content. This report describes volume, runtime, and structure. It says **nothing** about what the readouts contain.
**Script:** `phase0/scripts/stage03_nightly_calibration.py` · **Log:** `phase0/logs/20260720T092551_stage03_nightly.log`
**Dumps (gitignored):** `phase0/data/pilot_readouts/nightly/v12_rep01..20.json` · summary `…/nightly_summary.json`

---

## Configuration

Identical to the Stage 0.3 pilot except for sampling: vignette **v12** + the same instruction (`Read the following case…`), **20 repetitions, seeds 1–20**, **temperature 0.7** (`do_sample=True`; the Phase-1 sampling temperature), `max_new_tokens=200`. Prompt is byte-identical every rep (98 tokens); only the sampling seed varies. Readouts dumped exactly as the pilot (all 27 fitted layers + model-output row × all positions, `prompt`/`generation` segment and `<16` OOD flag marked).

## Completion & contention

- **20 / 20 reps completed. No pause. No VRAM contention.** Device-free stayed ~16.5 GB before every rep (the run holds ~15.6 GB, so nothing else could claim the card mid-run). `mistral-sim` did **not** reappear during the 12:10–12:12 UTC window.
- Per-rep start timestamps + device-free are recorded in `nightly_summary.json` (orphan-process-hunt data, per protocol). Nothing to report there this run.

## Volume stability

| metric | min | max | mean | std |
|---|---|---|---|---|
| rows / rep | 8,344 | 8,344 | 8,344 | **0** |
| file size (MB) | 5.230 | 5.255 | 5.240 | 0.006 |

- Row count is **exactly constant** (8,344 = 298 positions × 28 layers [27 lens + 1 model-output]); file size varies only by ~±0.5 % from token-string byte lengths in the JSON.
- **Total on disk: 104.8 MB for 20 reps** (~5.24 MB/rep) — validates the dump format at 20× scale; JSON is well-behaved and linear.

## Runtime stability (per rep)

| phase | min | max | mean | std |
|---|---|---|---|---|
| generation (s) | 4.24 | 6.12 | 4.50 | 0.43 |
| readout extraction (s) | 1.34 | 1.81 | 1.48 | 0.12 |
| VRAM peak (GB) | 15.63 | 15.63 | 15.63 | **0** |

- ~**6 s/rep** steady-state (gen + readout). The single 6.12 s generation is **rep 1 (warm-up)**; reps 2–20 are 4.2–4.5 s. VRAM peak is bit-for-bit constant.
- Model loaded once (~8 s) and kept resident across all 20 reps — the correct amortization pattern for batch scaling.

## Structural note / anomaly

- **Every rep generated exactly 200 tokens** (`gen_tokens` std = 0): all 20 hit the `max_new_tokens` cap; none terminated early on EOS. Consequently volume is **cap-determined, not length-determined**, and the near-zero volume variance above reflects the fixed cap rather than intrinsic run-to-run length stability. **Implication for storage planning:** a Phase-1 configuration that allows natural EOS termination (or a different cap) would produce variable-length runs and hence variable per-run volume; this run therefore validates the *format and its per-token scaling*, not the *distribution of generation lengths*. Flagged for the prereg storage decision.
- No other anomalies: no OOM, no NaN/inf, no dump failures, no encoding errors (UTF-8 dumps clean), no device errors across 20 consecutive reps.

## Bottom line (technical)

The pipeline is technically stable across 20 seeded reps at temperature 0.7: constant VRAM, ~6 s/rep steady-state, linear and well-formed storage at ~5.24 MB/rep (104.8 MB/20). The one thing to carry to the prereg is that generation length here was cap-bound, so volume estimates for a variable-length Phase-1 configuration should be derived from a length distribution rather than from this fixed-cap run. **No statement is made about readout content.**
