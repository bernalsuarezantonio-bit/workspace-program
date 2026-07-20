# Stage 0.2 — Environment check + pipeline validation

**Date:** 2026-07-20
**Machine:** `[host]` — Windows 11 Pro 10.0.26200, RTX 5090 (the designated GPU host)
**Author:** Claude Code (Phase 0 delegate)
**Status:** ⛔ **BLOCKED mid-stage — awaiting two PI inputs before the heavy half.** Not STOP POINT 2 yet.
**Log:** `phase0/logs/20260720T092551_stage02_env.log`
**jacobian-lens commit (re-vendored here):** `581d398613e5602a5af361e1c34d3a92ea82ba8e` ("Initial release") ✓ matches pin

Nothing was loaded into VRAM. No model or lens was downloaded. No venv was built. The only writes were a **gitignored** clone of `jacobian-lens` at the pinned commit and this report/log/PROVENANCE. I stopped exactly where the brief tells me to: *"Environment first, before downloading anything heavy"* and *"wait for the PI to confirm the [VRAM] window before loading anything."* Both conditions are live.

---

## 0. Headline — the hardware premise now holds, but two gates block the heavy half

Unlike the Stage 0.1 iMac, **this is the real GPU host**: RTX 5090, driver present, CUDA-capable, cu128 wheels available. The pipeline can run here. But I cannot complete the validation right now because of two things, **both of which the brief reserves for you**:

1. **VRAM is contended and pinned.** `ollama ps` shows `mistral-sim:latest` (24 GB) pinned `UNTIL: Forever`, and the card is at **93 % util, 570 W, 74 °C, ~2.9 GB free of 32.6 GB**. The J-lens 7B in fp16 needs ~15 GB. It will not co-reside with a Forever-pinned 24 GB model. The brief: *"if any model is loaded/pinned, report it and wait for the PI to confirm the window before loading anything."* → **Reported; waiting.**
2. **Substrate decision (Windows-native vs WSL) is yours.** The brief: *"Report your recommendation with reasons; the PI decides if it means extra setup."* My recommendation is below. Building the venv and downloading ~15 GB commits to a substrate, so I'm holding until you confirm.

Everything in §§1–3 is done and green. §§4–6 are the blocked remainder, with the exact unblock conditions spelled out.

---

## 1. Environment verdict

| Check | Result |
|---|---|
| OS | Windows 11 Pro 10.0.26200 (`[host]`); MSYS/MinGW shell available |
| GPU | **NVIDIA GeForce RTX 5090**, WDDM, present |
| NVIDIA driver | **610.47**; `nvidia-smi` works |
| CUDA capability (driver UMD) | **13.3** — comfortably ≥ the 12.8 floor (driver is forward-compatible with cu128 runtime) |
| CUDA toolkit (`nvcc`) | Not installed — **expected and fine**; cu128 PyTorch wheels ship their own runtime |
| Python (native) | **3.12.10** (`Programs\Python\Python312`) — satisfies jlens `requires-python >=3.10` |
| `py` launcher default | 3.14.4 — **too new for the ML wheel stack; will not use** |
| uv | 0.11.7 (windows-msvc) |
| git | 2.53.0.windows.3 |
| Disk (C:) | 154 GB free of 931 GB — enough for model + lens + venv |
| `torch.cuda.is_available()` in pinned cu128 venv | **Not yet tested** — this requires building the venv, which is gated on the substrate decision (§2, §4) |

### cu128 wheels natively available for this Python? — **Yes.**

Queried `https://download.pytorch.org/whl/cu128/torch/` directly:

```
torch 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1, 2.10.0, 2.11.0   (+cu128, cp312, win_amd64)
```

So the native-Windows path is fully viable for Python 3.12 — no WSL is *required* to get cu128 torch onto this box. (Blackwell / sm_120 support landed in the cu128 line from torch 2.7 onward, which matches the wheels above.)

### Native vs WSL — recommendation: **Windows-native**, with WSL as fallback only

| Factor | Windows-native | WSL2 |
|---|---|---|
| cu128 torch wheels | ✅ present (cp312 win_amd64, confirmed) | ✅ present (Linux is torch's reference platform) |
| Setup cost on *this* box | none beyond a venv | **the only WSL distro here is `docker-desktop`** — a general Linux distro must be installed first |
| Workload shape | apply-only: forward pass + per-layer `[d_model,d_model]` matmul (recon §5) — no autograd, no Triton/flash-attn needed. Portable. | same, but through the WDDM GPU-passthrough layer |
| Ollama coexistence | Ollama runs **natively on Windows** here (`llama-server.exe`) — coordination happens on one OS | WSL adds a second VRAM consumer sharing the same physical card via the host driver |
| Ecosystem reliability (general) | occasional Windows gaps (Triton wheels, some fused kernels) — **not on our apply-only path** | the battle-tested default for the HF/torch stack |

**Why native:** the pilot is *apply-only* (pre-fitted lens ⇒ no fitting, no autograd graph, no Triton/flash-attention). That removes the usual reasons to prefer Linux. cu128 Windows wheels exist for cp312; Ollama already lives on Windows so VRAM coordination stays on one OS; and WSL here would mean installing a distro (extra setup, extra passthrough layer, second VRAM claimant). The general "Linux is more reliable for ML" argument is real but bites hardest on *fitting*/Triton paths we don't touch.

**When I'd switch to WSL:** if, once we build, the apply path hits a Windows-specific dependency gap (e.g. a transformers/torch code path that pulls Triton or an unbuilt kernel). I'll surface it rather than silently paper over it. **Your call** — it means extra setup (a distro install) either way.

---

## 2. VRAM coordination (Ollama) — the blocking coexistence fact

```
$ ollama ps
NAME                 SIZE   PROCESSOR   CONTEXT   UNTIL
mistral-sim:latest   24 GB  100% GPU    8192      Forever
```

- **Pinned `Forever`** (`keep_alive = -1`) and the GPU is **actively at 93 % util / 570 W** — another study is mid-run.
- Free VRAM ≈ **2.9 GB**; the 7B fp16 J-lens run needs ~15 GB. **No coexistence is possible in this state.**
- Options for you (I'm not choosing): (a) give me a window where `mistral-sim` is unloaded/unpinned; (b) authorize me to request its unload; or (c) accept a smaller-footprint plan (e.g. load the 7B in a lower precision / offload) — but that changes the pilot spec and is your decision, not mine.

On-disk Ollama models (context): `qwen2.5:32b` (19 GB) — the family the behavioural study used — plus `llama3.3:70b`, `gemma2:27b`, several `mistral-small` 24b, `phi4:14b`, `nomic-embed-text`.

---

## 3. Safe prep already done (decision-independent, no VRAM)

- **Re-vendored `jacobian-lens`** into `vendor/` (gitignored) and checked out the pinned commit `581d398…` — verified HEAD + subject match PROVENANCE.
- **Inspected the repo** to pin down exactly what "reproduce one example" means here (§4). `pyproject` deps: `torch, huggingface_hub, transformers>=5.5, numpy` (dev: `pytest, ruff, datasets`).
- **Portability note for the build:** `torch` is unconstrained in `pyproject`, and the committed `uv.lock` was resolved on the **iMac (arm64-macOS)** — its torch pin will not carry cu128-win wheels. On this host torch must be installed from the **cu128 index**; the exact resolved version will be pinned into PROVENANCE at venv-build time. The recon's noted `.cuda()` hard-coding in the walkthrough is fine here (this *is* CUDA) — no MPS port needed.

---

## 4. Pipeline validation — the reproduction, staged (BLOCKED)

I identified a two-tier reproduction. This matters because Tier 1 needs no GPU at all, so it can validate the library's mechanics even while VRAM is contended — but it does **not** by itself satisfy the brief's success criterion.

| Tier | What | VRAM | Satisfies brief's success criterion? |
|---|---|---|---|
| **1** | `pytest` against the **CPU-only `TinyDecoder`** (`tests/tiny.py`) — exercises the real `jacobian_for_prompt` / `ActivationRecorder` / `compute_slice` code paths | none (CPU) | **No** — asserts numeric properties (late-layer `diag(J)≈1`), not the documented *qualitative* readouts |
| **2** | The **documented demo**: `examples.py` `multihop` currency prompt *"…the country shaped like a boot is"* → expect euro/lira at position −2; and/or the ascii-face **"nose"** readout at mid layers | ~15 GB (7B + lens) | **Yes** — this is *"readouts qualitatively match the documented expected output"* |

Neither has been run. Tier 1 needs a venv (gated on §2 substrate). Tier 2 needs a venv **and** a VRAM window. The standing **`torch.isfinite()` guard** on the lens (PROVENANCE / upstream issue #6) will run immediately after lens load, before any use, as part of Tier 2 — abort-and-report on failure.

---

## 5. What unblocks the rest (precise)

Give me these two and I finish Stage 0.2 to STOP POINT 2 without further questions:

1. **Substrate:** confirm **Windows-native** (my recommendation) or direct me to WSL (I'll install a distro first).
2. **VRAM window:** confirm a window where `mistral-sim` is unloaded/unpinned, or authorize me to request its unload — *or* tell me Tier 1 (CPU mechanics check) alone is acceptable to run **now** while the GPU stays busy, deferring Tier 2 to a later window.

On go-ahead I will, in order: build the pinned cu128 venv (`uv`, Python 3.12) and record versions + `torch.cuda.is_available()`; download `Qwen/Qwen2.5-7B-Instruct @ a09a3545` and the lens `neuronpedia/jacobian-lens @ qwen-n1000 / 16a01f3`, recording paths + disk footprint + the lens file's sha256; run the `isfinite` guard; reproduce Tier 1 and Tier 2; then fill in §§1 (`cuda.is_available`), 4, and the VRAM/time footprint here, update PROVENANCE, commit + push, and **stop at STOP POINT 2**.

---

## 6. What I did NOT do (per brief)

No heavy download; no venv/torch install (a torch install *is* "heavy" — deferred behind the environment verdict per the brief's sequencing); nothing loaded into VRAM (mistral-sim pinned); no token sets proposed or touched (rule 4); no reification-gradient originals accessed (rule 3); no vignette read (that is Stage 0.3). The clone is gitignored and re-creatable, committing me to nothing.
