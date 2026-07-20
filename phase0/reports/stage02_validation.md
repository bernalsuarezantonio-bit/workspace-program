# Stage 0.2 — Environment check + pipeline validation

**Date:** 2026-07-20
**Machine:** `[host]` — Windows 11 Pro 10.0.26200, RTX 5090 (the designated GPU host)
**Author:** Claude Code (Phase 0 delegate)
**Status:** 🟡 **VRAM-free half complete; one gate remains.** Substrate confirmed (Windows-native); venv built; Tier 1 CPU tests pass; model + lens downloaded, hash-verified, `isfinite` guard PASS. **Only Tier 2 (GPU load + documented-readout reproduction) is left — it waits on the PI's VRAM window.** Not STOP POINT 2 yet.
**Log:** `phase0/logs/20260720T092551_stage02_env.log`, `…_stage02_fetch.log`, `…_stage02_pip_freeze.txt`
**jacobian-lens commit (re-vendored here):** `581d398613e5602a5af361e1c34d3a92ea82ba8e` ("Initial release") ✓ matches pin

**PI decisions applied (2026-07-20):** substrate = **Windows-native** (WSL = documented fallback); the **VRAM window is coordinated by the PI personally** with the colleague running `mistral-sim` — I must **not** download, unload, or request unloading `mistral-sim`, and I have not. All work below stayed strictly off the GPU (every Python call ran with `CUDA_VISIBLE_DEVICES=""`).

---

## 0. Headline — pipeline stands up on this host; only the GPU reproduction is left

This is the real GPU host (RTX 5090, driver 610.47, CUDA UMD 13.3, cu128 wheels present). On PI go-ahead I completed everything that does not touch VRAM, and it is all green:

- **Windows-native cu128 venv** built (Python 3.12.10, torch 2.11.0+cu128 / CUDA 12.8, transformers 5.14.1).
- **Tier 1 CPU mechanics: 32/32 tests pass** against the repo's own `TinyDecoder` suite — the library's real code paths work here.
- **Model (15.24 GB) + lens (693.67 MB)** downloaded at pinned revisions, every file sha256'd into PROVENANCE.
- **issue-#6 `isfinite` guard: PASS** (27 tensors, 0 non-finite), run CPU-side right after download.

The one remaining gate is the brief's actual success criterion — **Tier 2**, the documented qualitative readout (the "nose" / currency examples) on the 7B + lens — which needs ~15 GB of VRAM. `mistral-sim` (24 GB) is pinned `Forever` and the card was at 93 % util when checked, so ~15 GB is not available. **You coordinate that window; I wait.** §§1–3 record the completed work; §§4–5 the remainder.

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
| torch build | **2.11.0+cu128**, `torch.version.cuda == 12.8` — metadata-verified in the venv |
| `torch.cuda.is_available()` (live device probe) | **Deliberately deferred to the VRAM window** — the call initializes a CUDA context (reserves VRAM), so running it now would touch the colleague's GPU. Build correctness is confirmed via metadata instead; the live `True` check is the first thing I run when the window opens. |

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

## 3. Completed VRAM-free work

### 3a. venv (Windows-native, pinned cu128)

Python **3.12.10**; `torch==2.11.0+cu128` from the cu128 index; then `jlens` editable + deps/dev extras. Key pins (full freeze in `…_stage02_pip_freeze.txt`):

| torch | transformers | tokenizers | huggingface-hub | safetensors | numpy | datasets | pytest |
|---|---|---|---|---|---|---|---|
| 2.11.0+cu128 (cuda 12.8) | 5.14.1 | 0.22.2 | 1.24.0 | 0.8.0 | 2.5.1 | 5.0.0 | 9.1.1 |

**torch pin note (touches your cu128 decision):** upstream `uv.lock` resolved **torch 2.12.0**, which on Windows exists **only as cu130** (no cu128 wheel — checked cu128/cu129/cu130 indices). I installed **2.11.0+cu128** (newest in the cu128 line) to honor your recorded cu128 pin. Matching 2.12.0 exactly means moving to the cu130 line (your driver supports it). Staying on cu128 unless you say otherwise.

### 3b. Tier 1 — CPU mechanics validation · **PASS**

`pytest` against the vendored repo's own suite (CPU-only `TinyDecoder`, `tests/tiny.py`), run with `CUDA_VISIBLE_DEVICES=""`: **32 passed in 3.47 s**. Exercises the real `jacobian_for_prompt` / `ActivationRecorder` / `compute_slice` / rank / vis / hf-layout paths. Confirms the library runs on this install. *(Not the brief's qualitative criterion — see Tier 2.)*

### 3c. Model + lens on disk · hash-verified

`huggingface_hub.snapshot_download` at pinned revisions → gitignored `phase0/data/hf_cache/`. Script `phase0/scripts/stage02_fetch_and_verify.py`; manifest `phase0/data/stage02_fetch_manifest.json`.

| Artifact | Revision | Files | Footprint |
|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | `a09a3545…` | 14 | **15.24 GB** |
| `neuronpedia/jacobian-lens` (`qwen2.5-7b-it/**`) | `16a01f3…` (tag `qwen-n1000`) | 3 | **693.67 MB** |

lens `.pt` sha256 `3b3ab44c…cba29`; all model-shard and sidecar sha256 recorded in PROVENANCE.

### 3d. issue-#6 `isfinite` guard · **PASS**

Run **CPU-side** immediately after download (`torch.load(map_location="cpu")`, GPU hidden): **27 tensors, 0 non-finite**. The downloaded lens has no float16 `inf` overflow. Re-affirmed on-device at Tier 2.

*(Also: `jacobian-lens` re-vendored at pinned `581d398…`, HEAD verified; walkthrough's `.cuda()` is fine here — this is CUDA, no MPS port needed.)*

---

## 4. Pipeline validation — the reproduction, staged

Two-tier reproduction. Tier 1 is **done** (CPU); Tier 2 is the brief's actual success criterion and is the only thing waiting on the VRAM window.

| Tier | What | VRAM | Status |
|---|---|---|---|
| **1** | `pytest` vs **CPU-only `TinyDecoder`** — real `jacobian_for_prompt` / `ActivationRecorder` / `compute_slice` paths | none | ✅ **32/32 pass** (mechanics; not the documented qualitative readouts) |
| **2** | **Documented demo** from `examples.py`: `multihop` currency *"…the country shaped like a boot is"* → expect euro/lira at position −2; and/or ascii-face **"nose"** at mid layers | ~15 GB (7B + lens) | ⏸ **waits on VRAM window** — this is *"readouts qualitatively match the documented expected output"* |

Lens integrity for Tier 2 is already de-risked: the issue-#6 `isfinite` guard passed CPU-side (§3d) and will be re-affirmed once tensors are on the GPU.

---

## 5. What's left, and what unblocks it

**One gate:** a VRAM window with ~15 GB free (you coordinate it with the `mistral-sim` owner). When you confirm, I will, in order:

1. `torch.cuda.is_available()` live check (expect `True`) — the deferred device probe.
2. Load `Qwen/Qwen2.5-7B-Instruct` (fp16) + the pre-fitted lens onto the GPU.
3. Re-affirm the `isfinite` guard on-device.
4. **Tier 2:** run the documented demo(s), confirm readouts qualitatively match (the "nose" / currency outputs), and record VRAM peak + runtime here.
5. Update PROVENANCE, commit + push, **stop at STOP POINT 2**.

If the window is wide and you authorize it, I can chain **Stage 0.3** in the same session once you designate the vignette and its source path.

I will **not** touch `mistral-sim` or the GPU before your window — nothing here has, and nothing will until then.

---

## 6. What I did NOT do (per brief / PI constraint)

Nothing loaded into VRAM; the GPU was never touched (every Python call ran `CUDA_VISIBLE_DEVICES=""`); `mistral-sim` was never downloaded, unloaded, or asked to unload; no live `cuda.is_available()` probe (deferred — it would allocate a context); no token sets proposed or touched (rule 4); no reification-gradient originals accessed (rule 3); no vignette read (that is Stage 0.3). All downloaded weights live under gitignored `phase0/data/`.
