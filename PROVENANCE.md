# PROVENANCE

Program: J-lens representational study (Phase 0).
This file is updated at every stage. All hashes and versions recorded verbatim from the machine.

---

## Stage 0.1 — Environment & repository reconnaissance

**Recorded:** 2026-07-17 (local time, `Admins-iMac.local`)
**Log:** `phase0/logs/20260717T102333_stage01_recon.log`

### Repository

| Item | Value |
|---|---|
| This repo | `/Users/admin/Downloads/workspace-program` (own git repo, `git init` 2026-07-17) |
| Nested inside another repo? | No — verified `git rev-parse --show-toplevel` returned "not a git repository" before `git init` |

### Vendored third-party code (gitignored)

| Item | Value |
|---|---|
| Repo | `https://github.com/anthropics/jacobian-lens` |
| Path | `vendor/jacobian-lens/` (gitignored) |
| **Commit** | `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Commit date | 2026-07-02 09:07:51 +0000 |
| Commit subject | `Initial release` |
| Default branch | `main` |
| License | Apache-2.0 |
| Existence verified | GitHub API returned a real record; a deliberate nonsense-repo control returned HTTP 404, confirming the API was not blanket-200'ing |

### Hardware actually present (NOT the hardware named in the Phase 0 brief)

| Item | Value |
|---|---|
| Machine | `Admins-iMac.local`, Darwin 25.5.0, `arm64` |
| Chip | Apple M1 (8 CPU cores, 8-core GPU, Metal 4) |
| Unified memory | 16 GB |
| Free disk | 58 GB |
| NVIDIA GPU | **None** — `nvidia-smi`: command not found |
| CUDA toolkit | **None** — `nvcc`: command not found |
| RTX 5090 / 32 GB VRAM / CUDA 12.8 | **Not present on this machine** |

### Software present

| Item | Value |
|---|---|
| System Python | 3.9.6 (**below jlens `requires-python = ">=3.10"`**) |
| System torch | 2.8.0 — `cuda.is_available()` = **False**, `backends.mps.is_available()` = **True** |
| uv | 0.10.7 |
| Ollama | Not installed locally (`ollama`: command not found; `localhost:11434` no response) |

### Upstream artifacts identified (not yet downloaded)

| Item | Value |
|---|---|
| Pre-fitted lens repo | `neuronpedia/jacobian-lens` (HuggingFace), public, not gated |
| Repo `main` sha | `a4114d7752d11eb546e6cf372213d7e75526d3a1` (lastModified 2026-07-06) |
| Revision used by walkthrough | `qwen-n1000` → sha `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a` |
| Lenses available | 38 model directories (see `phase0/reports/stage01_recon.md`) |
| Fitting corpus (upstream default) | `Salesforce/wikitext`, `wikitext-103-raw-v1`, 1000 sequences × 128 tokens |

### Seeds

None used at this stage — no model was run and no stochastic operation was performed.

### Reification-gradient materials

Not yet copied. Verified read-only that the sources exist (listing only, no reads into this repo, no git commands against that repo):
`materials/vignettes.yaml` (60 vignettes), `materials/legitimacy/L1_forum.md`…`L5_pseudodsm.md`, `materials/disorders.yaml`.
Source path and commit will be recorded here when Stage 0.1b copies them.

### .gitignore contents recorded

Ignored: `vendor/`, `env/ .venv/ venv/ .conda/`, `__pycache__/ *.pyc *.pyo *.pyd`, `.pytest_cache/`, `.ipynb_checkpoints/`,
`phase0/data/`, model weight caches (`*.pt *.safetensors *.bin *.gguf *.ckpt`, `.cache/`, `huggingface/`, `hf_cache/`),
large binaries (`*.parquet *.npy *.npz *.tar *.tar.gz *.zip`), `.DS_Store`.
