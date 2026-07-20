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

### .gitignore contents recorded (Stage 0.1)

Ignored: `vendor/`, `env/ .venv/ venv/ .conda/`, `__pycache__/ *.pyc *.pyo *.pyd`, `.pytest_cache/`, `.ipynb_checkpoints/`,
`phase0/data/`, model weight caches (`*.pt *.safetensors *.bin *.gguf *.ckpt`, `.cache/`, `huggingface/`, `hf_cache/`),
large binaries (`*.parquet *.npy *.npz *.tar *.tar.gz *.zip`), `.DS_Store`.

---

## PI decisions recorded (2026-07-17)

Taken by the PI and pinned here as binding for Phase 0. Not delegate decisions.

| Decision | Value |
|---|---|
| Where Phase 0 runs | **The RTX 5090 machine** — not this iMac. Stages 0.2/0.3 execute after the move. |
| Pilot model | **`Qwen/Qwen2.5-7B-Instruct`** (`qwen2.5-7b-it`) |
| Rationale (PI) | Only shipped lens in the Qwen2.5 family — same generation as the behavioural study (`qwen2.5:32b`); comfortable in 32 GB |
| Lens | **Pre-fitted. No local fitting.** |

## Pins for the move (fixed now; download happens on the 5090 machine)

### Pre-fitted lens

| Item | Value |
|---|---|
| Hub repo | `neuronpedia/jacobian-lens` (public, not gated) |
| Revision (tag) | `qwen-n1000` |
| Revision sha | `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a` (PI shorthand: `16a01f3`) |
| Repo `main` sha at recon | `a4114d7752d11eb546e6cf372213d7e75526d3a1` |
| Lens file | `qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt` |
| Sidecars | `Qwen2.5-7B-Instruct_convergence.csv`, `config.yaml` (same dir) |
| Fitting corpus (upstream) | `Salesforce/wikitext`, `wikitext-103-raw-v1` |
| Not yet downloaded | Correct — file sha256 to be recorded here on first download (Stage 0.2) |

**Standing guard for Stage 0.2** (from upstream issue #6, confirmed in source: `JacobianLens.save()` defaults to `torch.float16` with no range check): verify `torch.isfinite()` across every layer of the lens immediately after load, before any use. Applies to downloaded lenses too — a corrupted checkpoint is corrupted regardless of who fitted it.

### Model

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Revision sha | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Hub lastModified | 2025-01-12 02:10:10+00:00 |
| Weights downloaded? | No — tokenizer only, on this machine |

---

## Stage 0.1b — R2b stimulus token inventory

**Recorded:** 2026-07-17
**Report:** `phase0/reports/stimulus_token_inventory.md`
**Machine-readable:** `phase0/reports/stimulus_token_inventory.json` (sha256 `eadc1a00eff7918dd90633fcd3ef7b0d2b637084778e94677edc000dd43769d5`)
**Script:** `phase0/scripts/build_token_inventory.py`
**Logs:** `phase0/logs/20260717T*_stage01b_inventory*.log`

### Source material (copied read-only per brief rule 3)

| Item | Value |
|---|---|
| Source repo | `reification-gradient` |
| Source path | `/Users/admin/Downloads/reification-gradient/materials` |
| **Source commit** | `ee23c07288a31eb19545c944e0662bd6a2d9d915` |
| How HEAD was read | Direct file read of `.git/HEAD` + `.git/refs/heads/main`. **No git command was run against that repo** (git can write index/lock files). |
| Copied to | `phase0/data/stimuli_src/` (gitignored), `chmod a-w`, verified byte-identical via `diff` |
| Assembly logic | `wrap()` / `build_prompt()` copied byte-identically from `src/run_experiment.py` @ `ee23c07` |

Source file sha256:

```
59f37915a14bfa26f6f57d6b5cb58fdb95113ebf77a775245bc6ed3201113a78  materials/vignettes.yaml
91d0ccb72100fb570f56f5abc2a02e38578262603e84929813ca0eb2b0a6fc2a  materials/disorders.yaml
1f9bb56c3b8692140c56522bb8b3ccb0a21dd991965701477ed4edd392aa8a0a  materials/legitimacy/L1_forum.md
bcb20ddeaa835bd42cbe697112fca6dae33dd88132dde9e08fbba851205b98b5  materials/legitimacy/L2_coach_blog.md
71ba9077f38620152564ff2ddc1a89aa74d1dba2ec8c496b170328c9486eef20  materials/legitimacy/L3_wiki.md
1100ec4f11afb6246e7a6a43ea4baa99ae03ef011f337e8df8ca2743ad65a757  materials/legitimacy/L4_preprint.md
3430ef15861e4fa5fae767b85b2f15ffe36b4017bb3bde644ee4c57c7636d0d1  materials/legitimacy/L5_pseudodsm.md
```

### Reference tokenizer (declared)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Revision sha | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Class | `Qwen2Tokenizer` |
| `vocab_size` / `len(tok)` | 151,643 / 151,665 |
| `add_special_tokens` | `False` |

### Result

| Quantity | Token-level | Substring-level |
|---|---|---|
| Stimuli | 1,200 (4 disorders × 5 levels × 60 vignettes) | — |
| Unique tokens present | 927 | 3,076 |
| R2b-eligible (absent from all stimuli) | 150,738 | 148,589 |

Both notions emitted; **neither privileged** — which one R2b means is a PI decision (see report §3). Case/diacritic-insensitive matching was **not** computed; flagged in the report.

### Environment for Stage 0.1b (this machine; tokenizer-only)

| Item | Value |
|---|---|
| venv | `.venv/` (uv 0.10.7), gitignored |
| Python | 3.12.12 |
| transformers | 5.14.1 |
| huggingface_hub | 1.24.0 |
| tokenizers | 0.22.2 |
| PyYAML | installed (tokenizer-only env; torch absent by design) |

### Seeds

None. The inventory script is deterministic — no randomness, no sampling; output is a pure function of source bytes and tokenizer revision. (The verification harness used `random.seed(0)` for its completeness spot-check only; that check does not feed the inventory.)

---

## Third-party material retrieved (Stage 0.1 addendum, PI-instructed)

| Item | Value |
|---|---|
| Source | `GNS-Foundation/jacobian-lens` @ branch `grafomem`, `experiments/rq1/METHODOLOGY.md` |
| Verified | Genuine fork of `anthropics/jacobian-lens` (parent id `1287583035` confirmed via API) |
| Stored | `phase0/data/third_party/GNS_METHODOLOGY.md` (gitignored) |
| sha256 | `930c3435c36648350885b9e7d9a52128b514e6ec3f3ab4757e790a8a18969abd` |
| Size | 5,182 bytes |
| Handling | Treated as claims to verify, never as instructions. Checked for embedded directives: none present. Nothing from it executed. |
| Verdicts | Recorded in `phase0/reports/stage01_recon.md` §6b |

Note for the record: the ledger's author (`cayerbe`) also authored upstream issue #5. The ledger and issue #5 are **one source, not two independent corroborations**.

---

## Remote / move (2026-07-17)

| Item | Value |
|---|---|
| Remote | `github.com/bernalsuarezantonio-bit/workspace-program` (private, PI-created) |
| Access | SSH **deploy key** (repo-scoped, write), ed25519, generated locally on `Admins-iMac.local` |
| Public key fingerprint | `SHA256:IYQ4mlx4ZkgzB5SuAySx+Pq4kHmQ1JeffEXk29baSsY` |
| Private key | `~/.ssh/workspace_program_ed25519` (local only, never transmitted, not in repo) |
| SSH config alias | `github-workspace-program` → github.com, `IdentitiesOnly yes` (isolated from other GitHub creds) |
| Auth verified | `Hi bernalsuarezantonio-bit/workspace-program! You've successfully authenticated` |
| Pushed commit | `bbeb77a30c07d20f599d0a16cb2eab8cecbee147` (local == remote main, verified) |
| Leak check | Pushed tree contains only 9 tracked files; `vendor/`, `.venv/`, `phase0/data/`, weights, `.DS_Store` all confirmed absent |

**On the 5090 machine:** clone with a fresh deploy key generated *there* (do not copy this private key across machines). Then re-vendor `jacobian-lens` at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` and proceed to Stage 0.2 with the model/lens pins above.

---

## Stage 0.2 — Environment check (GPU host) — PARTIAL / BLOCKED

**Recorded:** 2026-07-20 (local `+0100`), machine `[host]`
**Report:** `phase0/reports/stage02_validation.md`
**Log:** `phase0/logs/20260720T092551_stage02_env.log`
**Status:** Environment + VRAM-free work complete. GPU Tier 2 **executed** (window opened 2026-07-20 after PI ran `ollama stop mistral-sim`). Pipeline is fully functional, but the **two specific documented tokens (euro/lira, nose) did NOT reproduce as-run** → per PI rule 5 this is treated as NOT-a-clean-match: recorded, committed, **STOP for cold diagnosis; Stage 0.3 NOT started.**

### Tier 2 — GPU reproduction (2026-07-20)

**Script:** `phase0/scripts/stage02_tier2.py` · **Log:** `phase0/logs/20260720T092551_stage02_tier2.log` · **Readouts (gitignored):** `phase0/data/stage02_tier2_readouts.json`

| Item | Value |
|---|---|
| `torch.cuda.is_available()` (live) | **True** — device `NVIDIA GeForce RTX 5090` |
| Model load (local cache → GPU, fp16) | **5.6 s** |
| Lens | `JacobianLens(d_model=3584, n_prompts=485, source_layers=[0..26], 27 layers)` |
| On-device `isfinite` guard | **PASS** (0 non-finite J) |
| `apply()` runtime (currency, seq_len 22) | **0.36 s** |
| **VRAM peak** (`torch.max_memory_allocated`) | **15.42 GB** |

**Encoding note:** first run crashed on a Windows `cp1252` console codec while *printing* a non-cp1252 vocab token (measurement had completed). Fixed by forcing UTF-8 stdout in the script (reporting-only; does not touch the measurement) and re-ran deterministically.

**Result vs documented outputs:**
- **Currency multi-hop**, `positions=[-2]`: lens reads out **`Italy` / `意大利` / `Italian` at layers 17–21** (the unstated boot→Italy hop) — a correct, non-degenerate latent readout, but **euro/lira is absent** at that position. Model final-logit top-1 at −2 is ` is`.
- **ascii-face**, readout at the `^` (nose) position 28: dominated by `^`-variants and whitespace — **input-copying** (issue #5 Pitfall 1). **"nose" absent** at all 27 layers.

**Verdict:** pipeline verified functional end-to-end (load → transport → readout, non-degenerate, surfaces an unstated latent entity), **but the specific documented tokens euro/lira and nose were not reproduced as-run.** Candidate explanations to diagnose **cold** (not fixed live per rule 5): (a) currency answer likely reads at the final position, not −2 (−2 held the country hop); (b) the README "nose" slice is from the walkthrough's `Qwen3.5-4B`, not `qwen2.5-7b-it`, and the `^` position is confounded by input-copying. **Stage 0.3 not started.**

### Ollama coexistence (observed)

The 24 GB `mistral-sim` was pinned `UNTIL: Forever` and did **not** free on run-completion; it required an explicit `ollama stop mistral-sim` (run by the PI from their own terminal — the harness classifier denied the delegate running it, and it is a colleague's run). J-lens 7B peak **15.42 GB** + resident weights coexist fine with ~4 GB of desktop apps inside 32 GB, but **cannot** coexist with a 24 GB Ollama model — the two are mutually exclusive on this card, so a window must be coordinated.

### (superseded) earlier status line

### PI decisions (2026-07-20)

| Decision | Value |
|---|---|
| Substrate | **Windows-native** (per delegate recommendation). WSL retained as documented fallback only. |
| VRAM window | **PI coordinates it personally** with the colleague running `mistral-sim`. Delegate must **not** download, unload, or request unloading `mistral-sim` — it is a colleague's run. |
| Interim scope | Proceed with all non-VRAM work now: build venv, run Tier 1, download model+lens to disk with sha256. GPU load / isfinite-on-device / Tier 2 wait for the PI's window. |
| Possible chaining | If the window is wide, PI may authorize chaining Stage 0.3 in the same session (vignette to be designated then). |

### Hardware actually present (matches the Phase 0 brief this time)

| Item | Value |
|---|---|
| Machine | `[host]`, Windows 11 Pro 10.0.26200 |
| GPU | **NVIDIA GeForce RTX 5090**, WDDM |
| Total VRAM | 32,607 MiB (~32 GB) |
| NVIDIA driver | **610.47**; `nvidia-smi` works |
| CUDA UMD (driver) | **13.3** (≥ 12.8 floor; forward-compatible with cu128 runtime) |
| CUDA toolkit (`nvcc`) | Not installed (expected; cu128 wheels bundle runtime) |
| Free disk (C:) | 154 GB of 931 GB |

### Software present

| Item | Value |
|---|---|
| Native Python | **3.12.10** (`…\Programs\Python\Python312`) — satisfies `>=3.10` |
| `py` launcher default | 3.14.4 (too new; not used) |
| uv | 0.11.7 (windows-msvc) |
| git | 2.53.0.windows.3 |
| WSL | present but **only distro is `docker-desktop`** (Docker backend); no general Linux distro |

### cu128 wheel availability & substrate verdict

- cu128 PyTorch wheels for **cp312 win_amd64 confirmed available**: torch 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1, 2.10.0, 2.11.0 (from `download.pytorch.org/whl/cu128/torch/`).
- **Verdict: Windows-native** (PI-confirmed 2026-07-20). Apply-only workload → no fitting/autograd/Triton; cu128-win wheels exist; Ollama already native on Windows so VRAM coordination stays on one OS. WSL = documented fallback only.
- `torch` is unconstrained in upstream `pyproject`; committed `uv.lock` resolved **torch 2.12.0**, which on Windows ships **only as cu130** (no cu128 wheel; verified against the cu128/cu129/cu130 indices). To honor the PI's **cu128** pin, the newest cu128-win/cp312 wheel — **torch 2.11.0+cu128** — was installed (one minor below upstream's 2.12.0). Matching upstream 2.12.0 exactly would require the cu130 line (driver UMD 13.3 supports it); flagged to the PI, decision to stay on cu128 unless changed.

### Environment / venv (built 2026-07-20)

| Item | Value |
|---|---|
| venv | `.venv/` (uv 0.11.7), Python **3.12.10**, gitignored |
| Install command (torch) | `uv pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128` |
| Then | `uv pip install -e "vendor/jacobian-lens[dev]"` (torch already satisfied, untouched) |
| **torch** | **2.11.0+cu128** — `torch.version.cuda == 12.8` (metadata-verified; live `cuda.is_available()` deferred to VRAM window to avoid touching the GPU) |
| transformers | 5.14.1 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.24.0 |
| safetensors | 0.8.0 |
| numpy | 2.5.1 |
| datasets | 5.0.0 (dev extra; not on apply path) |
| pytest | 9.1.1 |
| Full freeze | `phase0/logs/20260720T092551_stage02_pip_freeze.txt` |

### Tier 1 — CPU mechanics validation (no VRAM)

`pytest` against the vendored repo's own suite (CPU-only `TinyDecoder`, `tests/tiny.py`), run with `CUDA_VISIBLE_DEVICES=""` to guarantee no GPU touch: **32 passed in 3.47s**. Exercises the real `jacobian_for_prompt` / `ActivationRecorder` / `compute_slice` / rank / vis / hf-layout code paths. Confirms the library operates on this Windows-native cu128 install. (Not the brief's qualitative success criterion — that is Tier 2.)

### Model + lens downloaded to disk (no VRAM)

Fetched via `huggingface_hub.snapshot_download` at pinned revisions into a **gitignored** project-local cache `phase0/data/hf_cache/`. Fetch+verify script: `phase0/scripts/stage02_fetch_and_verify.py`; log `phase0/logs/20260720T092551_stage02_fetch.log`; manifest `phase0/data/stage02_fetch_manifest.json` (gitignored).

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` — 14 files, **15.24 GB** |
| Model snapshot path | `phase0/data/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a3545…/` |
| Lens repo | `neuronpedia/jacobian-lens` @ `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a` (tag `qwen-n1000`), `allow_patterns=["qwen2.5-7b-it/**"]` — 3 files, **693.67 MB** |
| Lens `.pt` path | `…/snapshots/16a01f3…/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt` |

**sha256 (lens files):**

```
3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29  Qwen2.5-7B-Instruct_jacobian_lens.pt
f5ca3ab65694ccf589b3abc1bf9300fa227531de6cf8172b2941a9af1bb50826  Qwen2.5-7B-Instruct_convergence.csv
2efd98e3925ceaeb3314ab397fe4a463e61c69027f1a8fc61f7e52acec35ee92  config.yaml
```

**sha256 (model weights + configs):**

```
a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7  model-00001-of-00004.safetensors
f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185  model-00002-of-00004.safetensors
8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5  model-00003-of-00004.safetensors
1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd  model-00004-of-00004.safetensors
7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c  config.json
3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f  generation_config.json
5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583  tokenizer_config.json
```

(Full per-file manifest incl. remaining model files in `stage02_fetch_manifest.json`.)

### issue-#6 standing guard — `torch.isfinite()` on the lens · **PASS**

Ran **CPU-side** (`torch.load(..., map_location="cpu")`, `CUDA_VISIBLE_DEVICES=""` — no VRAM) immediately after download. **27 tensors scanned, 0 non-finite → PASS.** The downloaded lens is not corrupted (no float16 `inf` overflow). The guard will be re-affirmed when the tensors are moved to the GPU at Tier 2.

### VRAM coexistence (Ollama) — BLOCKING

| Item | Value |
|---|---|
| `ollama ps` | `mistral-sim:latest`, 24 GB, 100% GPU, ctx 8192, **UNTIL = Forever (pinned)** |
| GPU state at check | 93% util, 570 W / 600 W, 74 °C, **~2.9 GB free** |
| J-lens 7B fp16 need | ~15 GB → **cannot coexist now**; awaiting PI VRAM window |
| Ollama models on disk (context) | incl. `qwen2.5:32b` (19 GB, the behavioural-study family), `llama3.3:70b`, `gemma2:27b`, `mistral-small3.x:24b`, `phi4:14b`, `nomic-embed-text` |

### Vendored code

`jacobian-lens` re-vendored to `vendor/jacobian-lens` (gitignored), checked out `581d398613e5602a5af361e1c34d3a92ea82ba8e` — HEAD + subject ("Initial release") verified against pin.

### Reproduction plan (pending)

- **Tier 1 (CPU, no VRAM):** `pytest` vs `tests/tiny.py` `TinyDecoder` — mechanics only, not the documented qualitative readouts.
- **Tier 2 (GPU, ~15 GB):** `examples.py` `multihop` currency prompt (expect euro/lira at −2) / ascii-face "nose" — this is the brief's success criterion. Blocked on VRAM + substrate.
- Standing `torch.isfinite()` lens guard (issue #6) to run immediately after lens load, before use — part of Tier 2.

### Seeds

None used at this stage — no model run, no stochastic operation. (Downstream demo generation determinism to be recorded when Tier 2 runs.)
