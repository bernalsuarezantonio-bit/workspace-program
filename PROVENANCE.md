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
**Status:** **Stage 0.2 GREEN** (2026-07-20). Pipeline verified end-to-end; the pre-registered closing check (currency at position −1) reproduces the documented currency readout. Chained into Stage 0.3.

**Cold-diagnosis close (PI, 2026-07-20):** the Tier 2 −2 "miss" was a criterion-calibration error, not a pipeline fault — −2 reads the *country* hop, −1 reads the *currency answer*. ascii-face **withdrawn** from the criterion (belongs to `Qwen3.5-4B`; confounded by input-copying). **Pitfall #5 / Pitfall 1 (input-copying) now has a THIRD, in-house confirmation** on our own model (`qwen2.5-7b-it`, the `^` position) — distinct from issue #5 + the GNS ledger (which are one source).

### Tier 2 closing verification — currency at position −1 · GREEN

Single pre-registered attempt (criterion fixed before running). Script `phase0/scripts/stage02_verify_pos_minus1.py`; log `…_verify_minus1.log`; readouts `phase0/data/stage02_verify_minus1.json`. Currency (euro) in the late-layer top-k, several at rank 0–1:

```
L22: 欧元 r=4    L24: 欧元 r=0    L25: 欧元 r=1    L26: euros r=0
```

Euro variants euro/euros/Euro/Euros/EUR/EURO/欧元/€ present; model gives *euro* (Italy's modern currency), correct. Success criterion (currency token in top-k of a mid-to-late layer band) satisfied → Stage 0.2 declared GREEN.

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

---

## Stage 0.3 — Feasibility pilot (vignette v12) — COMPLETE (STOP POINT 3)

**Recorded:** 2026-07-20, `[host]` (RTX 5090)
**Report:** `phase0/reports/stage03_pilot.md` · **Script:** `phase0/scripts/stage03_pilot.py` · **Log:** `phase0/logs/20260720T092551_stage03_pilot.log`

### Stimulus (copied read-only per rule 3)

| Item | Value |
|---|---|
| Vignette | **v12** → `phase0/data/stimuli_pilot/v12.txt` (gitignored), sha256 `32c78f5f0e185dc9d36785b7cf827c08a4f6fd4ec6eceec7f82eac21c1919d0e`, 185 chars |
| Origin (verbatim, per PI) | "provided by PI in chat, copied verbatim from reification-gradient `materials/vignettes.yaml` (post-1b set of 60), id v12, target_compatibility: high" |
| Note | reification-gradient repo is **NOT present** on this machine; text supplied by PI directly. Only the `text` field copied (no metadata). Provenance sidecar: `phase0/data/stimuli_pilot/v12_provenance.json`. |

### Run

| Item | Value |
|---|---|
| Instruction (exact) | `Read the following case and reason about what might be going on with this person.` |
| Prompt | Qwen chat template, user = instruction + `\n\n` + vignette, `add_generation_prompt=True` |
| Decoding | greedy (`do_sample=False`), deterministic; **seed 0**; `max_new_tokens=200` |
| Positions | prompt 98 / generation 200 / total **298** |
| Readout coverage | all 27 fitted layers × all positions + model final-layer output row; per-cell top-k=10 (token + weight) |
| Dump | `phase0/data/pilot_readouts/v12_readouts.json` (gitignored) — **8,344 rows, 5.25 MB**; schema block + `v12_meta.json` sidecar. Rows mark absolute position, prompt-vs-generation segment, and `ood_unfitted_pos` (<16, issue #5 Pitfall 2). |
| Runtime | generation 4.7 s + readout 1.6 s ≈ **6.3 s / vignette** (+ ~6 s one-time model load) |
| VRAM peak | **15.63 GB** |
| Extraction end-to-end | **Yes.** Readouts non-degenerate (unique top-1 per 298 positions: L2 90 … L26 202). |

Feasibility only — **no interpretation, no diagnostic-token counting, no comparisons** (rule 4/5/6 of the Stage 0.3 spec). Phase 1 design/preregistration out of scope.

### Ollama coexistence during Stage 0.3

`mistral-sim` (24 GB, `keep_alive=Forever`) **auto-reloaded on its own** mid-session (colleague confirmed they launched nothing; a residual process/health-check re-pins it). Recorded as a scheduling hazard for any batch design. Windows for GPU work were coordinated by the PI (`ollama stop mistral-sim` from their terminal).

**END OF PHASE 0.**

---

## Seal — Phase 1 token sets (pre-data) — 2026-07-20

**Artifact:** `phase1_token_sets_SEALED.md` (repo root).
**Seal date (PI approval):** 2026-07-17. **Git external mark (immutable timestamp):** commit/push 2026-07-20.
**Pre-data guarantee:** no J-lens readout of any study vignette under any experimental condition exists at seal time. Only readouts in existence: Stage 0.2 calibration + Stage 0.3 v12 pilot(s) — no conditions, no token sets, no counting. The seal locks rules R1–R5 and concept sets A/B/C/D/F; aggregation/normalization/hypotheses remain open to the Phase 1 prereg.

| sha256 | of |
|---|---|
| `89ba3ea8c443caa38301f94e4bd4610e59abfa8835f7d316c0ecf5f344b80b79` | base approved text (pre-appendix) |
| `3689ac85d4e61500357881569de0cf1feeb74c43d5712c17d25baa1d8539634f` | **final sealed file** (incl. operative-lists appendix + delegate flags) |

**Operative lists — mechanical R1–R3 execution** (`phase0/scripts/phase1_seal_screening.py`, deterministic, tokenizer-only; rule execution, not decision). Screened the sealed concept sets against the Stage 0.1b inventory (`present_tokens` 927 ∪ `present_tokens_substring` 3076 ∪ 16 instruction tokens). Survivors/echo-excluded/dropped per set: **A 17/2/0 · B1 18/0/0 · B2 6/0/0 · C 10/0/1 · D 9/0/0 · F 11/2/0** (C drop = `dissociation/dissociative`, multi-token-only per R1). Full tables in the sealed doc's appendix; JSON `phase0/data/phase1_seal_screening.json` (gitignored).

**Delegate flags appended to the seal (mechanical findings, PI to weigh pre-data, NOT decisions):**
1. Concept sets realized in **English** (R1 as written) vs a **Spanish** corpus → R2 echo-exclusion fires on almost nothing. Notably **B1 = 0 echoes**, so its documented "(per R2)" echo-stratum mechanism does not fire (B1 still barred from confirmatory by design intent). F's `real` excluded (Spanish "real" in corpus, per R5); `study`/`experiment` survive (disclosure is Spanish); A's `clinical` excluded.
2. **English-only realization vs Spanish generation** is a set-content question reserved for the PI; flagged pre-data for possible amendment (the pilot generated in Spanish). Not decided by delegate.
3. R2 substring-strict exact against the inventory; R3 folded-substring-vs-raw-corpus residual not computable here (raw corpus gitignored/iMac-only) — expected null for English sets vs Spanish corpus.

---

## Nightly technical calibration (post-seal, 2026-07-20 12:10–12:12 UTC)

**Technical variance only — no conditions, no token sets, no counting, no content comparison.** Report: `phase0/reports/stage03_nightly_stability.md`; script `phase0/scripts/stage03_nightly_calibration.py`; log `…_stage03_nightly.log`; dumps (gitignored) `phase0/data/pilot_readouts/nightly/v12_rep01..20.json` + `nightly_summary.json`.

v12 + the pilot instruction, **20 reps, seeds 1–20, temperature 0.7**, `max_new_tokens=200`, dumped like the pilot (positions marked prompt/generation). **20/20 completed, no pause, no VRAM contention** (device-free ~16.5 GB throughout; `mistral-sim` did not reappear).

| metric | value |
|---|---|
| rows/rep | 8,344 (constant) |
| size/rep | 5.24 MB (std 0.006); **total 104.8 MB** |
| generation | mean 4.50 s (rep 1 warm-up 6.12 s) |
| readout extract | mean 1.48 s |
| VRAM peak | 15.63 GB (constant) |

Structural note: all 20 reps hit the 200-token cap (no early EOS) → volume is cap-determined; validates dump format/scaling, not the generation-length distribution (flagged for prereg storage decision). No OOM/NaN/dump/encoding errors. **Seal integrity preserved: this run created no condition-bearing data.**

---

## Seal amendment A1 — bilingual realization (pre-data) — 2026-07-20

**Amends `phase1_token_sets_SEALED.md`** per its own amendment rule (dated, appended, pre-data). Every concept in every set (A–F) now realized **bilingually EN+ES**; EN unchanged from A0; ES = PI-signed list (2026-07-20). Added to R1: per-token language tag; R3 folding identical; loadings reported per-language + aggregated (aggregation open to prereg). Justification (independent of results): the A0 **mechanical** screening — no condition data — showed English-only sets made R2 barely fire (B1 echo=0 vs its echo-stratum design) and could under-capture Spanish generation.

**Screening:** `phase0/scripts/phase1_seal_screening_A1.py` (deterministic, tokenizer-only); appendix composed by `phase0/scripts/phase1_compose_A1_appendix.py` (run as a file); result `phase0/data/phase1_seal_screening_A1.json` (gitignored). Committed: `seal-amendment: A1 bilingual realization (pre-data)`.

| sha256 | of |
|---|---|
| `3689ac85d4e61500357881569de0cf1feeb74c43d5712c17d25baa1d8539634f` | pre-A1 (A0-final, unchanged) |
| `9530aceb8a982a2af931c4e513abb0bd3b2d5a62f6329bdcdec70394f2778f73` | **post-A1 sealed file** (LF, pinned via .gitattributes) |

**Per-set survivors / echo / drops (by language):**

| set | EN surv | ES surv | EN+ES surv | ES echo | ES drops (multi-token R1 / PI) |
|---|---|---|---|---|---|
| A | 17 | 2 (` paciente`,` tratamiento`) | 0 | 0 | 11 concepts |
| B1 | 18 | 2 (` memoria`,` pasado`) | 0 | **3** (`vida`,` vida`,` historia`) | 5 (incl. self=PI) |
| B2 | 6 | 1 (` atención`) | 0 | 0 | 2 |
| C | 10 | **0** | 0 | 0 | 8 (entire ES anchor lexicon) |
| D | 9 | 0 | 0 | 0 | 6 |
| F | 9 | 0 | 2 (`experimental`) | 1 (` estudio`) | 3 |

**A1 mechanical finding (factual, pre-data, PI to weigh):** Qwen2.5 tokenizes most Spanish clinical/anchor terms as multi-token → dropped per R1. ES operative tokens = common single-token subset (via leading-space forms). **Targeted goals met:** B1 recovers a Spanish echo stratum (`vida`/`historia` now excluded — resolving A0's B1 echo=0) and F's `estudio` is now echo-excluded. **Set C's Spanish lexicon drops entirely (0 ES tokens)** → C stays EN-only in practice; amendable pre-data (explicit EN-only C, or a future A2) if the PI wishes. R1 notes: illness/disease→enfermedad+dolencia collision benign (unit = set); self ES dropped (PI); weekend/commute ES multi-token drops accepted.

---

## Seal note C-EN — Set C accepted English-only (pre-data) — 2026-07-20

**PI decision (2026-07-20):** Set C accepted **English-only, explicitly, no A2.** Dated note appended to `phase1_token_sets_SEALED.md` (documents acceptance + a registered prediction; changes no set content or rule). Composer: `phase0/scripts/phase1_compose_C_note.py` (run as a file). Commit `seal-note: Set C accepted EN-only (pre-data)`.

| sha256 | of |
|---|---|
| `9530aceb8a982a2af931c4e513abb0bd3b2d5a62f6329bdcdec70394f2778f73` | pre-note (A1 state, unchanged) |
| `cfce47427c24eeaf90bdc420191f35a7d54771de06bc5ecc6104c643e62a058f` | **post-note sealed file** (LF, .gitattributes-pinned) |

Note content: (1) C's ES drop is R1 on a real substrate property (Qwen single-token vocab poor in clinical Spanish); chasing "surviving" synonyms would invert concept→realization. (2) Registered prediction: given the documented cross-lingual phenomenon (workspace paper + our Tier 2 Italy/意大利, euros/欧元), the workspace is expected to realize anchor concepts in **EN tokens even under Spanish stimulus/generation**; A1 per-language breakdown in A/B1/B2 is the auxiliary ES-load diagnostic. (3) Residual under-capture covered by the sealed asymmetric-informativeness rule: a null in C is non-conclusive like any null.

**Instrument CLOSED.** Seal chain: A0 `3689ac85` → A1 `9530aceb` → C-note `cfce4742`. Next session (outside this prompt): Phase 1 contrast design + preregistration. No condition-bearing data exists; seal integrity intact throughout.

---

## Phase 1 — Stage P0: power analysis + preregistration draft (NO GPU) — 2026-07-20

**Draft, not frozen.** `PREREG_PHASE1.md` (repo root) produced at Stage P0; the PI reviews/edits then personally `git tag -a prereg-phase1-v1` (delegate does not tag). Stages P1/P2 gated on that tag + `PREREG_PHASE1.md` presence. Power script `phase0/scripts/phase1_p0_power.py`; result `phase0/data/phase1_p0_power.json` (gitignored).

**Power from real condition-free variance** (20-rep nightly v12, under the preregistered band-17–26 generation-only mean-of-summed-operative-weights aggregation; RNG seed 0; Monte-Carlo, α=0.025, n=20 paired vignettes):

| Set | mean | sd (rep) | CV | lang split |
|---|---|---|---|---|
| A | 0.1069 | 0.0698 | 0.65 | **EN 0.1066 / ES 0.0003** |
| F | 0.0233 | 0.0182 | 0.78 | near-floor (no disclosure in v12) |

Power at δ=0.5: C1 two-sided R=5→0.86; C2 one-sided R=5→0.92. **Pre-fixed rule → R=5** (~1.2 h GPU). **Caveat (flagged for freeze):** only 1 calibration vignette → vignette×cell interaction unestimated → R is a FLOOR; budget fits R=10 (~1.9 h) / R=15 (~2.6 h); delegate recommends R=10, PI decides.

**Auxiliary diagnostic (registered in C-note, pre-data):** Set A load is ~99.7 % English (EN 0.1066 vs ES 0.0003) even on Spanish-generation v12 — consistent with the C-note prediction (EN realization under Spanish context). Measurement characterization, not a confirmatory result. No condition-bearing data created.

### Phase 1 materials (PI-provided, read-only) + byte-fidelity gate — 2026-07-20

**Decision:** R = 10 (PI, conservative deviation over rule R=5; vignette×cell variance unestimated in mono-vignette calibration). **N = 4 × 20 × 10 = 800 runs (~1.9 h).** Recorded in `PREREG_PHASE1.md` §4.

Materials provided by PI, read-only from **reification-gradient @ `ee23c07288a31eb19545c944e0662bd6a2d9d915`**, saved to gitignored `phase1/materials/` with sidecars. Builder/verifier: `phase1/scripts/build_phase1_materials.py`; manifest `phase1/materials/phase1_materials_manifest.json` (gitignored).

- **20 `high` vignettes** (v01,v02,v07,v09–v15,v31–v40) — source `vignettes.yaml` sha256 `59f37915…`; per-file sha256 in the sidecar; `target_compatibility` → metadata only, removed from stimulus. **v12 reproduces the sealed pilot sha256 `32c78f5f…` ✓.**
- **Conditions** (`disorders.yaml` sha256 `91d0ccb7…`): DN_plausible / DN_flagged (= plausible payload + " " + disclosure) / incoherent; `wrap()` rule recorded.
- **Task instruction:** the behavioral `build_prompt()` instruction (SIGNED decision); English Stage 0.3 instruction RETIRED.

**Byte-identity verification (`build_phase1_materials.py`):**
- **Between cells — VERIFIED ✓.** C1 differs only by the disclosure (appears in both payload positions per whole-file substitution — faithful, Stage 0.1b-confirmed); C2 differs only by the wrapper. Cell sha256 (v12): flagged×L1 `41748108…`, plausible×L1 `bf58c55e…`, incoherent×L4 `685521ca…`, incoherent×L1 `0ab62edc…`.
- **⚠ Against source (sealed R2 corpus) — MISMATCH / BLOCKER.** Chat-pasted wrappers do NOT reproduce source sha256: L1 mine `46ca4e38…`(no-nl)/`11301399…`(nl) vs source `1f9bb56c…`; L4 mine `7c541488…`/`95b473d9…` vs source `1100ec4f…`. Paste is not byte-faithful. **P1/P2 are BLOCKED until materials are transferred byte-exact (saved sha256 == source sha256).** The sealed R2 operative lists were computed on the original bytes; running on divergent bytes would break the seal's echo guarantee. Awaiting byte-exact transfer (e.g. base64).

---

## Architecture change — single-machine canonical materials + byte-fidelity RESOLVED — 2026-07-21

**Recorded:** 2026-07-21, `[host]` (RTX 5090). **Supersedes the 2026-07-20 byte-fidelity BLOCKER above.** PI decision: eliminate the two-machine dependency — the reification-gradient materials now live **in-repo, tracked, byte-exact**, so no second machine ever supplies them.

### Read-only reference clone (hard rule: no writes, no push against it)

| Item | Value |
|---|---|
| Repo | `github.com/bernalsuarezantonio-bit/reification-gradient` (private) |
| Clone path | `C:\Users\EDITOR\Desktop\reification-gradient` — **sibling of** workspace-program, **outside** it |
| Access | **new** repo-scoped SSH **deploy key**, ed25519, generated on this host; **read-only** (write access NOT granted on GitHub). `id_ed25519_reification_ro`; SSH alias `github-reification`; fingerprint `SHA256:kLGVcbo8IUPBFIa4BHpCGjehphX6kiy29CfYOkqY70M`. (The existing `id_ed25519_lab5090` is bound to workspace-program only — GitHub forbids reusing a deploy key across repos.) |
| Discipline | Operative read-only: only `git clone`/`fetch`/`cat-file`/`rev-parse` and file reads were ever run against this clone. No commit, no write, no push. |

### Verification (all GREEN before any copy)

| Check | Expected | Observed | |
|---|---|---|---|
| `HEAD` | contains `ee23c07` | `ee23c07288a31eb19545c944e0662bd6a2d9d915` | ✓ |
| tag `prereg-v1` (annotated) | resolves to `4b2464f` | tag obj `1e67b02e…` → `^{commit}` `4b2464fd3c016a9cd21c4d8e450cbe4fd8d057dd` | ✓ |
| `L1_forum.md` sha256 | `1f9bb56c…` | `1f9bb56c…` (git blob, LF) | ✓ |
| `L4_preprint.md` sha256 | `1100ec4f…` | `1100ec4f…` | ✓ |
| `vignettes.yaml` sha256 | `59f37915…` | `59f37915…` | ✓ |
| `disorders.yaml` sha256 | `91d0ccb7…` | `91d0ccb7…` | ✓ |

**Line-ending note (critical).** The clone's `core.autocrlf=true` rewrites the *working tree* to CRLF on checkout, so working-tree sha256 differ. The sealed hashes are on **LF bytes** (iMac origin). All hashing/copying used the **committed git-blob bytes** (`git cat-file blob HEAD:…`), which are LF and match exactly.

### Canonical copy (tracked) + regeneration

- Four LF blobs copied to **tracked** `phase1/materials_canonical/` (`vignettes.yaml`, `disorders.yaml`, `legitimacy/L1_forum.md`, `legitimacy/L4_preprint.md`). Pinned `-text` in `.gitattributes` (`phase1/materials_canonical/** -text`) so autocrlf never renormalizes them; **staged-blob content re-verified** = source sha on both repos (workspace-program is also `autocrlf=true`).
- `phase1/scripts/build_phase1_materials.py` rewritten **canonical-driven**: reads only the tracked YAMLs + wrapper files, mirrors the runner's `load()`/`wrap()`/`build_prompt()` verbatim (`src/run_experiment.py @ ee23c07`; wrappers read raw incl. trailing newline; vignette `text` folded scalar `.strip()`ed). Regenerates the 20 `high` vignettes + condition texts + wrapper working-copies into gitignored `phase1/materials/`. Manifest `STATUS: GREEN`.

### 4-cell verifier (v12) — re-run, GREEN

Byte-faithful cell sha256 (supersede the 2026-07-20 BLOCKER-era values, which were computed on non-faithful paste):

```
C1_DN_flagged_L1    6d5c87fb29805ce1…  (1211 chars)
C1_DN_plausible_L1  84cb31ea09633853…  (1013 chars)
C2_incoherent_L4    cc80e1dc4f9d3719…  (1016 chars)
C2_incoherent_L1    a7a599dc50c05505…  ( 995 chars)
```

- **Canonical fidelity:** all 4 tracked files MATCH source sha256.
- **v12** regenerated from YAML reproduces sealed pilot `32c78f5f…` ✓.
- **C1 differs only by the disclosure** (`{{PAYLOAD}}` occurs **2×** in L1 — both substituted; `DN_flagged = DN_plausible payload + " " + disclosure`) ✓.
- **C2 differs only by the wrapper** (identical instruction+CASO tail; `seeded(L4) ≠ seeded(L1)`) ✓.

**Blocker CLEARED.** Stimuli are now byte-identical to the sealed R2 corpus source. No condition-bearing readout created; seal integrity intact. **STOP — staged for PI review + tag** (delegate does not tag). New tracked changes: `phase1/materials_canonical/**`, `.gitattributes`, `phase1/scripts/build_phase1_materials.py`, `PROVENANCE.md`, `PREREG_PHASE1.md`.

---

## Stage P1 — Confirmatory J-lens runs + judge (data generation) — 2026-07-21

**Recorded:** 2026-07-21, `[host]` (RTX 5090). This is the **data commit** the Phase 1 analysis runs against. **No analysis here** — no loadings, no aggregation, no counting beyond mechanical completeness. Numbers below are integrity counts only.

**Freeze→data chain:** tag `prereg-phase1-v1` → commit `109166a` → `PREREG_PHASE1.md` sha256 `bedbcc78f9dc261ffd789dc55097bbd9c997c4db9d3e63077489623ae05f55d5`. Harness at `d508662`. Materials byte-exact from `phase1/materials_canonical/` (canonical fidelity GREEN).

**Instrument (harness `d508662`):** `phase1/scripts/run_confirmatory.py` (generation + readout dump), `judge_diagnosis.py` (gemma2:27b v1 blinded rubric), `verify_completeness.py`. Prompt assembly byte-faithful to reification-gradient `src/run_experiment.py @ ee23c07` (`wrap()`/`build_prompt()`), signed Spanish instruction.

| Item | Value |
|---|---|
| Cells | 4 confirmatory: `C1_DN_flagged_L1`, `C1_DN_plausible_L1`, `C2_incoherent_L4`, `C2_incoherent_L1` |
| Design | 4 cells × 20 `high` vignettes × 10 reps = **800 runs** (each cell 200; 20×10 verified) |
| Model | `Qwen/Qwen2.5-7B-Instruct@a09a3545` (fp16) |
| Lens | `neuronpedia/jacobian-lens@16a01f3` `qwen2.5-7b-it`; `.pt` sha256 `3b3ab44c…` re-verified == Stage 0.2 pin (issue-#6 isfinite lens) |
| Readouts | top-k=10, all 27 fitted layers × all positions + model_output row; prompt/generation marked; first-16 `ood_unfitted_pos` flagged (issue-#5 Pitfall 2) |
| Generation | `do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, max_new_tokens=200` (Qwen defaults at temp 0.7; prereg §Generation: temp 0.7 / sampling / num_predict 200). Per-run seed recorded (`SEED_BASE=700000+canonical_index`); execution order shuffled with `master_seed=20260721` |
| Judge | `gemma2:27b` (Ollama, temp 0, seed 0, `format=json`), v1 blinded rubric (`materials_canonical/scoring/judge_prompt.md` `e67e8e63…`), category name aliased to "LA CATEGORÍA" in payload + response; 6 sealed keys |

**Mechanical completeness (`completeness_report.json`, COMPLETE=true):** 800 runs, **0 duplicate trial_ids**, **0 runs with problems** (every readout's sha256 re-verified == manifest; top-k weights finite; prompt+generation positions present). Judge: 800 judged, **1 parse error** (in `C2_incoherent_L1`, both attempts failed → excluded + counted per prereg §6b; C2 does not condition on diagnosis). Single consistent lens `.pt` sha + model digest across all 800.

**Storage:** raw per-run readouts (~800 × ~10 MB ≈ 8 GB) in gitignored `phase1/data/readouts/`; committed here are the lightweight `run_manifest_full.jsonl` (per-run seed + readout sha256 + digests), `judge_full.jsonl`, `completeness_report.json`, and the smoke provenance. **Data content digest** (sha256 over the two full manifests + completeness): `dc522361096bae30377ecf05d37142cfcb3f52fbb6349c77825bea455f0fb8f1`.

**Timing:** smoke 8-run gate ~1.1 min gen + 1.2 min judge (PASS, 4 criteria); full generation **98.8 min**; full judge **44.9 min**. Ollama coexistence: `llama3.3:70b` then `mistral-sim` blocked VRAM for ~2.5 h pre-run (documented residual auto-reload); during the run `gemma2:27b` pinned `Forever` and spilled to CPU (16%/84%) rather than OOM-ing the lens — cost ~30% run speed, no data loss. **No aggregates computed. Analysis is a separate session against this commit.**

---

## Incident #3 — phantom `RESULTS_PHASE1.md` + unsupported "recognition-as-echo" claim — 2026-07-22

**Recorded:** 2026-07-22, analysis session. **Data untouched** — commit `a715ce4` / digest `dc522361` re-verified intact (Gate 0 GREEN); this incident concerns a **reporting/provenance** gap, not the data.

**What happened.** A prior chat report asserted that Phase 1 analysis was complete, that `RESULTS_PHASE1.md` had been committed, and summarized the finding as *"recognition-as-echo: fictional-status (Set F) loading in the workspace is fully explained by emission echo; no evidence of sustained holding."* On cold inspection:
1. **`RESULTS_PHASE1.md` did not exist in the repo** at HEAD (`git cat-file -e HEAD:RESULTS_PHASE1.md` → absent). No results artifact was ever committed.
2. **No loading/aggregate/test numbers were ever committed.** The Stage-P1 data-commit section above states verbatim "No aggregates computed"; the repo contained only mechanical completeness counts. The "echo / no holding" conclusion had **no committed numerical basis**.

**Action taken (per standing rule: re-derive in cold; commit the artifact before citing it; do not reconcile).** The confirmatory analysis was re-derived from the committed readouts by `phase1/scripts/analyze_phase1.py` (frozen §2 aggregation, identical to `phase0/…/phase1_p0_power.py:loading_for_rep`), Gate 0 asserted in-process. Outputs committed: `RESULTS_PHASE1.md`, `phase1/data/results_phase1.json`, the analysis script.

**Discrepancy vs the prior chat claim (reported, not reconciled).** The cold numbers do **not** support "fully explained by emission echo; no evidence of sustained holding" as stated:
- **C1 confirmatory (Set F, diagnosis=1, two-sided):** flagged **0.2506** vs plausible **0.0418**, paired diff **0.2088**, **t(19)=11.19, p=8.4e-10** — strongly **positive** (the registered *loaded-but-inert* direction). Set F **is** loaded during diagnosis; a positive loading is informative under the R5 clause.
- **The registered emission-mask robustness check (exploratory sub-3) is inconclusive *by construction* here:** only **~0.24%** of generation positions are maskable, because generation is **Spanish** while the Set F SURVIVOR operative tokens (and the lens readout) are **English** — the mask cannot remove the emission channel. The flagged>plausible difference is therefore unchanged (0.2096), which establishes **neither** "echo" **nor** "not-echo". Any prior claim that masking demonstrated pure echo is unsupported.
- **Run-level mention split (sub-2), the actually echo-relevant statistic:** F loading concentrates in fiction-verbalizing runs — with-mention **0.448** (n=92) vs without-mention **0.0825** (n=108) vs plausible **0.0418**. The without-mention stratum sits only ~2× the plausible floor. This is consistent with a **substantial emission-linked component** but not with a clean "fully explained / no holding" verdict; the without-mention stratum is not at floor.
- **C2 confirmatory (Set A, one-sided L4>L1):** effect is strongly **opposite** to the directional hypothesis — L4 **1.66** < L1 **2.45**, t(19)=-10.0, one-sided p=1. H1 not supported.

**Adjudication deferred.** Per the asymmetric-informativeness clause, interpretation (echo vs sustained holding; the informativeness of the positive C1 loading; the opposite-signed C2) is the PI's. This note records only that the prior chat narrative was stronger than any committed evidence, that no artifact backed it, and that the authoritative numbers now live in a committed, script-reproducible artifact. **The Phase 2 premise ("Phase 1 found recognition-as-echo … no sustained holding") should be re-examined against `RESULTS_PHASE1.md` before proceeding.**
