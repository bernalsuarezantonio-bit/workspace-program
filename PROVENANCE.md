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
