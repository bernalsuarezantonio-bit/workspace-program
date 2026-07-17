# Stage 0.1 — Environment & Repository Reconnaissance

**Date:** 2026-07-17
**Author:** Claude Code (Phase 0 delegate)
**Status:** STOP POINT 1 — awaiting PI decision
**jacobian-lens commit:** `581d398613e5602a5af361e1c34d3a92ea82ba8e` ("Initial release", 2026-07-02)
**Log:** `phase0/logs/20260717T102333_stage01_recon.log`

Nothing heavy was run. No model was downloaded, no lens was fitted, no GPU work occurred.

---

## 0. Headline: the brief's hardware premise does not hold on this machine

The brief specifies an **RTX 5090, 32 GB VRAM, CUDA 12.8+, cu128 PyTorch wheels**, and a GPU shared with an Ollama server. This machine is none of those:

| Brief assumes | This machine actually is |
|---|---|
| RTX 5090, 32 GB VRAM | Apple M1, 8-core integrated GPU, **16 GB unified memory** |
| CUDA 12.8+, cu128 wheels | **No CUDA at all** — `nvidia-smi` and `nvcc` both not found; `torch.cuda.is_available()` = `False` |
| Linux/NVIDIA stack | Darwin 25.5.0 `arm64` (`Admins-iMac.local`); MPS backend available |
| GPU shared with Ollama | **Ollama is not installed here** — no binary, nothing on `localhost:11434` |

Everything downstream in Phase 0 (Stages 0.2 and 0.3) is scheduled against compute that does not exist at this location. This is the one blocking finding, and it is a question only you can answer: **is the RTX 5090 a separate machine that Claude Code has not been pointed at?** If so, Phase 0 is running in the wrong place and the recon below should be re-run there before Stage 0.2. If instead this iMac is the intended host, the model choice changes materially — see §5.

Two secondary environment facts:

- **System Python is 3.9.6**; `jlens` requires `>=3.10`. Not a blocker — `uv` 0.10.7 is present and can build a pinned 3.11/3.12 venv in Stage 0.2.
- **System torch is 2.8.0** (CPU/MPS). Irrelevant once an isolated venv is created, but note the version pinned in the venv must be an arm64/MPS wheel, **not** a cu128 wheel, if this machine is the host. The cu128 requirement in the brief is meaningless on Apple silicon.

---

## 1. Which models does the repo support out of the box?

The library is **architecture-generic, not model-specific**. `jlens/hf.py` targets the modern HuggingFace decoder layout and names Llama / Qwen / Mistral / Gemma / OLMo / StableLM as conforming; `jlens.from_hf(hf_model, tokenizer)` wraps any such model. The README says "Examples use Qwen; other HuggingFace decoders adapt cleanly."

So "support" is best read as: *any HF decoder whose layer layout matches*, with Qwen being the documented/exercised path.

## 2. Are precomputed averaged Jacobians distributed?

**Yes — and this is the most consequential finding after the hardware.** Fitting is very likely unnecessary.

The walkthrough pulls pre-fitted lenses from the HuggingFace Hub:

- **Repo:** `neuronpedia/jacobian-lens` (public, not gated)
- **Revision used by the walkthrough:** `qwen-n1000` → sha `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`
- **Repo `main` sha:** `a4114d7752d11eb546e6cf372213d7e75526d3a1` (lastModified 2026-07-06)

That revision carries lenses for **38 model directories**, each with a `.pt` lens, a `convergence.csv`, and a `config.yaml`, all fitted on `Salesforce/wikitext`:

> gemma-2 (2b, 2b-it, 9b, 9b-it, 27b) · gemma-3 (270m, 270m-it, 1b, 1b-it, 4b, 4b-it, 12b, 12b-it, 27b, 27b-it) · gemma-4 (e2b, e4b, 31b) · gpt2-small · gpt-oss-20b · llama3.1-8b, llama3.1-8b-it, llama3.3-70b-it · olmo-3 (1025-7b, 1125-32b) · pythia-70m-deduped · **qwen2.5-7b-it** · **qwen3-1.7b**, qwen3-4b, qwen3-8b, qwen3-14b, qwen3-32b · qwen3.5-0.8b, qwen3.5-2b-pt, **qwen3.5-4b**, qwen3.5-9b-pt, qwen3.5-27b · qwen3.6-27b

The walkthrough itself defaults to `Qwen/Qwen3.5-4B` (with `Qwen/Qwen3.6-27B` commented out) — **not** the Qwen-1.5B / Qwen-7B named in the brief.

**The brief's pilot candidates don't exist under those names.** The closest available lenses are:

| Brief's candidate | Closest shipped lens | Note |
|---|---|---|
| Qwen-7B | `qwen2.5-7b-it` (Qwen2.5-7B-Instruct) | **Same family as the behavioral study's `qwen2.5:32b`** |
| Qwen-1.5B | `qwen3-1.7b` (Qwen3-1.7B) | Different generation from the behavioral study |

I flag one point for your decision and am explicitly not making it: `qwen2.5-7b-it` is the only shipped lens in the **Qwen2.5** family, which is the family the reification-gradient behavioral study actually ran (`qwen2.5:32b`). The brief already anticipates a follow-up behavioral mini-run on the smaller model "to restore the two-levels-one-substrate link." Picking a Qwen2.5 model keeps that link within one family; picking Qwen3/3.5 crosses a generation boundary. That is a design call, so it is yours.

## 3. If local computation is needed: corpus, compute, VRAM, time

Recorded for completeness, but **if a shipped lens covers the chosen model, none of this is on the critical path.**

- **Upstream default corpus:** `Salesforce/wikitext`, `wikitext-103-raw-v1`, streamed; paper lenses use **1000 sequences × 128 tokens**. README: "Quality saturates quickly (§9.3); ~100 prompts is usable."
- **Cost structure** (from `jlens/fitting.py`): per prompt, **one forward pass + `ceil(d_model / dim_batch)` backward passes** (`dim_batch=8` by default). The forward is replicated `dim_batch` times along the batch axis and the graph is retained across all backward passes.

That per-prompt backward count is the thing to notice:

| Model | d_model (approx) | backward passes/prompt @ dim_batch=8 | × 1000 prompts |
|---|---|---|---|
| Qwen3-1.7B | 2048 | 256 | **256,000** |
| Qwen2.5-7B-Instruct | 3584 | 448 | **448,000** |

The README is candid that this is "a reference implementation and is not optimized; fitting time is dominated by the model's own backward pass." I am **not** giving you a wall-clock estimate for a 32 GB CUDA card, because I have no such card to measure on and an invented number would be worse than none. What I can say concretely:

- **Fitting either model on this M1/16 GB machine is not realistic.** Qwen2.5-7B in bf16 is ~15 GB of weights alone, before the retained graph and the `dim_batch`-replicated activations — it will not fit in 16 GB shared with the OS. Qwen3-1.7B (~3.4 GB bf16) would fit in memory, but 256k backward passes through the M1's 8-core GPU is a multi-day proposition at best.
- **Jacobian accumulator size** (CPU RAM, not GPU): `n_source_layers × d_model² × 4` bytes — ~0.9 GB for Qwen3-1.7B (28 layers), ~1.4 GB for Qwen2.5-7B (28 layers). Checkpoints are written at this size; `checkpoint_every=1` by default, which the docstring itself warns to raise for large models.

**Applying** a pre-fitted lens is a completely different cost class: one forward pass plus a `[d_model, d_model]` matmul per layer. Qwen3-1.7B apply-only is plausibly feasible on this machine; Qwen2.5-7B apply-only in bf16 (~15 GB) is borderline-to-impossible in 16 GB unified memory.

## 4. Dependencies and cu128 compatibility

From `pyproject.toml`: `torch`, `huggingface_hub`, **`transformers>=5.5`**, `numpy`; dev extras `pytest`, `ruff`, `datasets`. `requires-python = ">=3.10"`. A `uv.lock` (556 KB) is committed, so the upstream resolution is fully pinnable.

- `transformers>=5.5` is satisfiable — PyPI latest is **5.14.1** (`requires_python >=3.10.0`).
- **No dependency is inherently incompatible with cu128 wheels.** None of the four pin a CUDA-specific build; `torch` is unconstrained, so the wheel index decides. On a real cu128 host this resolves normally.
- The incompatibility on *this* machine is the mirror image: **cu128 wheels do not exist for arm64 macOS**. If this iMac is the host, the venv must use standard arm64 wheels and the MPS backend.
- Portability caveat if MPS is used: the walkthrough hard-codes `.cuda()` (`hf_model = ...from_pretrained(...).cuda()`). Running on MPS requires changing device placement. I consider that a **port, not a customization** of fitting settings, but I am not making that change without your say-so, and it carries real risk — `torch.autograd.grad` coverage on MPS is not guaranteed for every op in these models, and fitting on MPS is untested upstream.

## 5. What format do J-lens readouts take?

`JacobianLens.apply(model, prompt, *, layers=None, positions=None, max_seq_len=512, use_jacobian=True)` returns a triple `(lens_logits, model_logits, input_ids)`:

- **`lens_logits`**: `dict[int, Tensor]` — **per-layer**, each `[n_positions, vocab_size]`.
- **`model_logits`**: the model's actual final-layer logits at the same positions, same shape.
- **Per-token-position**: yes — `positions` takes Python indexing (negatives count from the end); `None` returns every position.
- **Top-k**: **not** the native format. The lens returns **full dense logits over the whole vocabulary**; top-k is the caller's choice (`logits.topk(5).indices`, then `tokenizer.decode`). Stage 0.3's "top-k tokens + weights" dump is therefore a reduction I would apply at write time — the raw object is far larger.
- `use_jacobian=False` gives the vanilla logit-lens baseline, i.e. a built-in control that skips the `J_l` transport.
- The visualisation layer (`jlens.vis.compute_slice` / `build_page`) renders a layer × position grid of top-1 tokens with vocabulary ranks.

**Volume warning for Stage 0.3:** dense logits are `n_layers × n_positions × vocab_size`. For Qwen (vocab ~152k) at, say, 28 layers × 300 positions, that is ~1.3 G floats — ~2.5 GB in fp16 **per run** if dumped raw. Reducing to top-k at write time is essential, and the choice of k is a measurement decision I'm flagging rather than making.

## 6. Open issues that look blocking

Six open items (4 issues, 2 PRs; repo is explicitly "not maintained and not accepting contributions", 1418 stars). Two matter to us, and one of them matters a great deal.

### Issue #5 — "Two easy-to-hit measurement pitfalls when reading `apply()` output as evidence of internal state" (open, 2026-07-13)

**This is the single most important thing I found for your study, and it independently corroborates your own R2b rule.**

> **Pitfall 1 — input-copying ceiling.** "If a probed token appears anywhere in the prompt, the lens reads it at rank ~1 at that position — the readout reflects the input token, not workspace content." The reporter's pipeline "silently saturated every measurement: erased/retained experimental conditions became identical to three decimals, at salience exactly `ln(vocab_size)`." They add: "Anyone measuring 'is concept X on the model's mind' over a prompt that mentions X will get a trivially confounded yes."

Your R2b rule — confirmatory sets restricted to tokens absent from **all** stimuli — is exactly the mitigation for this failure mode, arrived at independently. Stage 0.1b's inventory is what makes it mechanically enforceable. This is convergent evidence that R2b is load-bearing rather than merely conservative, and I'd treat it as strengthening the case for sealing token sets before any measurement runs.

> **Pitfall 2 — unfitted-position readouts.** `fit()` skips the first `SKIP_FIRST_N_POSITIONS` (= 16) positions, but `apply()` "happily returns readouts at positions 0–15, where the lens is out-of-distribution." Reported as "inflated and unstable," with short prompts the trap: "a 25-token prompt gets 60% of its positions from the unfitted region."

I confirmed both against the source: `SKIP_FIRST_N_POSITIONS = 16` is enforced in `fitting.valid_position_mask` and has **no counterpart in `apply()`**. This directly shapes Stage 0.3 and any Phase 1 design — vignette prompts are long, so the fraction of unfitted positions is small, but the first 16 positions of every run are in that region and the wrapper text (L1–L5) is what occupies them.

Neither pitfall is a code bug; both are silent-confound traps. Not blocking for feasibility, decisive for design.

*Provenance note:* issue #5 links a third-party fork (`GNS-Foundation/jacobian-lens`) with a fuller "pitfall ledger." I have not fetched or acted on it — it is unvetted third-party content, and its claims would be data to evaluate, not instructions to follow. Say the word if you want it retrieved and summarised.

### Issue #6 — "`save()` silently overflows large finite Jacobians to `inf` with default float16" (open, 2026-07-14)

Confirmed in source: `JacobianLens.save(path, *, dtype=torch.float16)` casts via `J.to(dtype)` with **no range check**. The docstring asserts "entries are O(1) so the range is not a constraint" — the reporter's counter-example is a real fit (LoRA-finetuned Qwen3-1.7B) with per-layer maxima of ~1.17e16–5.43e16, where all 4,194,304 entries in layers 0–16 became `inf` after save, and downstream products became `NaN`. Tested against **our exact commit** `581d398`.

**Blocking only if we fit locally.** Since shipped lenses cover our candidates, the practical exposure is low — but if any lens (downloaded or fitted) is ever used, a `torch.isfinite()` check after load costs nothing and I'd recommend it as a standing guard. I have not written that check yet; it's a code change beyond this stage's remit.

The rest — #4 (`canonical_digest`), #3 (`from_fit_checkpoint`), #2 (swap-intervention interest), #1 (aiohttp bump) — are non-blocking. #4 is mildly interesting for preregistration sealing later, since a content digest of a fitted lens is exactly the kind of thing a seal wants to pin.

## 7. Can the J-lens pipeline coexist with the Ollama server?

**Moot on this machine — there is no Ollama here** (no binary; `localhost:11434` unreachable), and no NVIDIA GPU to contend over. The coexistence question belongs to whatever machine actually hosts the RTX 5090, which I have not been able to inspect.

Answering in principle, for that host: **fitting** would need an exclusive window — it retains a graph across hundreds of backward passes with the prompt replicated `dim_batch` times, so peak memory is large and sustained, and Ollama's own VRAM reservation would collide with it. **Applying** a pre-fitted lens is much lighter (one forward + per-layer matmul) and could plausibly share a card with a small Ollama model, though co-residency still risks OOM if Ollama loads a large model mid-run. Since §2 suggests we can skip fitting entirely, the scheduling problem may largely evaporate — but I can't verify any of this without access to that machine.

---

## 8. What I did not do

Per the brief's rules: no token sets proposed or filtered (rule 4); nothing run beyond metadata reads and one clone (rule 3 of Stage 0.1); the `reification-gradient` repo was **listed read-only only** — no files copied yet, no git commands issued against it, nothing opened for writing.

## 9. Decisions needed from you (STOP POINT 1)

1. **Where does Phase 0 actually run?** This machine has no CUDA GPU. Is the RTX 5090 elsewhere? This gates Stages 0.2–0.3 entirely.
2. **Which model?** The brief's Qwen-1.7B/7B don't exist under those names. Shipped-lens candidates: `qwen2.5-7b-it` (same family as the behavioral `qwen2.5:32b`), `qwen3-1.7b`, `qwen3.5-4b` (the walkthrough default). If this iMac is the host, 7B is likely out on memory grounds and the realistic candidates are `qwen3-1.7b` or possibly `qwen3.5-4b`, apply-only.
3. **Pre-fitted lens or local fit?** I recommend pre-fitted — it removes the dominant compute cost and the issue-#6 exposure. Confirm the `qwen-n1000` revision pin is acceptable.
4. **Noted for Phase 1, not for me to decide:** issue #5's pitfall 2 (unfitted positions 0–15) interacts with where wrapper text sits in the prompt.

Stage 0.1b (R2b stimulus scan, no GPU) is unblocked by all of the above and I can start it on your go-ahead.
