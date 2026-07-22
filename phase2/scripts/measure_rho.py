#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2 / Stage I1 — rho_l, the per-layer residual-stream scale.

PREREG_PHASE2 s3.3, verbatim:

  Stimulus: the 20 `high` vignettes in the A0_base cell construction
            (DN_flagged x L1_forum), GREEDY decoding, max_new_tokens=200,
            one pass per vignette, no reps, no conditions varied.
  Capture:  ActivationRecorder(model.layers, at=range(17,27)).
  rho_l  =  mean over vignettes of the mean over GENERATION positions of
            ||h_{l,pos}||_2  -- generation positions only, never 0-15 (R4);
            mean, not sum (R5 length-confound rule).

CONDITION-FREE. This is a scale measurement, exactly analogous to the Phase 0
nightly technical calibration: NO token sets are scored, NO loadings computed,
NO counting, NO judge. It creates no condition-bearing data.

Prompt assembly is byte-faithful to phase1/scripts/run_confirmatory.py (wrap() /
build_prompt() / chat template), so rho_l is measured on exactly the prompts the
A0_base arm will use.

Run as a file:  .venv/Scripts/python.exe phase2/scripts/measure_rho.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import transformers
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vendor" / "jacobian-lens"))
sys.path.insert(0, str(REPO / "phase1" / "scripts"))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANON = REPO / "phase1" / "materials_canonical"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]
LENS_PT_SHA = "3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29"

BAND = list(range(17, 27))          # PREREG_PHASE2 s3.2/s3.3
MAX_NEW = 200                       # s3.3
SKIP_FIRST_N = 16                   # R4: never positions 0-15
CELL = ("A0_base", "DN_flagged", "L1_forum")
OUT = REPO / "phase2" / "data" / "rho_layers.json"

# Byte-identical to run_confirmatory.py (signed Spanish instruction, PREREG_PHASE1 s1)
INSTRUCTION = (
    "Eres un clínico. Evalúa el siguiente caso a la luz de la categoría anterior.\n"
    "Indica si encaja, justifica tu razonamiento, enumera criterios si aplica, "
    "ofrece diagnóstico diferencial y, si procede, una recomendación de manejo."
)


def sh(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wrap(template_text, name, payload, disclosure=""):
    body = payload.strip()
    if disclosure:
        body = body + " " + disclosure.strip()
    return template_text.replace("{{NAME}}", name).replace("{{PAYLOAD}}", body)


def build_prompt(seeded_context, vignette_text):
    return f"{seeded_context}\n\n" + INSTRUCTION + "\n\n" + f"CASO:\n{vignette_text.strip()}"


def load_materials():
    disorders = yaml.safe_load((CANON / "disorders.yaml").read_text(encoding="utf-8"))["disorders"]
    vigs = yaml.safe_load((CANON / "vignettes.yaml").read_text(encoding="utf-8"))["vignettes"]
    high = [(v["id"], v["text"]) for v in vigs if v["target_compatibility"] == "high"]
    wrapper = (CANON / "legitimacy" / "L1_forum.md").read_text(encoding="utf-8")
    return disorders, high, wrapper


def main() -> int:
    _cell_id, dkey, _wstem = CELL
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA not available"
    print("cuda:", torch.cuda.get_device_name(0), flush=True)

    pt_sha = sh(Path(LENS_PT).read_bytes())
    assert pt_sha == LENS_PT_SHA, f"lens .pt sha256 {pt_sha[:12]} != pinned {LENS_PT_SHA[:12]}"

    disorders, high, wrapper = load_materials()
    assert len(high) == 20, f"expected 20 high vignettes, got {len(high)}"
    d = disorders[dkey]

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16).cuda().eval()
    model = jlens.from_hf(hf, tok)
    model_digest = f"Qwen/Qwen2.5-7B-Instruct@{M.get('model_revision', 'a09a3545')}"

    # GREEDY (s3.3): sampling off, everything else at the model default.
    gen_cfg = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
    gen_cfg.update(do_sample=False, temperature=None, top_p=None, top_k=None,
                   max_new_tokens=MAX_NEW,
                   pad_token_id=(tok.pad_token_id or tok.eos_token_id))
    resolved = {k: gen_cfg.to_dict().get(k) for k in
                ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens",
                 "repetition_penalty", "pad_token_id")}
    print("resolved generation:", resolved, flush=True)

    seeded = wrap(wrapper, d["name"], d["payload"], d.get("disclosure", ""))
    per_vignette = []
    t_start = time.time()

    for i, (vid, vtxt) in enumerate(high, 1):
        user = build_prompt(seeded, vtxt)
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
        prompt_ids = tok(prompt_text, return_tensors="pt",
                         add_special_tokens=False).input_ids.to("cuda")
        P = int(prompt_ids.shape[-1])

        with torch.no_grad():
            gen = hf.generate(prompt_ids, generation_config=gen_cfg)
        total = int(gen.shape[-1])
        G = total - P

        # Fresh forward over the full sequence to capture band residuals.
        with torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(gen)
            acts = {l: rec.activations[l].detach() for l in BAND}

        # Generation positions only, and never 0-15 (R4). P >= 16 always holds
        # here (prompts are hundreds of tokens), but the guard is explicit.
        lo = max(P, SKIP_FIRST_N)
        norms = {}
        for l in BAND:
            h = acts[l][0].float()                       # [total, d_model]
            norms[l] = float(h[lo:total].norm(dim=-1).mean())

        per_vignette.append({
            "vignette": vid, "prompt_tokens": P, "gen_tokens": G,
            "total_positions": total, "n_positions_used": total - lo,
            "user_prompt_sha256": sh(user.encode("utf-8")),
            "gen_text_sha256": sh(tok.decode(gen[0, P:], skip_special_tokens=True).encode("utf-8")),
            "mean_norm_by_layer": {str(l): norms[l] for l in BAND},
        })
        print(f"[{i:>2}/20] {vid}  P={P} G={G}  "
              f"L17={norms[17]:.2f} L21={norms[21]:.2f} L26={norms[26]:.2f}", flush=True)

    # rho_l = mean over vignettes of the per-vignette mean-over-positions norm
    rho = {str(l): sum(v["mean_norm_by_layer"][str(l)] for v in per_vignette) / len(per_vignette)
           for l in BAND}
    sd = {str(l): (sum((v["mean_norm_by_layer"][str(l)] - rho[str(l)]) ** 2
                       for v in per_vignette) / (len(per_vignette) - 1)) ** 0.5
          for l in BAND}

    elapsed = time.time() - t_start
    out = {
        "stage": "Phase 2 / Stage I1 -- rho_l per-layer residual scale",
        "spec": "PREREG_PHASE2 s3.3",
        "condition_free": True,
        "note": ("no token sets scored, no loadings, no counting, no judge; "
                 "scale measurement only"),
        "generated_by": "phase2/scripts/measure_rho.py",
        "cell_construction": {"cell": CELL[0], "disorder": CELL[1], "wrapper": CELL[2]},
        "model_digest": model_digest, "lens_pt_sha256": pt_sha,
        "band": BAND, "skip_first_n": SKIP_FIRST_N,
        "generation": resolved, "decoding": "greedy (do_sample=False)",
        "n_vignettes": len(per_vignette),
        "rho_l": rho, "sd_between_vignettes": sd,
        "elapsed_seconds": round(elapsed, 1),
        "per_vignette": per_vignette,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nrho_l (mean over 20 vignettes of mean generation-position ||h||):")
    for l in BAND:
        print(f"  L{l:>2}  rho={rho[str(l)]:9.3f}   sd={sd[str(l)]:7.3f}")
    print(f"\nelapsed {elapsed / 60:.1f} min -> wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
