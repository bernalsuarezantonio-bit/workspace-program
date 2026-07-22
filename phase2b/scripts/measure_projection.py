#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b / Stage J0 — how much of the residual actually lies along v_hat?

Condition-free feasibility measurement, taken BEFORE the 2b design is frozen.
This is the direct correction of what killed Phase 2: there, a frozen dose
parameter turned out mis-scaled by two orders of magnitude and the pilot only
discovered it after the tag. For a projection ablation the analogous quantity
is the fraction of the residual that the ablation actually removes:

    frac_l = |h . v_hat_l| / ||h||     at each band layer, generation positions

If frac_l is tiny, full projection is a null manipulation BY CONSTRUCTION and
the design is not worth freezing. If it is appreciable, the ablation has
something to remove.

Also reports the ANGLE-preserving quantity ||h - (h.v_hat)v_hat|| / ||h|| (the
residual norm retained), and the same for the 50% partial ablation.

Same construction as Phase 2 (carried forward unchanged): u_gain target,
Tikhonov lambda = 0.1, band 17-26, generation positions only, never 0-15.
Stimulus is the A0_base / B0_none cell (DN_flagged x L1_forum x high), greedy.

NO conditions varied, NO token sets scored, NO loadings, NO counting, NO judge.

Run as a file:  .venv/Scripts/python.exe phase2b/scripts/measure_projection.py
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
sys.path.insert(0, str(REPO / "phase2" / "scripts"))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from intervene import BAND, LENS_PT_SHA, build_vhat, f_survivor_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANON = REPO / "phase1" / "materials_canonical"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]
OUT = REPO / "phase2b" / "data" / "projection_feasibility.json"
SKIP_FIRST_N, MAX_NEW = 16, 200

INSTRUCTION = (
    "Eres un clínico. Evalúa el siguiente caso a la luz de la categoría anterior.\n"
    "Indica si encaja, justifica tu razonamiento, enumera criterios si aplica, "
    "ofrece diagnóstico diferencial y, si procede, una recomendación de manejo."
)


def sh(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    assert torch.cuda.is_available(), "CUDA not available"
    pt_sha = sh(Path(LENS_PT).read_bytes())
    assert pt_sha == LENS_PT_SHA, "lens .pt sha mismatch"

    disorders = yaml.safe_load((CANON / "disorders.yaml").read_text(encoding="utf-8"))["disorders"]
    vigs = yaml.safe_load((CANON / "vignettes.yaml").read_text(encoding="utf-8"))["vignettes"]
    high = [(v["id"], v["text"]) for v in vigs if v["target_compatibility"] == "high"]
    wrapper = (CANON / "legitimacy" / "L1_forum.md").read_text(encoding="utf-8")
    d = disorders["DN_flagged"]
    body = d["payload"].strip() + " " + d.get("disclosure", "").strip()
    seeded = wrapper.replace("{{NAME}}", d["name"]).replace("{{PAYLOAD}}", body)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    W_U = hf.get_output_embeddings().weight.detach().float().cpu()
    gain = hf.model.norm.weight.detach().float().cpu()
    vhat, landing = build_vhat(lens, W_U, gain, f_survivor_ids())
    print(f"landing band_min_cos={landing['band_min_cos']:.4f} (lambda=0.1, u_gain)")

    gcfg = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
    gcfg.update(do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=MAX_NEW, pad_token_id=(tok.pad_token_id or tok.eos_token_id))

    per_v, t0 = [], time.time()
    for i, (vid, vtext) in enumerate(high, 1):
        user = f"{seeded}\n\n" + INSTRUCTION + "\n\n" + f"CASO:\n{vtext.strip()}"
        pt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     add_generation_prompt=True, tokenize=False)
        pid = tok(pt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        P = int(pid.shape[-1])
        with torch.no_grad():
            seq = hf.generate(pid, generation_config=gcfg)
        total = int(seq.shape[-1])
        with torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(seq)
            acts = {l: rec.activations[l].detach() for l in BAND}

        lo, hi = max(P, SKIP_FIRST_N), total - 1
        row = {"vignette": vid, "prompt_tokens": P, "gen_tokens": total - P, "by_layer": {}}
        for l in BAND:
            h = acts[l][0].float()[lo:hi].cpu()                 # [n_pos, d_model]
            v = vhat[l]
            comp = h @ v                                        # signed scalar per position
            nrm = h.norm(dim=-1)
            frac = (comp.abs() / nrm)
            retained_full = (1.0 - (comp / nrm) ** 2).clamp(min=0).sqrt()
            row["by_layer"][str(l)] = {
                "mean_abs_component": float(comp.abs().mean()),
                "mean_frac_of_norm": float(frac.mean()),
                "max_frac_of_norm": float(frac.max()),
                "mean_norm_retained_full": float(retained_full.mean()),
                "mean_signed_component": float(comp.mean()),
            }
        per_v.append(row)
        print(f"[{i:>2}/20] {vid}  L17 frac={row['by_layer']['17']['mean_frac_of_norm']:.4f}  "
              f"L21={row['by_layer']['21']['mean_frac_of_norm']:.4f}  "
              f"L26={row['by_layer']['26']['mean_frac_of_norm']:.4f}", flush=True)

    agg = {}
    for l in BAND:
        fr = [v["by_layer"][str(l)]["mean_frac_of_norm"] for v in per_v]
        sg = [v["by_layer"][str(l)]["mean_signed_component"] for v in per_v]
        ab = [v["by_layer"][str(l)]["mean_abs_component"] for v in per_v]
        agg[str(l)] = {
            "mean_frac_of_norm": sum(fr) / len(fr),
            "sd_frac_between_vignettes": (sum((x - sum(fr) / len(fr)) ** 2
                                              for x in fr) / (len(fr) - 1)) ** 0.5,
            "mean_abs_component": sum(ab) / len(ab),
            "mean_signed_component": sum(sg) / len(sg),
            "mean_pct_norm_removed_full": 100.0 * (1.0 - (1.0 - (sum(fr) / len(fr)) ** 2) ** 0.5),
        }

    elapsed = time.time() - t0
    out = {"stage": "Phase 2b / Stage J0 projection feasibility",
           "condition_free": True,
           "note": "no conditions varied, no token sets scored, no loadings, "
                   "no counting, no judge",
           "construction": "u_gain target, Tikhonov lambda=0.1, band 17-26 "
                           "(carried forward unchanged from Phase 2)",
           "landing": landing, "model_digest": f"Qwen/Qwen2.5-7B-Instruct@"
           f"{M.get('model_revision', 'a09a3545')}", "lens_pt_sha256": pt_sha,
           "decoding": "greedy", "n_vignettes": len(per_v),
           "aggregate": agg, "per_vignette": per_v,
           "elapsed_seconds": round(elapsed, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nfraction of residual norm lying along v_hat (generation positions):")
    for l in BAND:
        a = agg[str(l)]
        print(f"  L{l:>2}  |h.v|/||h|| = {a['mean_frac_of_norm']:.5f} "
              f"(sd {a['sd_frac_between_vignettes']:.5f})  "
              f"-> full ablation removes {a['mean_pct_norm_removed_full']:.4f}% of the norm")
    print(f"\nelapsed {elapsed / 60:.1f} min -> wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
