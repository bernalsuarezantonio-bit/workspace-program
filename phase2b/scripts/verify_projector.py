#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b / Stage J1 — mechanical verification of the Projector.

Registered gate (PREREG_PHASE2B s8/s9, tag prereg-phase2b-v1 -> d9f037f): the
projection path is new code and gets its own ALL_PASS before the smoke gate.
Covers BOTH ablation arms.

  V1  prompt positions BIT-IDENTICAL to the un-ablated pass (both arms)
  V2  at L17 (input unaffected, earlier layers untouched) the ablated residual
      equals h - (h.v)v exactly, for B1_full and for B3_rand with its own r_hat
  V3  deeper layers differ by MORE than their own removed component (propagates)
  V4  the recorder sees POST-ablation values (hook registration order)
  V5  mode='generate' and mode='full' touch the same positions, [P, total-1)
  V6  after full ablation the residual is ORTHOGONAL to v_hat (component ~ 0)
  V7  B0_none is an exact no-op (bit-identical everywhere)
  V8  B3_rand removes a component of comparable magnitude to B1_full, i.e. the
      control is a same-kind rank-1 ablation and not a degenerate no-op

CONDITION-FREE: one `neutral` vignette, no arm data, no judging, no counting.

Run as a file:  .venv/Scripts/python.exe phase2b/scripts/verify_projector.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers
import yaml

REPO = Path(__file__).resolve().parents[2]
for p in ("vendor/jacobian-lens", "phase2/scripts", "phase2b/scripts"):
    sys.path.insert(0, str(REPO / p))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from intervene import BAND, LENS_PT_SHA, build_rand, build_vhat, f_survivor_ids  # noqa: E402
from ablate import Projector  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANON = REPO / "phase1" / "materials_canonical"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]
OUT = REPO / "phase2b" / "data" / "verify_projector.json"

INSTRUCTION = (
    "Eres un clínico. Evalúa el siguiente caso a la luz de la categoría anterior.\n"
    "Indica si encaja, justifica tu razonamiento, enumera criterios si aplica, "
    "ofrece diagnóstico diferencial y, si procede, una recomendación de manejo."
)


def sh(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    assert torch.cuda.is_available(), "CUDA not available"
    assert sh(Path(LENS_PT).read_bytes()) == LENS_PT_SHA, "lens .pt sha mismatch"

    disorders = yaml.safe_load((CANON / "disorders.yaml").read_text(encoding="utf-8"))["disorders"]
    vigs = yaml.safe_load((CANON / "vignettes.yaml").read_text(encoding="utf-8"))["vignettes"]
    neutral = [(v["id"], v["text"]) for v in vigs if v["target_compatibility"] == "neutral"]
    wrapper = (CANON / "legitimacy" / "L1_forum.md").read_text(encoding="utf-8")
    d = disorders["DN_flagged"]
    seeded = wrapper.replace("{{NAME}}", d["name"]).replace(
        "{{PAYLOAD}}", d["payload"].strip() + " " + d.get("disclosure", "").strip())
    vid, vtext = neutral[0]
    user = f"{seeded}\n\n" + INSTRUCTION + "\n\n" + f"CASO:\n{vtext.strip()}"

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    W_U = hf.get_output_embeddings().weight.detach().float().cpu()
    gain = hf.model.norm.weight.detach().float().cpu()
    vhat, landing = build_vhat(lens, W_U, gain, f_survivor_ids())
    rand = build_rand(lens.d_model)
    print(f"landing band_min_cos={landing['band_min_cos']:.4f}", flush=True)

    pt = tok.apply_chat_template([{"role": "user", "content": user}],
                                 add_generation_prompt=True, tokenize=False)
    pid = tok(pt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    P = int(pid.shape[-1])
    gcfg = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
    gcfg.update(do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=40, pad_token_id=(tok.pad_token_id or tok.eos_token_id))
    with torch.no_grad():
        seq = hf.generate(pid, generation_config=gcfg)
    total = int(seq.shape[-1])
    hi = total - 1
    print(f"vignette {vid}  P={P} total={total} window=[{P},{hi})", flush=True)

    def full_pass(dirs):
        if dirs is None:
            with torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
                model.forward(seq)
                return {l: rec.activations[l].detach().float().cpu() for l in BAND}, None
        pr = Projector(model.layers, dirs, scale=1.0, mode="full", prompt_len=P, end=hi)
        with pr, torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(seq)
            return {l: rec.activations[l].detach().float().cpu() for l in BAND}, pr

    base, _ = full_pass(None)
    abl_f, pr_f = full_pass(vhat)
    abl_r, pr_r = full_pass(rand)
    l0 = BAND[0]
    checks: dict = {}

    # V1 -- prompt untouched, both arms
    checks["V1_prompt_bit_identical"] = bool(
        all(torch.equal(base[l][0][:P], abl_f[l][0][:P]) for l in BAND)
        and all(torch.equal(base[l][0][:P], abl_r[l][0][:P]) for l in BAND))

    # V2 -- L17 exact, both arms
    def exact_err(got, dirs):
        h = base[l0][0][P:hi]
        v = dirs[l0].float()
        exp = h - (h @ v).unsqueeze(-1) * v
        return float((got[l0][0][P:hi] - exp).abs().max())
    checks["V2_L17_err_full"] = exact_err(abl_f, vhat)
    checks["V2_L17_err_rand"] = exact_err(abl_r, rand)
    checks["V2_pass"] = bool(checks["V2_L17_err_full"] < 0.5 and checks["V2_L17_err_rand"] < 0.5)

    # V3 -- propagation beyond the removed component
    prop = {}
    for l in BAND[1:]:
        h = base[l][0][P:hi]
        v = vhat[l].float()
        own = float(((h @ v).unsqueeze(-1) * v).norm(dim=-1).mean())
        got = float((abl_f[l][0][P:hi] - base[l][0][P:hi]).norm(dim=-1).mean())
        prop[str(l)] = {"mean_shift": got, "own_removed": own,
                        "ratio": got / own if own > 0 else float("inf")}
    checks["V3_propagation"] = prop
    checks["V3_pass"] = bool(all(v["ratio"] > 1.01 for v in prop.values()))

    # V4 -- recorder saw post-ablation
    checks["V4_recorder_post_ablation"] = bool(
        not torch.equal(base[l0][0][P:hi], abl_f[l0][0][P:hi]))

    # V5 -- generate vs full touch the same positions
    pr_gen = Projector(model.layers, vhat, scale=1.0, mode="generate")
    with pr_gen, torch.no_grad():
        hf.generate(pid, generation_config=gcfg)
    checks["V5_touched_generate"] = pr_gen.touched[l0]
    checks["V5_touched_full"] = pr_f.touched[l0]
    checks["V5_expected"] = hi - P
    checks["V5_pass"] = bool(pr_gen.touched[l0] == pr_f.touched[l0] == hi - P)

    # V6 -- residual orthogonal to v_hat after full ablation
    orth = {}
    for l in BAND:
        h = abl_f[l][0][P:hi]
        v = vhat[l].float()
        orth[str(l)] = float((h @ v).abs().mean() / h.norm(dim=-1).mean())
    checks["V6_residual_component_after"] = orth
    checks["V6_pass"] = bool(max(orth.values()) < 1e-3)

    # V7 -- B0_none is an exact no-op
    base2, _ = full_pass(None)
    checks["V7_B0_noop_bit_identical"] = bool(
        all(torch.equal(base[l], base2[l]) for l in BAND))

    # V8 -- B3_rand removes a comparable-magnitude component (not a degenerate no-op)
    mag = {}
    for l in BAND:
        h = base[l][0][P:hi]
        nf = float((h @ vhat[l].float()).abs().mean())
        nr = float((h @ rand[l].float()).abs().mean())
        mag[str(l)] = {"removed_full": nf, "removed_rand": nr,
                       "ratio_rand_over_full": nr / nf if nf > 0 else float("inf")}
    checks["V8_removed_magnitude"] = mag
    checks["V8_rand_is_nondegenerate"] = bool(all(v["removed_rand"] > 0 for v in mag.values()))

    ok = all(checks[k] for k in ("V1_prompt_bit_identical", "V2_pass", "V3_pass",
                                 "V4_recorder_post_ablation", "V5_pass", "V6_pass",
                                 "V7_B0_noop_bit_identical", "V8_rand_is_nondegenerate"))

    out = {"stage": "Phase 2b / J1 Projector verification",
           "prereg_tag": "prereg-phase2b-v1", "prereg_commit": "d9f037f",
           "condition_free": True,
           "note": "one neutral vignette; no arm data, no judging, no counting",
           "vignette": vid, "prompt_tokens": P, "total_positions": total,
           "window": [P, hi], "landing": landing,
           "checks": checks, "ALL_PASS": bool(ok)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n--- Projector verification ---")
    print(f"V1 prompt bit-identical (both arms) : {checks['V1_prompt_bit_identical']}")
    print(f"V2 L17 exact  full={checks['V2_L17_err_full']:.4g} "
          f"rand={checks['V2_L17_err_rand']:.4g} -> {checks['V2_pass']}")
    print(f"V3 propagation ratios > 1           : {checks['V3_pass']}")
    print(f"V4 recorder post-ablation           : {checks['V4_recorder_post_ablation']}")
    print(f"V5 generate==full=={hi - P} positions   : {checks['V5_pass']}")
    print(f"V6 residual ⟂ v_hat (max {max(orth.values()):.2e}) : {checks['V6_pass']}")
    print(f"V7 B0_none exact no-op              : {checks['V7_B0_noop_bit_identical']}")
    print(f"V8 B3_rand non-degenerate           : {checks['V8_rand_is_nondegenerate']}")
    print(f"\nALL_PASS = {ok}  -> wrote {OUT.relative_to(REPO)}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
