#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2 / Stage I1 — mechanical verification of the injecting hook.

Gates the calibration pilot. A silently no-op hook, or one that writes to the
wrong positions, would invalidate every arm, so this asserts the mechanism on
the real model before any experimental run:

  V1  prompt positions are BIT-IDENTICAL to the un-injected pass
  V2  at layer 17 (whose input is unaffected, all earlier layers being untouched)
      the injected residual at generation positions == base + delta_17 exactly
  V3  deeper layers differ by MORE than their own delta (the perturbation
      propagates -- i.e. the model really computed something different)
  V4  the recorder sees POST-addition values (hook registration order)
  V5  mode='generate' and mode='full' add at the same positions, [P, total-1)
  V6  the Set F readout rises monotonically with k at the added positions

CONDITION-FREE: one `neutral` vignette, no token-set scoring is recorded as
data, no judging, no counting toward any arm. Output is a verification report.

Run as a file:  .venv/Scripts/python.exe phase2/scripts/verify_injection.py
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
sys.path.insert(0, str(REPO / "vendor" / "jacobian-lens"))
sys.path.insert(0, str(REPO / "phase2" / "scripts"))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from intervene import (  # noqa: E402
    BAND, K_LADDER, LENS_PT_SHA, Injector, build_rand, build_vhat,
    deltas_for, f_loading, f_survivor_ids, load_rho,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANON = REPO / "phase1" / "materials_canonical"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]
OUT = REPO / "phase2" / "data" / "verify_injection.json"

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

    body = d["payload"].strip() + " " + d.get("disclosure", "").strip()
    seeded = wrapper.replace("{{NAME}}", d["name"]).replace("{{PAYLOAD}}", body)
    vid, vtext = neutral[0]
    user = f"{seeded}\n\n" + INSTRUCTION + "\n\n" + f"CASO:\n{vtext.strip()}"

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)

    cfg = json.loads((Path(MODEL_DIR) / "config.json").read_text(encoding="utf-8"))
    key = "model.embed_tokens.weight" if cfg.get("tie_word_embeddings") else "lm_head.weight"
    W_U = hf.get_output_embeddings().weight.detach().float().cpu() \
        if key == "lm_head.weight" else hf.get_input_embeddings().weight.detach().float().cpu()
    gain = hf.model.norm.weight.detach().float().cpu()

    ids = f_survivor_ids()
    rho = load_rho()
    vhat, landing = build_vhat(lens, W_U, gain, ids)
    rand = build_rand(lens.d_model)
    print(f"landing: band_min_cos={landing['band_min_cos']:.4f} (frozen lambda=0.1, u_gain)")

    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, return_tensors="pt",
                     add_special_tokens=False).input_ids.to("cuda")
    P = int(prompt_ids.shape[-1])

    # A short greedy continuation gives us a fixed token sequence to teacher-force.
    gcfg = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
    gcfg.update(do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=40, pad_token_id=(tok.pad_token_id or tok.eos_token_id))
    with torch.no_grad():
        seq = hf.generate(prompt_ids, generation_config=gcfg)
    total = int(seq.shape[-1])
    print(f"vignette {vid}  P={P}  total={total}  G={total - P}")

    def full_pass(deltas):
        inj = Injector(model.layers, deltas, mode="full", prompt_len=P,
                       end=total - 1) if deltas else None
        if inj:
            with inj, torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
                model.forward(seq)
                return {l: rec.activations[l].detach().float().cpu() for l in BAND}, inj
        with torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(seq)
            return {l: rec.activations[l].detach().float().cpu() for l in BAND}, None

    base_acts, _ = full_pass({})
    k_test = K_LADDER[3]                                   # 0.4, mid ladder
    deltas = deltas_for("F", k_test, K_LADDER[-1], vhat, rand, rho)
    inj_acts, inj = full_pass(deltas)

    checks = {}

    # V1 -- prompt positions untouched
    checks["V1_prompt_bit_identical"] = all(
        torch.equal(base_acts[l][0][:P], inj_acts[l][0][:P]) for l in BAND)

    # V2 -- L17 generation positions == base + delta_17 (input to L17 is unchanged)
    l0 = BAND[0]
    # window is [P, total-1): the last generated position is never injected
    # during real generation, so it is excluded here too (see Injector docstring).
    exp = base_acts[l0][0][P:total - 1] + deltas[l0].float()
    got = inj_acts[l0][0][P:total - 1]
    checks["V2_L17_exact_max_abs_err"] = float((got - exp).abs().max())
    checks["V2_L17_delta_norm"] = float(deltas[l0].norm())
    checks["V2_pass"] = checks["V2_L17_exact_max_abs_err"] < 0.5   # fp16 residual tolerance

    # V3 -- deeper layers moved by more than their own delta (perturbation propagated)
    prop = {}
    for l in BAND[1:]:
        diff = float((inj_acts[l][0][P:total - 1]
                      - base_acts[l][0][P:total - 1]).norm(dim=-1).mean())
        prop[str(l)] = {"mean_shift": diff, "own_delta_norm": float(deltas[l].norm()),
                        "ratio": diff / float(deltas[l].norm())}
    checks["V3_propagation"] = prop
    checks["V3_pass"] = all(v["ratio"] > 1.01 for v in prop.values())

    # V4 -- the recorder saw POST-addition values (implied by V2 being nonzero-shifted)
    checks["V4_recorder_sees_post_addition"] = bool(
        not torch.equal(base_acts[l0][0][P:total - 1], inj_acts[l0][0][P:total - 1]))

    # V5 -- 'generate' mode touches the same number of positions
    inj_gen = Injector(model.layers, deltas, mode="generate")
    with inj_gen, torch.no_grad():
        hf.generate(prompt_ids, generation_config=gcfg)
    checks["V5_added_generate"] = inj_gen.added[l0]
    checks["V5_added_full"] = inj.added[l0]
    checks["V5_expected"] = total - 1 - P
    checks["V5_pass"] = inj_gen.added[l0] == inj.added[l0] == (total - 1 - P)

    # V6 -- Set F readout rises monotonically with k at the added positions
    lo, hi = max(P, 16), total - 1
    curve = {"0.0": f_loading(model, lens, base_acts, ids, lo, hi)}
    for k in K_LADDER:
        acts_k, _ = full_pass(deltas_for("F", k, K_LADDER[-1], vhat, rand, rho))
        curve[str(k)] = f_loading(model, lens, acts_k, ids, lo, hi)
        print(f"  k={k:<5} F_readout={curve[str(k)]:.4f}")
    seq_vals = [curve["0.0"]] + [curve[str(k)] for k in K_LADDER]
    checks["V6_curve"] = curve
    checks["V6_monotone_nondecreasing"] = all(
        b >= a - 1e-9 for a, b in zip(seq_vals, seq_vals[1:]))

    ok = (checks["V1_prompt_bit_identical"] and checks["V2_pass"] and checks["V3_pass"]
          and checks["V4_recorder_sees_post_addition"] and checks["V5_pass"])

    out = {"stage": "Phase 2 / I1 injecting-hook verification",
           "prereg_tag": "prereg-phase2-v1", "prereg_commit": "72bf075",
           "condition_free": True,
           "note": "one neutral vignette; no arm data, no judging, no counting",
           "vignette": vid, "prompt_tokens": P, "total_positions": total,
           "k_tested_for_V1_V5": k_test, "landing": landing,
           "checks": checks, "ALL_PASS": bool(ok)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n--- injecting-hook verification ---")
    print(f"V1 prompt bit-identical      : {checks['V1_prompt_bit_identical']}")
    print(f"V2 L17 == base+delta (maxerr): {checks['V2_L17_exact_max_abs_err']:.4g} "
          f"(delta norm {checks['V2_L17_delta_norm']:.1f}) -> {checks['V2_pass']}")
    print(f"V3 propagation ratios >1     : {checks['V3_pass']}")
    print(f"V4 recorder post-addition    : {checks['V4_recorder_sees_post_addition']}")
    print(f"V5 generate==full=={total - 1 - P} pos : {checks['V5_pass']}")
    print(f"V6 readout monotone in k     : {checks['V6_monotone_nondecreasing']}")
    print(f"\nALL_PASS = {ok}  -> wrote {OUT.relative_to(REPO)}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
