#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2 / Stage I1 — calibration pilot (PREREG_PHASE2 s3.4, tag prereg-phase2-v1).

Held-out: 2 `neutral` vignettes drawn by PILOT_SEED from the 20 neutral, OUTSIDE
the confirmatory set. The 20 `high` vignettes are untouched. Pilot outputs are
used only to set k; they are not analysed as arm data and no judge is run.

Executes the frozen ladder k in {0.05,0.1,0.2,0.4,0.8,1.6}, 5 reps/rung, plus an
alpha=0 reference, measuring per run:
  (i)  the s2 Set F loading restricted to the injected positions [P, total-1)
  (ii) the s7 malformed flag
then applies the s3.4 k-selection rule mechanically and reports its output --
including the case where the rule cannot be satisfied.

Also runs an EXTENDED downward sweep (exploratory, labelled) so that, if the
frozen ladder overshoots the registered 2x/10x/50x targets, the PI has the
numbers to locate them without another GPU round-trip.

Run as a file:  .venv/Scripts/python.exe phase2/scripts/run_pilot.py
"""

from __future__ import annotations

import hashlib
import json
import random
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
from intervene import (  # noqa: E402
    BAND, K_LADDER, LENS_PT_SHA, NATURAL_F, TARGETS, Injector, build_rand,
    build_vhat, deltas_for, f_loading, f_survivor_ids, is_malformed, load_rho,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANON = REPO / "phase1" / "materials_canonical"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]
OUT = REPO / "phase2" / "data" / "pilot_calibration.json"

PILOT_SEED = 20260722
REPS = 5
EXT_LADDER = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]   # exploratory
EXT_REPS = 2
MAX_NEW, TEMPERATURE = 200, 0.7
MALFORMED_MAX = 0.10                 # s3.4 k_max rule

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
    neutral = [(v["id"], v["text"]) for v in vigs if v["target_compatibility"] == "neutral"]
    assert len(neutral) == 20
    picked = random.Random(PILOT_SEED).sample(neutral, 2)
    print("pilot vignettes (neutral, held out):", [v for v, _ in picked], flush=True)

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
    ids = f_survivor_ids()
    rho = load_rho()
    vhat, landing = build_vhat(lens, W_U, gain, ids)
    rand = build_rand(lens.d_model)

    gcfg = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
    gcfg.update(do_sample=True, temperature=TEMPERATURE, max_new_tokens=MAX_NEW,
                pad_token_id=(tok.pad_token_id or tok.eos_token_id))
    resolved = {k: gcfg.to_dict().get(k) for k in
                ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens",
                 "repetition_penalty")}
    print("resolved generation:", resolved, flush=True)

    prompts = {}
    for vid, vtext in picked:
        user = f"{seeded}\n\n" + INSTRUCTION + "\n\n" + f"CASO:\n{vtext.strip()}"
        pt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     add_generation_prompt=True, tokenize=False)
        prompts[vid] = (user, tok(pt, return_tensors="pt",
                                  add_special_tokens=False).input_ids.to("cuda"))

    runs = []
    seed_counter = [0]

    def one_run(vid: str, k: float, rep: int, tagname: str):
        user, prompt_ids = prompts[vid]
        P = int(prompt_ids.shape[-1])
        seed = 900000 + seed_counter[0]
        seed_counter[0] += 1
        torch.manual_seed(seed)

        deltas = deltas_for("base" if k == 0.0 else "F", k, K_LADDER[-1], vhat, rand, rho)
        t0 = time.time()
        if deltas:
            with Injector(model.layers, deltas, mode="generate"), torch.no_grad():
                seq = hf.generate(prompt_ids, generation_config=gcfg)
        else:
            with torch.no_grad():
                seq = hf.generate(prompt_ids, generation_config=gcfg)
        t_gen = time.time() - t0
        total = int(seq.shape[-1])
        gen_ids = seq[0, P:].tolist()
        gen_text = tok.decode(seq[0, P:], skip_special_tokens=True)

        hi = total - 1
        if hi > P:
            inj2 = (Injector(model.layers, deltas, mode="full", prompt_len=P, end=hi)
                    if deltas else None)
            if inj2:
                with inj2, torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(seq)
                    acts = {l: rec.activations[l].detach() for l in BAND}
            else:
                with torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(seq)
                    acts = {l: rec.activations[l].detach() for l in BAND}
            load = f_loading(model, lens, acts, ids, max(P, 16), hi)
        else:
            load = float("nan")

        mal, why = is_malformed(gen_text, gen_ids)
        r = {"tag": tagname, "vignette": vid, "k": k, "rep": rep, "seed": seed,
             "prompt_tokens": P, "gen_tokens": total - P, "injected_positions": max(0, hi - P),
             "f_loading": load, "malformed": mal, "malformed_reason": why,
             "gen_seconds": round(t_gen, 2),
             "gen_text_sha256": sh(gen_text.encode("utf-8"))}
        runs.append(r)
        return r

    t_start = time.time()
    # --- alpha = 0 reference -------------------------------------------------
    for vid, _ in picked:
        for rep in range(1, REPS + 1):
            r = one_run(vid, 0.0, rep, "reference")
            print(f"  [ref ] {vid} rep{rep} F={r['f_loading']:.4f} mal={r['malformed']}",
                  flush=True)

    # --- frozen ladder -------------------------------------------------------
    for k in K_LADDER:
        for vid, _ in picked:
            for rep in range(1, REPS + 1):
                r = one_run(vid, k, rep, "frozen_ladder")
                print(f"  [k={k:<6}] {vid} rep{rep} F={r['f_loading']:9.3f} "
                      f"mal={r['malformed']} {r['malformed_reason']}", flush=True)

    # --- extended downward sweep (EXPLORATORY, labelled) ---------------------
    for k in EXT_LADDER:
        for vid, _ in picked:
            for rep in range(1, EXT_REPS + 1):
                r = one_run(vid, k, rep, "extended_exploratory")
                print(f"  [ext k={k:<8}] {vid} rep{rep} F={r['f_loading']:9.4f} "
                      f"mal={r['malformed']}", flush=True)
    elapsed = time.time() - t_start

    # --- aggregate -----------------------------------------------------------
    def agg(tagname, ks):
        out = {}
        for k in ks:
            sel = [r for r in runs if r["tag"] == tagname and r["k"] == k]
            n = len(sel)
            good = [r["f_loading"] for r in sel if r["f_loading"] == r["f_loading"]]
            out[str(k)] = {
                "n": n,
                "mean_f_loading": sum(good) / len(good) if good else float("nan"),
                "malformed_rate": sum(r["malformed"] for r in sel) / n if n else float("nan"),
            }
        return out

    ref = agg("reference", [0.0])
    frozen = agg("frozen_ladder", K_LADDER)
    ext = agg("extended_exploratory", EXT_LADDER)

    # --- s3.4 k-selection rule, applied mechanically -------------------------
    ok_rungs = [k for k in K_LADDER if frozen[str(k)]["malformed_rate"] < MALFORMED_MAX]
    k_max = max(ok_rungs) if ok_rungs else None

    def rung_hitting(target):
        hit = [k for k in K_LADDER if frozen[str(k)]["mean_f_loading"] >= target]
        return min(hit) if hit else None

    hit50 = rung_hitting(TARGETS["50x"])
    k3 = None if (hit50 is None or k_max is None) else min(hit50, k_max)

    def nearest(target):
        cand = [k for k in K_LADDER if k_max is None or k <= k_max]
        if not cand:
            return None
        return min(cand, key=lambda k: abs(frozen[str(k)]["mean_f_loading"] - target))

    k2, k1 = nearest(TARGETS["10x"]), nearest(TARGETS["2x"])
    distinct = k1 is not None and k2 is not None and k3 is not None and k1 < k2 < k3

    lowest = frozen[str(K_LADDER[0])]["mean_f_loading"]
    rule_satisfiable = lowest <= TARGETS["2x"] * 1.5

    selection = {
        "rule": "k_max = largest rung with malformed rate < 10%; k3 = min(rung hitting "
                "50x, k_max); k2,k1 = rungs nearest 10x and 2x, subject to k1<k2<k3",
        "k_max": k_max, "rung_hitting_50x": hit50,
        "k1": k1, "k2": k2, "k3": k3, "three_distinct_doses": bool(distinct),
        "rule_satisfiable": bool(rule_satisfiable),
        "lowest_frozen_rung_mean_f": lowest,
        "lowest_rung_over_natural": lowest / NATURAL_F,
        "targets": TARGETS, "natural_reference": NATURAL_F,
    }

    out = {
        "stage": "Phase 2 / Stage I1 calibration pilot",
        "prereg_tag": "prereg-phase2-v1", "prereg_commit": "72bf075",
        "held_out_vignettes": [v for v, _ in picked],
        "pilot_seed": PILOT_SEED, "reps_per_rung": REPS,
        "generation": resolved, "lens_pt_sha256": pt_sha,
        "landing": landing, "rho_l": {str(l): rho[l] for l in BAND},
        "reference_alpha0": ref, "frozen_ladder": frozen,
        "extended_exploratory": ext, "extended_reps": EXT_REPS,
        "selection": selection,
        "elapsed_seconds": round(elapsed, 1), "n_runs": len(runs), "runs": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== frozen ladder ===")
    print(f"  alpha=0 reference        mean F = {ref['0.0']['mean_f_loading']:.4f}")
    for k in K_LADDER:
        f = frozen[str(k)]
        print(f"  k={k:<6} mean F = {f['mean_f_loading']:10.3f}  "
              f"({f['mean_f_loading'] / NATURAL_F:9.1f}x natural)  "
              f"malformed = {f['malformed_rate']:.0%}")
    print("\n=== extended (exploratory) ===")
    for k in EXT_LADDER:
        f = ext[str(k)]
        print(f"  k={k:<8} mean F = {f['mean_f_loading']:10.4f}  "
              f"({f['mean_f_loading'] / NATURAL_F:9.1f}x natural)  "
              f"malformed = {f['malformed_rate']:.0%}")
    print(f"\ntargets: 2x={TARGETS['2x']:.4f} 10x={TARGETS['10x']:.4f} 50x={TARGETS['50x']:.4f}")
    print(f"selection: {selection}")
    print(f"\n{len(runs)} runs in {elapsed / 60:.1f} min -> wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
