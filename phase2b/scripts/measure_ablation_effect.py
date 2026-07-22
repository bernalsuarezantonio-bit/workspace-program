#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b / Stage J0-b — what full projection ablation actually does.

Condition-free feasibility, taken BEFORE the 2b design is frozen (the direct
correction of what closed Phase 2, where a frozen parameter proved mis-scaled
and the pilot only found out after the tag).

Per `high` vignette, in the B0_none cell construction (DN_flagged x L1_forum),
greedy decoding:

  1. generate WITHOUT ablation           -> seq_base, text_base
  2. generate WITH full ablation         -> seq_abl,  text_abl
     (divergence point of the two token streams recorded)
  3. teacher-force seq_base twice, with and without ablation, and read the
     Set F loading over [P, total-1) -> the readout reduction, isolated from
     any change in what was generated

Also reports the lesson-#6 degradation terms (set-vocabulary share, lexical
entropy) and the s7 malformed flag under ablation.

NO conditions varied, NO judge, NO counting toward any arm. The Set F readout
here uses the INSTRUCT lens and is therefore the circular estimator that
correction (b) warns about -- it is feasibility evidence, never a landing
criterion. See PREREG_PHASE2B s6.

Run as a file:  .venv/Scripts/python.exe phase2b/scripts/measure_ablation_effect.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
import transformers
import yaml

REPO = Path(__file__).resolve().parents[2]
for p in ("vendor/jacobian-lens", "phase2/scripts", "phase2b/scripts"):
    sys.path.insert(0, str(REPO / p))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from intervene import BAND, LENS_PT_SHA, build_vhat, f_loading, f_survivor_ids, is_malformed  # noqa: E402
from ablate import Projector, degradation_report  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANON = REPO / "phase1" / "materials_canonical"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]
OUT = REPO / "phase2b" / "data" / "ablation_effect.json"

SKIP_FIRST_N, MAX_NEW = 16, 200
MENTION_RX = re.compile(r"inventad|estudio|no reconocid|fictici", re.I)   # registered

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
    seeded = wrapper.replace("{{NAME}}", d["name"]).replace(
        "{{PAYLOAD}}", d["payload"].strip() + " " + d.get("disclosure", "").strip())

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    W_U = hf.get_output_embeddings().weight.detach().float().cpu()
    gain = hf.model.norm.weight.detach().float().cpu()
    ids = f_survivor_ids()
    vhat, landing = build_vhat(lens, W_U, gain, ids)
    print(f"landing band_min_cos={landing['band_min_cos']:.4f}", flush=True)

    gcfg = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
    gcfg.update(do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=MAX_NEW, pad_token_id=(tok.pad_token_id or tok.eos_token_id))

    rows, t0 = [], time.time()
    for vid, vtext in high:
        user = f"{seeded}\n\n" + INSTRUCTION + "\n\n" + f"CASO:\n{vtext.strip()}"
        pt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     add_generation_prompt=True, tokenize=False)
        pid = tok(pt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        P = int(pid.shape[-1])

        with torch.no_grad():
            sb = hf.generate(pid, generation_config=gcfg)
        with Projector(model.layers, vhat, scale=1.0, mode="generate"), torch.no_grad():
            sa = hf.generate(pid, generation_config=gcfg)

        ib, ia = sb[0, P:].tolist(), sa[0, P:].tolist()
        tb = tok.decode(sb[0, P:], skip_special_tokens=True)
        ta = tok.decode(sa[0, P:], skip_special_tokens=True)
        div = next((i for i, (x, y) in enumerate(zip(ib, ia)) if x != y),
                   min(len(ib), len(ia)) if len(ib) != len(ia) else None)

        total = int(sb.shape[-1])
        hi = total - 1
        with torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(sb)
            a0 = {l: rec.activations[l].detach() for l in BAND}
        f0 = f_loading(model, lens, a0, ids, max(P, SKIP_FIRST_N), hi)
        with Projector(model.layers, vhat, scale=1.0, mode="full", prompt_len=P, end=hi), \
                torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(sb)
            a1 = {l: rec.activations[l].detach() for l in BAND}
        f1 = f_loading(model, lens, a1, ids, max(P, SKIP_FIRST_N), hi)

        rows.append({
            "vignette": vid, "prompt_tokens": P, "gen_tokens": total - P,
            "identical": ib == ia, "divergence_at": div,
            "F_base": f0, "F_ablated": f1,
            "mention_base": bool(MENTION_RX.search(tb)),
            "mention_ablated": bool(MENTION_RX.search(ta)),
            "deg_base": degradation_report(ib, ids),
            "deg_ablated": degradation_report(ia, ids),
            "malformed_ablated": is_malformed(ta, ia)[0],
            "gen_text_sha256_base": sh(tb.encode("utf-8")),
            "gen_text_sha256_ablated": sh(ta.encode("utf-8")),
        })
        print(f"{vid} ident={ib == ia} div@{div} F {f0:.4f}->{f1:.4f} "
              f"mention {rows[-1]['mention_base']}->{rows[-1]['mention_ablated']}", flush=True)

    n = len(rows)
    fb = sum(r["F_base"] for r in rows) / n
    fa = sum(r["F_ablated"] for r in rows) / n
    summary = {
        "n_vignettes": n,
        "identical_generations": sum(r["identical"] for r in rows),
        "mean_F_base": fb, "mean_F_ablated": fa,
        "F_reduction_pct": 100.0 * (1.0 - fa / fb),
        "mention_base": sum(r["mention_base"] for r in rows),
        "mention_ablated": sum(r["mention_ablated"] for r in rows),
        "malformed_ablated": sum(r["malformed_ablated"] for r in rows),
        "mean_set_vocab_share_base": sum(r["deg_base"]["set_vocab_share"] for r in rows) / n,
        "mean_set_vocab_share_ablated": sum(r["deg_ablated"]["set_vocab_share"] for r in rows) / n,
        "mean_lexical_entropy_base": sum(r["deg_base"]["lexical_entropy"] for r in rows) / n,
        "mean_lexical_entropy_ablated": sum(r["deg_ablated"]["lexical_entropy"] for r in rows) / n,
    }
    out = {"stage": "Phase 2b / Stage J0-b ablation effect", "condition_free": True,
           "note": "no conditions varied, no judge, no counting; the Set F readout here "
                   "uses the INSTRUCT lens and is the circular estimator (correction b) -- "
                   "feasibility evidence only, never a landing criterion",
           "construction": "u_gain target, Tikhonov lambda=0.1, band 17-26, scale=1.0",
           "landing": landing, "decoding": "greedy",
           "model_digest": f"Qwen/Qwen2.5-7B-Instruct@{M.get('model_revision', 'a09a3545')}",
           "lens_pt_sha256": pt_sha, "mention_regex": MENTION_RX.pattern,
           "summary": summary, "per_vignette": rows,
           "elapsed_seconds": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {n} vignettes, {(time.time() - t0) / 60:.1f} min ===")
    print(f"greedy text identical with vs without ablation: {summary['identical_generations']}/{n}")
    print(f"F readout {fb:.4f} -> {fa:.4f}  ({summary['F_reduction_pct']:.1f}% reduction)")
    print(f"ES mention {summary['mention_base']} -> {summary['mention_ablated']} of {n}")
    print(f"malformed under ablation: {summary['malformed_ablated']}")
    print(f"set_vocab_share {summary['mean_set_vocab_share_base']:.5f} -> "
          f"{summary['mean_set_vocab_share_ablated']:.5f}")
    print(f"lexical_entropy {summary['mean_lexical_entropy_base']:.3f} -> "
          f"{summary['mean_lexical_entropy_ablated']:.3f}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
