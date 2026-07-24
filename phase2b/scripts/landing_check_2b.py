#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b analysis — registered landing check (PREREG_PHASE2B s6.1).

s6.1 verbatim: "PRIMARY -- untied input-embedding readout. This checkpoint has
tie_word_embeddings = false (verified at Stage I0), so model.embed_tokens.weight
is a genuinely different matrix from lm_head.weight -- it shares the residual but
not the readout head... Set F loading recomputed per arm on the captured
residuals with this head; reported per arm as the achieved ablation depth."

The residuals were not stored (only top-k lens logits), so each run's sequence is
replayed teacher-forced with its own arm's projection re-applied over the recorded
window [P, total-1), and the Set F loading is recomputed with the embed_tokens
head. The s2 estimator shape is inherited unchanged, including the top-k=10
membership rule, so the head is the only thing that differs.

Alignment guard (Phase 1 precedent, RESULTS_PHASE1 mask note): the sequence is
rebuilt by re-tokenizing the stored generation_text; runs whose re-tokenized
length != the recorded gen_tokens are EXCLUDED and counted.

Also recomputes, on the same pass, the instruct-lens F loading (s6.3, labelled
circular, descriptive only) so the two heads are directly comparable.

Run as a file:  .venv/Scripts/python.exe phase2b/scripts/landing_check_2b.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import transformers

REPO = Path(__file__).resolve().parents[2]
for p in ("vendor/jacobian-lens", "phase2/scripts", "phase2b/scripts"):
    sys.path.insert(0, str(REPO / p))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from intervene import BAND, LENS_PT_SHA, build_rand, build_vhat, f_survivor_ids  # noqa: E402
from ablate import Projector  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = REPO / "phase2b" / "data"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]
OUT = D / "landing_check.json"
TOPK, SKIP_FIRST_N = 10, 16


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@torch.no_grad()
def f_loading_head(resid_by_layer, lens, head_W, gain, f_ids, eps: float):
    """s2 estimator with an arbitrary readout head: mean over band of the mean over
    positions of the summed top-k logits of the Set F operative tokens."""
    id_t = torch.tensor(f_ids, device=head_W.device)
    per_layer = []
    for l in BAND:
        h = resid_by_layer[l]
        x = lens.transport(h, l)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        logits = (x * gain) @ head_W.T
        vals, idx = logits.topk(TOPK, dim=-1)
        hit = (idx.unsqueeze(-1) == id_t.view(1, 1, -1)).any(-1)
        per_layer.append(float((vals * hit).sum(-1).mean()))
    return sum(per_layer) / len(per_layer)


def main() -> int:
    assert torch.cuda.is_available(), "CUDA not available"
    man = jl(D / "run_manifest_full.jsonl")

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    assert lens is not None

    lm_head = hf.get_output_embeddings().weight.detach().float()
    embed = hf.get_input_embeddings().weight.detach().float()
    gain = hf.model.norm.weight.detach().float()
    eps = float(getattr(hf.config, "rms_norm_eps", 1e-6))
    assert not bool(getattr(hf.config, "tie_word_embeddings", False)), \
        "tie_word_embeddings must be false for s6.1 to be semi-independent"
    cos_heads = float(torch.nn.functional.cosine_similarity(
        lm_head.flatten(), embed.flatten(), dim=0))

    f_ids = f_survivor_ids()
    vhat, landing_geom = build_vhat(lens, lm_head.cpu(), gain.cpu(), f_ids)
    rand = build_rand(lens.d_model)
    DIRS = {"B0_none": None, "B1_full": vhat, "B3_rand": rand}
    print(f"lens_pt pinned={LENS_PT_SHA[:12]} band_min_cos={landing_geom['band_min_cos']:.4f} "
          f"cos(lm_head, embed_tokens)={cos_heads:.4f}", flush=True)

    rows, misaligned, t0 = [], [], time.time()
    for i, m in enumerate(man, 1):
        d = json.loads((REPO / m["readout_file"]).read_text(encoding="utf-8"))
        pt = tok.apply_chat_template([{"role": "user", "content": d["user_prompt"]}],
                                     add_generation_prompt=True, tokenize=False)
        pid = tok(pt, return_tensors="pt", add_special_tokens=False).input_ids
        P = int(pid.shape[-1])
        gid = tok(d["generation_text"], return_tensors="pt",
                  add_special_tokens=False).input_ids
        if P != m["prompt_tokens"] or int(gid.shape[-1]) != m["gen_tokens"]:
            misaligned.append({"trial_id": m["trial_id"], "P": P,
                               "P_rec": m["prompt_tokens"],
                               "G": int(gid.shape[-1]), "G_rec": m["gen_tokens"]})
            continue
        seq = torch.cat([pid, gid], dim=-1).cuda()
        total = int(seq.shape[-1])
        hi, lo = total - 1, max(P, SKIP_FIRST_N)
        dirs = DIRS[m["arm"]]
        if dirs is None:
            with ActivationRecorder(model.layers, at=BAND) as rec:
                model.forward(seq)
                acts = {l: rec.activations[l].detach()[0][lo:hi].float() for l in BAND}
        else:
            with Projector(model.layers, dirs, scale=1.0, mode="full",
                           prompt_len=P, end=hi), \
                    ActivationRecorder(model.layers, at=BAND) as rec:
                model.forward(seq)
                acts = {l: rec.activations[l].detach()[0][lo:hi].float() for l in BAND}
        rows.append({
            "trial_id": m["trial_id"], "arm": m["arm"], "vignette": m["vignette"],
            "rep": m["rep"], "n_positions": hi - lo,
            "F_embed": f_loading_head(acts, lens, embed, gain, f_ids, eps),
            "F_lens": f_loading_head(acts, lens, lm_head, gain, f_ids, eps),
        })
        if i % 50 == 0:
            el = (time.time() - t0) / 60
            print(f"  {i}/{len(man)} elapsed={el:.1f}m eta={el / i * (len(man) - i):.1f}m",
                  flush=True)

    out = {"stage": "Phase 2b landing check (PREREG s6.1)",
           "data_commit": "317ddb9", "prereg_tag": "prereg-phase2b-v1",
           "head_primary": "model.embed_tokens.weight (untied)",
           "head_descriptive_circular": "lm_head.weight (instruct lens)",
           "cos_lm_head_vs_embed_tokens": cos_heads,
           "landing_geometry": landing_geom,
           "estimator": "s2 estimator, top-k=10 membership, band 17-26, "
                        "readout window [max(P,16), total-1)",
           "n_runs": len(rows), "n_misaligned_excluded": len(misaligned),
           "misaligned": misaligned,
           "elapsed_minutes": round((time.time() - t0) / 60, 1),
           "per_run": rows}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nn={len(rows)} misaligned_excluded={len(misaligned)} -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
