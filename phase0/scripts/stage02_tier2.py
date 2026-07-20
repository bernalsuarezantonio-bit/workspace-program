# Copyright 2026 — Phase 0 delegate. Stage 0.2 Tier 2 reproduction (GPU).
"""Load Qwen2.5-7B-Instruct + the pre-fitted lens from the local cache, re-affirm
the isfinite guard on-device, and reproduce the repo's documented examples:
  - currency multi-hop  -> expect euro / lira in the lens readout
  - ascii-face          -> expect "nose" at mid layers at the '^' position
Records load time, apply time, and VRAM peak. Prints readouts verbatim; makes
NO interpretation beyond the documented qualitative match. (Tier 2 of Stage 0.2.)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Windows console defaults to cp1252; decoded vocab tokens include non-cp1252
# characters (CJK, symbols). Force UTF-8 so printing readouts can't crash.
# (Reporting only — does not touch the measurement.)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import transformers

import jlens

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((REPO_ROOT / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR = MANIFEST["model_path"]
LENS_PT = MANIFEST["lens_pt"]

TOPK = 8
summary: dict = {}


def show(lens_logits, tok, note):
    out = {}
    for layer in sorted(lens_logits):
        idx = lens_logits[layer][0].topk(TOPK).indices.tolist()
        toks = [tok.decode([t]) for t in idx]
        out[layer] = toks
        print(f"  L{layer:>2}: {toks}")
    print(f"  ({note})")
    return out


def main() -> int:
    print("=== 1. live CUDA probe ===", flush=True)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()

    print("\n=== 2. load model + lens (local cache) ===", flush=True)
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16
    ).cuda()
    hf.eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    t_load = time.time() - t0
    print(f"loaded in {t_load:.1f}s ; lens = {lens!r}")

    print("\n=== 3. on-device isfinite re-affirm ===", flush=True)
    bad = 0
    for layer, J in lens.jacobians.items():
        if not bool(torch.isfinite(J.to("cuda")).all()):
            bad += 1
    ondev = "PASS" if bad == 0 else f"FAIL ({bad} non-finite J)"
    print("on-device isfinite:", ondev)

    print("\n=== 4a. Tier 2 — currency multi-hop (expect euro/lira) ===", flush=True)
    currency = (
        "Fact: The capital of Japan is Tokyo.\n"
        "Fact: The currency used in the country shaped like a boot is"
    )
    t1 = time.time()
    ll, ml, ids = lens.apply(model, currency, positions=[-2])
    t_apply = time.time() - t1
    print(f"prompt: {currency!r}")
    print(f"read position: -2 ; apply {t_apply:.2f}s ; seq_len={ids.shape[-1]}")
    cur_read = show(ll, tok, "lens top-k per fitted layer at pos -2")
    model_top = [tok.decode([t]) for t in ml[0].topk(TOPK).indices.tolist()]
    print(f"  model final logits top-{TOPK} @ -2: {model_top}")

    print("\n=== 4b. Tier 2 — ascii-face (expect 'nose' at mid layers) ===", flush=True)
    from jlens.examples import _ASCII_FACE  # documented example prompt

    face = _ASCII_FACE + "\n\nWhat is this?"
    ll2, ml2, ids2 = lens.apply(model, face, positions=None)
    id_list = ids2[0].tolist()
    decoded = [tok.decode([t]) for t in id_list]
    caret_pos = [i for i, s in enumerate(decoded) if "^" in s]
    print(f"seq_len={len(id_list)} ; '^' token position(s): {caret_pos}")
    nose_hit = None
    if caret_pos:
        p = caret_pos[0]
        print(f"readout at '^' (nose) position {p}:")
        face_read = {}
        for layer in sorted(ll2):
            idx = ll2[layer][p].topk(TOPK).indices.tolist()
            toks = [tok.decode([t]) for t in idx]
            face_read[layer] = toks
            if any("nose" in t.lower() for t in toks) and nose_hit is None:
                nose_hit = layer
            print(f"  L{layer:>2}: {toks}")
        summary["ascii_face_read_at_caret"] = face_read
    print(f"  'nose' first appears at layer: {nose_hit}")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n=== VRAM peak (torch max_memory_allocated): {peak_gb:.2f} GB ===")

    summary.update({
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0),
        "load_seconds": round(t_load, 1),
        "apply_seconds_currency": round(t_apply, 2),
        "vram_peak_gb": round(peak_gb, 2),
        "lens_repr": repr(lens),
        "on_device_isfinite": ondev,
        "currency_prompt": currency,
        "currency_read_pos": -2,
        "currency_lens_topk": cur_read,
        "currency_model_topk": model_top,
        "ascii_caret_positions": caret_pos,
        "ascii_nose_first_layer": nose_hit,
    })
    out = REPO_ROOT / "phase0" / "data" / "stage02_tier2_readouts.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSUMMARY -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
