# Copyright 2026 — Phase 0 delegate. Stage 0.3 feasibility pilot (GPU).
"""Feasibility pilot on vignette v12. Runs the model on the vignette + the exact
Stage 0.3 instruction, GENERATES a response, and extracts J-lens readouts across
all fitted layers and ALL positions of the full (prompt+generation) sequence.
Dumps raw readouts (top-k tokens + weights) marking absolute position and
prompt-vs-generation (and the first-16 unfitted/OOD region per issue #5 Pitfall 2).

FEASIBILITY ONLY. No interpretation, no diagnostic-token counting, no comparisons.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]

VIG_PATH = REPO / "phase0" / "data" / "stimuli_pilot" / "v12.txt"
VIGNETTE = VIG_PATH.read_text(encoding="utf-8")
INSTRUCTION = "Read the following case and reason about what might be going on with this person."

TOPK = 10
MAX_NEW = 200
SKIP_FIRST_N = 16  # unfitted/OOD region in apply() per issue #5 Pitfall 2
SEED = 0

OUT_DIR = REPO / "phase0" / "data" / "pilot_readouts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    torch.manual_seed(SEED)
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16
    ).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    print("lens:", repr(lens), flush=True)
    torch.cuda.reset_peak_memory_stats()

    # ---- build prompt (chat) + generate ----
    messages = [{"role": "user", "content": INSTRUCTION + "\n\n" + VIGNETTE}]
    prompt_text = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tok(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to("cuda")
    P = int(prompt_ids.shape[-1])

    t0 = time.time()
    with torch.no_grad():
        gen = hf.generate(
            prompt_ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=(tok.pad_token_id or tok.eos_token_id),
        )
    t_gen = time.time() - t0
    full_ids = gen  # [1, P+G]
    total = int(full_ids.shape[-1])
    G = total - P
    gen_text = tok.decode(full_ids[0, P:], skip_special_tokens=True)
    print(f"prompt_tokens={P} gen_tokens={G} total={total} gen_time={t_gen:.1f}s", flush=True)

    # ---- readouts over the full sequence (prompt + generation) ----
    source_layers = lens.source_layers
    final_layer = model.n_layers - 1
    record_at = sorted(set(source_layers) | {final_layer})

    t1 = time.time()
    with torch.no_grad(), ActivationRecorder(model.layers, at=record_at) as rec:
        model.forward(full_ids)
        acts = {i: rec.activations[i].detach() for i in record_at}

    def rows_for(layer: int, logits_cpu: torch.Tensor, kind: str) -> list:
        vals, idx = logits_cpu.topk(TOPK, dim=-1)
        rows = []
        for pos in range(logits_cpu.shape[0]):
            rows.append({
                "kind": kind,           # "lens" or "model_output"
                "layer": layer,
                "position": pos,        # absolute position in full sequence
                "segment": "prompt" if pos < P else "generation",
                "ood_unfitted_pos": pos < SKIP_FIRST_N,
                "topk": [
                    {"id": int(idx[pos, k]),
                     "token": tok.decode([int(idx[pos, k])]),
                     "weight": round(float(vals[pos, k]), 4)}
                    for k in range(TOPK)
                ],
            })
        return rows

    all_rows = []
    for layer in source_layers:
        resid = acts[layer][0].float()               # [seq, d_model]
        transported = lens.transport(resid, layer)    # J_l @ h
        logits = model.unembed(transported).float().cpu()
        all_rows.extend(rows_for(layer, logits, "lens"))
    # model's actual final-layer output row (reference; L = n_layers-1)
    model_logits = model.unembed(acts[final_layer][0].float()).float().cpu()
    all_rows.extend(rows_for(final_layer, model_logits, "model_output"))
    t_read = time.time() - t1

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # ---- write dumps ----
    readouts_path = OUT_DIR / "v12_readouts.json"
    payload = {
        "schema": {
            "description": "J-lens raw readouts for pilot vignette v12. One row per "
                           "(kind, layer, position). kind='lens' = Jacobian-lens "
                           "readout at that source layer; kind='model_output' = the "
                           "model's actual final-layer logits (L=n_layers-1). "
                           "position is absolute in the full prompt+generation "
                           "sequence; segment marks prompt vs generation; "
                           "ood_unfitted_pos flags positions < 16 (unfitted region, "
                           "issue #5 Pitfall 2). weight = raw lens/model logit.",
            "topk": TOPK,
        },
        "meta": {
            "vignette_id": "v12",
            "vignette_sha256": hashlib.sha256(VIGNETTE.encode("utf-8")).hexdigest(),
            "instruction": INSTRUCTION,
            "model": "Qwen/Qwen2.5-7B-Instruct@a09a3545",
            "lens": "neuronpedia/jacobian-lens@16a01f3 qwen2.5-7b-it",
            "lens_repr": repr(lens),
            "source_layers": source_layers,
            "final_layer": final_layer,
            "prompt_tokens": P,
            "gen_tokens": G,
            "total_positions": total,
            "seed": SEED,
            "decoding": "greedy (do_sample=False), deterministic",
            "max_new_tokens": MAX_NEW,
            "gen_seconds": round(t_gen, 2),
            "readout_seconds": round(t_read, 2),
            "vram_peak_gb": round(peak_gb, 2),
            "n_rows": len(all_rows),
        },
        "generation_text": gen_text,
        "rows": all_rows,
    }
    readouts_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_mb = readouts_path.stat().st_size / 1e6

    # small meta sidecar for the report
    meta_path = OUT_DIR / "v12_meta.json"
    meta = dict(payload["meta"]); meta["readouts_file"] = str(readouts_path)
    meta["readouts_mb"] = round(size_mb, 2)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nrows={len(all_rows)} readouts={size_mb:.2f} MB read_time={t_read:.1f}s "
          f"vram_peak={peak_gb:.2f}GB", flush=True)
    print(f"generation ({G} tok):\n{gen_text[:600]}", flush=True)
    print(f"\n-> {readouts_path}\n-> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
