# Copyright 2026 — Phase 0 delegate. Nightly TECHNICAL calibration run (GPU).
"""Pipeline technical-variance run — NO experimental conditions, NO token sets,
NO counting, NO comparison. Runs vignette v12 with the SAME pilot instruction,
20 repetitions (seeds 1-20) at temperature 0.7 (the Phase-1 sampling temp),
dumping readouts exactly like the Stage 0.3 pilot (all layers x all positions,
prompt/generation marked). Purpose: between-rep technical stability + validating
the storage format at scale. Incremental per-rep saves; graceful pause + timestamp
if VRAM contention appears (never compete).

This script does NOT read, score, count, or compare readout CONTENT. It records
only volume, runtime, and structural facts.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
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
TEMPERATURE = 0.7
SKIP_FIRST_N = 16
SEEDS = list(range(1, 21))
FREE_VRAM_PAUSE_BYTES = 3 * 1024**3  # pause if device free < 3 GB (contention guard)

OUT_DIR = REPO / "phase0" / "data" / "pilot_readouts" / "nightly"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = REPO / "phase0" / "data" / "pilot_readouts" / "nightly_summary.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_rows(tok, model, lens, full_ids, P):
    source_layers = lens.source_layers
    final_layer = model.n_layers - 1
    record_at = sorted(set(source_layers) | {final_layer})
    with torch.no_grad(), ActivationRecorder(model.layers, at=record_at) as rec:
        model.forward(full_ids)
        acts = {i: rec.activations[i].detach() for i in record_at}

    def rows_for(layer, logits_cpu, kind):
        vals, idx = logits_cpu.topk(TOPK, dim=-1)
        rows = []
        for pos in range(logits_cpu.shape[0]):
            rows.append({
                "kind": kind, "layer": layer, "position": pos,
                "segment": "prompt" if pos < P else "generation",
                "ood_unfitted_pos": pos < SKIP_FIRST_N,
                "topk": [{"id": int(idx[pos, k]), "token": tok.decode([int(idx[pos, k])]),
                          "weight": round(float(vals[pos, k]), 4)} for k in range(TOPK)],
            })
        return rows

    rows = []
    for layer in source_layers:
        resid = acts[layer][0].float()
        logits = model.unembed(lens.transport(resid, layer)).float().cpu()
        rows.extend(rows_for(layer, logits, "lens"))
    model_logits = model.unembed(acts[final_layer][0].float()).float().cpu()
    rows.extend(rows_for(final_layer, model_logits, "model_output"))
    return rows


def main() -> int:
    print(f"[{now_iso()}] cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0), flush=True)
    free0, total0 = torch.cuda.mem_get_info()
    print(f"[{now_iso()}] device free at start: {free0/1e9:.2f} GB / {total0/1e9:.2f} GB",
          flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16
    ).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    print(f"[{now_iso()}] loaded; lens {lens!r}", flush=True)

    # fixed prompt (identical every rep; sampling varies via seed only)
    messages = [{"role": "user", "content": INSTRUCTION + "\n\n" + VIGNETTE}]
    prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    P = int(prompt_ids.shape[-1])

    reps, paused = [], None
    for seed in SEEDS:
        free, _ = torch.cuda.mem_get_info()
        stamp = now_iso()
        if free < FREE_VRAM_PAUSE_BYTES:
            paused = {"at_seed": seed, "timestamp": stamp, "device_free_gb": round(free/1e9, 2),
                      "reason": "device free < 3 GB — possible mistral-sim auto-reload; "
                                "pausing at safe point, not competing"}
            print(f"[{stamp}] PAUSE before seed {seed}: free {free/1e9:.2f} GB", flush=True)
            break

        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            gen = hf.generate(prompt_ids, max_new_tokens=MAX_NEW, do_sample=True,
                              temperature=TEMPERATURE,
                              pad_token_id=(tok.pad_token_id or tok.eos_token_id))
        t_gen = time.time() - t0
        total = int(gen.shape[-1]); G = total - P

        t1 = time.time()
        rows = extract_rows(tok, model, lens, gen, P)
        t_read = time.time() - t1
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        rep_path = OUT_DIR / f"v12_rep{seed:02d}.json"
        payload = {
            "vignette_id": "v12", "seed": seed, "temperature": TEMPERATURE,
            "decoding": "sampling (do_sample=True)", "max_new_tokens": MAX_NEW,
            "prompt_tokens": P, "gen_tokens": G, "total_positions": total,
            "topk": TOPK, "n_rows": len(rows),
            "generation_text": tok.decode(gen[0, P:], skip_special_tokens=True),
            "rows": rows,
        }
        rep_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        size_mb = rep_path.stat().st_size / 1e6

        rec = {"seed": seed, "timestamp": stamp, "device_free_gb_before": round(free/1e9, 2),
               "gen_tokens": G, "total_positions": total, "n_rows": len(rows),
               "size_mb": round(size_mb, 3), "gen_seconds": round(t_gen, 2),
               "readout_seconds": round(t_read, 2), "vram_peak_gb": round(peak_gb, 2)}
        reps.append(rec)
        # incremental summary write (survives a pause/crash)
        SUMMARY_PATH.write_text(json.dumps({"reps": reps, "paused": paused}, indent=2), encoding="utf-8")
        print(f"[{stamp}] seed {seed:>2}: gen_tok={G} rows={len(rows)} "
              f"{size_mb:.2f}MB gen={t_gen:.1f}s read={t_read:.1f}s peak={peak_gb:.2f}GB "
              f"free_before={free/1e9:.1f}GB", flush=True)

    # ---- technical stability summary (volume/runtime/structure ONLY) ----
    def stats(key):
        xs = [r[key] for r in reps]
        if not xs:
            return None
        n = len(xs); mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n
        return {"n": n, "min": min(xs), "max": max(xs), "mean": round(mean, 3),
                "std": round(var ** 0.5, 3)}

    summary = {
        "run": "stage03 nightly technical calibration (v12, no conditions/sets/counting)",
        "completed_reps": len(reps), "requested_reps": len(SEEDS),
        "temperature": TEMPERATURE, "paused": paused,
        "prompt_tokens_constant": P,
        "gen_tokens": stats("gen_tokens"),
        "n_rows": stats("n_rows"),
        "size_mb": stats("size_mb"),
        "gen_seconds": stats("gen_seconds"),
        "readout_seconds": stats("readout_seconds"),
        "vram_peak_gb": stats("vram_peak_gb"),
        "total_disk_mb": round(sum(r["size_mb"] for r in reps), 2),
        "reps": reps,
        "finished": now_iso(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[{now_iso()}] DONE reps={len(reps)}/{len(SEEDS)} "
          f"total_disk={summary['total_disk_mb']:.1f}MB paused={bool(paused)}")
    print(f"-> {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
