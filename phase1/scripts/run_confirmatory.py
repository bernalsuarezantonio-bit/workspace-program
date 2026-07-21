# Copyright 2026 — Phase 1 delegate. Stage P1: confirmatory J-lens runs (GPU).
"""Generate the 4 confirmatory cells × 20 high vignettes × R reps and dump J-lens
readouts (top-k=10, all fitted layers × all positions, prompt/gen marked, first-16
OOD flagged) — exactly the pilot's readout machinery, now under conditions.

Prompt assembly is byte-faithful to reification-gradient src/run_experiment.py @ ee23c07
(wrap()/build_prompt()); materials from the tracked byte-exact phase1/materials_canonical/.
Generation: temperature 0.7, do_sample=True, max_new_tokens=200 (prereg §Generation); the
full resolved GenerationConfig is recorded per run. Per-run seed recorded; execution order
randomized with a recorded master seed. Resumable: completed trials (present + manifest line)
are skipped. Raw readouts go to gitignored phase1/data/readouts/; a lightweight per-run line
is appended to phase1/data/run_manifest.jsonl (committed).

NO analysis here: no loadings, no aggregation, no counting. Generation + readout dump only.

Usage:
  python run_confirmatory.py --smoke      # 8 runs (4 cells × v01,v02 × rep1) -> smoke dir
  python run_confirmatory.py --full       # 4 × 20 × R reps
"""
from __future__ import annotations

import argparse, hashlib, json, random, sys, time
from pathlib import Path

import torch
import transformers
import yaml

import jlens
from jlens.hooks import ActivationRecorder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
CANON = REPO / "phase1" / "materials_canonical"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]
LENS_PT_SHA = "3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29"  # PROVENANCE 0.2

TOPK, MAX_NEW, SKIP_FIRST_N = 10, 200, 16
TEMPERATURE = 0.7
MASTER_SEED = 20260721   # execution-order randomization (recorded)
SEED_BASE = 700000       # per-run generation seed = SEED_BASE + canonical_trial_index (recorded)
REPS_FULL = 10           # prereg §4 R=10

INSTRUCTION = (
    "Eres un clínico. Evalúa el siguiente caso a la luz de la categoría anterior.\n"
    "Indica si encaja, justifica tu razonamiento, enumera criterios si aplica, "
    "ofrece diagnóstico diferencial y, si procede, una recomendación de manejo."
)

# 4 confirmatory cells: (cell_id, disorder_key, wrapper_stem)
CELLS = [
    ("C1_DN_flagged_L1",   "DN_flagged",   "L1_forum"),
    ("C1_DN_plausible_L1", "DN_plausible", "L1_forum"),
    ("C2_incoherent_L4",   "incoherent",   "L4_preprint"),
    ("C2_incoherent_L1",   "incoherent",   "L1_forum"),
]


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
    wrappers = {
        "L1_forum": (CANON / "legitimacy" / "L1_forum.md").read_text(encoding="utf-8"),
        "L4_preprint": (CANON / "legitimacy" / "L4_preprint.md").read_text(encoding="utf-8"),
    }
    return disorders, high, wrappers


def build_trials(reps, vignette_ids=None):
    """Canonical enumeration -> per-trial recorded seed; then shuffle by MASTER_SEED."""
    disorders, high, _ = load_materials()
    if vignette_ids is not None:
        high = [(vid, txt) for vid, txt in high if vid in vignette_ids]
    trials = []
    idx = 0
    for cell_id, dkey, wstem in CELLS:
        for vid, _vtxt in high:
            for rep in range(1, reps + 1):
                trials.append({
                    "trial_id": f"{cell_id}__{vid}__rep{rep:02d}",
                    "cell": cell_id, "disorder": dkey, "wrapper": wstem,
                    "vignette": vid, "rep": rep,
                    "run_seed": SEED_BASE + idx, "canonical_index": idx,
                })
                idx += 1
    random.Random(MASTER_SEED).shuffle(trials)
    return trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.full):
        sys.exit("choose --smoke or --full")

    reps = 1 if args.smoke else REPS_FULL
    vig_filter = {"v01", "v02"} if args.smoke else None
    tag = "smoke" if args.smoke else "full"
    OUT = REPO / "phase1" / "data" / ("readouts_smoke" if args.smoke else "readouts")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO / "phase1" / "data" / (f"run_manifest_{tag}.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    disorders, high, wrappers = load_materials()
    trials = build_trials(reps, vig_filter)
    print(f"[{tag}] trials={len(trials)} (cells={len(CELLS)} vignettes={len(high)} reps={reps}) "
          f"master_seed={MASTER_SEED}", flush=True)

    # resume: skip trials already recorded AND whose readout file exists
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if (OUT / f"{r['trial_id']}.json").exists():
                    done.add(r["trial_id"])
    todo = [t for t in trials if t["trial_id"] not in done]
    print(f"[{tag}] already done={len(done)}  todo={len(todo)}", flush=True)
    if not todo:
        print(f"[{tag}] nothing to do (resume complete)", flush=True); return 0

    # ---- load model + lens; verify lens .pt digest (prereg §6 exclusion c) ----
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA not available"
    print("cuda:", torch.cuda.get_device_name(0), flush=True)
    pt_sha = sh(Path(LENS_PT).read_bytes())
    assert pt_sha == LENS_PT_SHA, f"lens .pt sha256 {pt_sha[:12]} != pinned {LENS_PT_SHA[:12]}"
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    # lens re-verification (prereg §6 exclusion c): .pt sha256 == the Stage 0.2 checkpoint,
    # which passed the issue-#6 isfinite guard on load — pin asserted above, no re-scan needed.
    gen_cfg = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
    gen_cfg.update(do_sample=True, temperature=TEMPERATURE, max_new_tokens=MAX_NEW,
                   pad_token_id=(tok.pad_token_id or tok.eos_token_id))
    resolved_gen = {k: gen_cfg.to_dict().get(k) for k in
                    ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens",
                     "repetition_penalty", "pad_token_id")}
    print(f"lens: {lens!r}  lens_pt_sha OK", flush=True)
    print(f"resolved generation: {resolved_gen}", flush=True)

    source_layers = lens.source_layers
    final_layer = model.n_layers - 1
    record_at = sorted(set(source_layers) | {final_layer})
    model_digest = f"Qwen/Qwen2.5-7B-Instruct@{M.get('model_revision','a09a3545')}"

    def readout_rows(full_ids, P):
        with torch.no_grad(), ActivationRecorder(model.layers, at=record_at) as rec:
            model.forward(full_ids)
            acts = {i: rec.activations[i].detach() for i in record_at}
        rows = []
        def emit(layer, logits_cpu, kind):
            vals, idx = logits_cpu.topk(TOPK, dim=-1)
            for pos in range(logits_cpu.shape[0]):
                rows.append({
                    "kind": kind, "layer": layer, "position": pos,
                    "segment": "prompt" if pos < P else "generation",
                    "ood_unfitted_pos": pos < SKIP_FIRST_N,
                    "topk": [{"id": int(idx[pos, k]), "token": tok.decode([int(idx[pos, k])]),
                              "weight": round(float(vals[pos, k]), 4)} for k in range(TOPK)],
                })
        for layer in source_layers:
            resid = acts[layer][0].float()
            logits = model.unembed(lens.transport(resid, layer)).float().cpu()
            emit(layer, logits, "lens")
        emit(final_layer, model.unembed(acts[final_layer][0].float()).float().cpu(), "model_output")
        return rows

    t_start = time.time()
    n_ok = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for ti, t in enumerate(todo, 1):
            d = disorders[t["disorder"]]
            seeded = wrap(wrappers[t["wrapper"]], d["name"], d["payload"], d.get("disclosure", ""))
            user = build_prompt(seeded, dict(high)[t["vignette"]])
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
            prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            P = int(prompt_ids.shape[-1])

            torch.manual_seed(t["run_seed"])
            t0 = time.time()
            with torch.no_grad():
                gen = hf.generate(prompt_ids, generation_config=gen_cfg)
            t_gen = time.time() - t0
            total = int(gen.shape[-1]); G = total - P
            gen_text = tok.decode(gen[0, P:], skip_special_tokens=True)

            rows = readout_rows(gen, P)
            payload = {
                "schema": {"topk": TOPK, "note": "kind=lens|model_output; position absolute; "
                           "segment prompt/generation; ood_unfitted_pos<16; weight raw logit"},
                "meta": {**{k: t[k] for k in ("trial_id","cell","disorder","wrapper","vignette","rep",
                                              "run_seed","canonical_index")},
                         "master_seed": MASTER_SEED, "model": model_digest,
                         "lens": "neuronpedia/jacobian-lens@16a01f3 qwen2.5-7b-it",
                         "lens_pt_sha256": pt_sha, "source_layers": source_layers,
                         "final_layer": final_layer, "prompt_tokens": P, "gen_tokens": G,
                         "total_positions": total, "generation": resolved_gen,
                         "user_prompt_sha256": sh(user.encode("utf-8")),
                         "gen_seconds": round(t_gen, 2), "n_rows": len(rows)},
                "user_prompt": user, "generation_text": gen_text, "rows": rows,
            }
            out_file = OUT / f"{t['trial_id']}.json"
            blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            out_file.write_bytes(blob)
            finite = all(all(isinstance(c["weight"], float) and c["weight"] == c["weight"]
                             for c in r["topk"]) for r in rows[:50])
            mline = {
                **{k: t[k] for k in ("trial_id","cell","disorder","wrapper","vignette","rep",
                                     "run_seed","canonical_index")},
                "master_seed": MASTER_SEED, "prompt_tokens": P, "gen_tokens": G,
                "total_positions": total, "n_rows": len(rows),
                "readout_file": str(out_file.relative_to(REPO)).replace("\\", "/"),
                "readout_sha256": sh(blob), "user_prompt_sha256": sh(user.encode("utf-8")),
                "gen_text_sha256": sh(gen_text.encode("utf-8")), "model_digest": model_digest,
                "lens_pt_sha256": pt_sha, "finite_sample_ok": finite,
            }
            mf.write(json.dumps(mline, ensure_ascii=False) + "\n"); mf.flush()
            n_ok += 1
            if ti % 20 == 0 or args.smoke:
                el = time.time() - t_start
                print(f"[{tag}] {ti}/{len(todo)} {t['trial_id']} P={P} G={G} rows={len(rows)} "
                      f"{t_gen:.1f}s vram={torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                      f"elapsed={el/60:.1f}m", flush=True)

    print(f"[{tag}] DONE new_runs={n_ok} total_elapsed={(time.time()-t_start)/60:.1f}m "
          f"manifest={manifest_path.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
