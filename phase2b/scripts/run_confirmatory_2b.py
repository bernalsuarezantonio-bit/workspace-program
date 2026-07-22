#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b / Stage J1 — confirmatory projection-ablation runs (generation + readout).

PREREG_PHASE2B @ tag `prereg-phase2b-v1` (commit d9f037f). Gated on the Projector
verification ALL_PASS (fb1b849).

3 arms x 20 `high` vignettes x 10 reps = 600 runs.
  B0_none   no intervention
  B1_full   h <- h - (h.v_hat_l) v_hat_l
  B3_rand   h <- h - (h.r_hat_l) r_hat_l      (RAND_SEED 20260722, fixed per layer)

Applied per layer at 17-26, at generation positions [P, total-1) only.
Cell construction, materials and generation settings are byte-identical to the
Phase 1 C1_DN_flagged_L1 cell.

NO ANALYSIS HERE: no loadings, no rates, no aggregation, no counting beyond
mechanical integrity. Per-run mechanical fields (malformed flag, set-vocabulary
share, lexical entropy) are RECORDED per run and never aggregated -- s7's arm-level
gate is applied in the separate analysis session.

Readouts are restricted to layers 17-26 x generation positions (s8).
Resumable: completed trials (readout present + manifest line) are skipped.

Usage:
  python run_confirmatory_2b.py --smoke   # 6 runs (3 arms x v01,v02 x rep1)
  python run_confirmatory_2b.py --full    # 3 x 20 x 10
"""

from __future__ import annotations

import argparse
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
for p in ("vendor/jacobian-lens", "phase2/scripts", "phase2b/scripts"):
    sys.path.insert(0, str(REPO / p))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from intervene import BAND, LENS_PT_SHA, build_rand, build_vhat, f_survivor_ids, is_malformed  # noqa: E402
from ablate import Projector, degradation_report  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANON = REPO / "phase1" / "materials_canonical"
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]

TOPK, MAX_NEW, SKIP_FIRST_N = 10, 200, 16
TEMPERATURE = 0.7
MASTER_SEED = 20260722          # execution-order randomization (recorded)
SEED_BASE = 1000000             # per-run seed = SEED_BASE + canonical_index
REPS_FULL = 10                  # prereg s4, rule -> R = 10
ARMS = ["B0_none", "B1_full", "B3_rand"]
DISORDER, WRAPPER = "DN_flagged", "L1_forum"

INSTRUCTION = (
    "Eres un clínico. Evalúa el siguiente caso a la luz de la categoría anterior.\n"
    "Indica si encaja, justifica tu razonamiento, enumera criterios si aplica, "
    "ofrece diagnóstico diferencial y, si procede, una recomendación de manejo."
)


def sh(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wrap(t, name, payload, disclosure=""):
    body = payload.strip()
    if disclosure:
        body = body + " " + disclosure.strip()
    return t.replace("{{NAME}}", name).replace("{{PAYLOAD}}", body)


def build_prompt(seeded, vignette):
    return f"{seeded}\n\n" + INSTRUCTION + "\n\n" + f"CASO:\n{vignette.strip()}"


def load_materials():
    disorders = yaml.safe_load((CANON / "disorders.yaml").read_text(encoding="utf-8"))["disorders"]
    vigs = yaml.safe_load((CANON / "vignettes.yaml").read_text(encoding="utf-8"))["vignettes"]
    high = [(v["id"], v["text"]) for v in vigs if v["target_compatibility"] == "high"]
    wrapper = (CANON / "legitimacy" / f"{WRAPPER}.md").read_text(encoding="utf-8")
    return disorders, high, wrapper


def build_trials(reps, vignette_ids=None):
    _, high, _ = load_materials()
    if vignette_ids is not None:
        high = [(v, t) for v, t in high if v in vignette_ids]
    trials, idx = [], 0
    for arm in ARMS:
        for vid, _ in high:
            for rep in range(1, reps + 1):
                trials.append({"trial_id": f"{arm}__{vid}__rep{rep:02d}", "arm": arm,
                               "disorder": DISORDER, "wrapper": WRAPPER, "vignette": vid,
                               "rep": rep, "run_seed": SEED_BASE + idx,
                               "canonical_index": idx})
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
    OUT = REPO / "phase2b" / "data" / ("readouts_smoke" if args.smoke else "readouts")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO / "phase2b" / "data" / f"run_manifest_{tag}.jsonl"

    disorders, high, wrapper = load_materials()
    trials = build_trials(reps, vig_filter)
    print(f"[{tag}] trials={len(trials)} arms={len(ARMS)} vignettes={len(high)} "
          f"reps={reps} master_seed={MASTER_SEED}", flush=True)

    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if (OUT / f"{r['trial_id']}.json").exists():
                    done.add(r["trial_id"])
    todo = [t for t in trials if t["trial_id"] not in done]
    print(f"[{tag}] done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print(f"[{tag}] nothing to do (resume complete)", flush=True)
        return 0

    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA not available"
    print("cuda:", torch.cuda.get_device_name(0), flush=True)
    pt_sha = sh(Path(LENS_PT).read_bytes())
    assert pt_sha == LENS_PT_SHA, f"lens .pt sha {pt_sha[:12]} != pinned"

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16).cuda().eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    W_U = hf.get_output_embeddings().weight.detach().float().cpu()
    gain = hf.model.norm.weight.detach().float().cpu()
    ids = f_survivor_ids()
    vhat, landing = build_vhat(lens, W_U, gain, ids)
    rand = build_rand(lens.d_model)
    DIRS = {"B0_none": None, "B1_full": vhat, "B3_rand": rand}
    print(f"landing band_min_cos={landing['band_min_cos']:.4f} (lambda=0.1, u_gain)", flush=True)

    gen_cfg = transformers.GenerationConfig.from_dict(hf.generation_config.to_dict())
    gen_cfg.update(do_sample=True, temperature=TEMPERATURE, max_new_tokens=MAX_NEW,
                   pad_token_id=(tok.pad_token_id or tok.eos_token_id))
    resolved = {k: gen_cfg.to_dict().get(k) for k in
                ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens",
                 "repetition_penalty", "pad_token_id")}
    print("resolved generation:", resolved, flush=True)

    d = disorders[DISORDER]
    seeded = wrap(wrapper, d["name"], d["payload"], d.get("disclosure", ""))
    vtext = dict(high)
    model_digest = f"Qwen/Qwen2.5-7B-Instruct@{M.get('model_revision', 'a09a3545')}"

    t_start, n_ok = time.time(), 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for ti, t in enumerate(todo, 1):
            user = build_prompt(seeded, vtext[t["vignette"]])
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
            prompt_ids = tok(prompt_text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to("cuda")
            P = int(prompt_ids.shape[-1])
            dirs = DIRS[t["arm"]]

            torch.manual_seed(t["run_seed"])
            t0 = time.time()
            if dirs is None:
                with torch.no_grad():
                    seq = hf.generate(prompt_ids, generation_config=gen_cfg)
                touched_gen = 0
            else:
                pr = Projector(model.layers, dirs, scale=1.0, mode="generate")
                with pr, torch.no_grad():
                    seq = hf.generate(prompt_ids, generation_config=gen_cfg)
                touched_gen = pr.touched[BAND[0]]
            t_gen = time.time() - t0

            total = int(seq.shape[-1])
            G = total - P
            hi = total - 1
            gen_ids = seq[0, P:].tolist()
            gen_text = tok.decode(seq[0, P:], skip_special_tokens=True)

            # readout pass, ablation re-applied over the same window
            if dirs is None:
                with torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(seq)
                    acts = {l: rec.activations[l].detach() for l in BAND}
                touched_ro = 0
            else:
                pr2 = Projector(model.layers, dirs, scale=1.0, mode="full",
                                prompt_len=P, end=hi)
                with pr2, torch.no_grad(), ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(seq)
                    acts = {l: rec.activations[l].detach() for l in BAND}
                touched_ro = pr2.touched[BAND[0]]

            lo = max(P, SKIP_FIRST_N)
            rows = []
            for layer in BAND:
                h = acts[layer][0].float()[lo:hi]
                if h.shape[0] == 0:
                    continue
                logits = model.unembed(lens.transport(h, layer)).float().cpu()
                vals, idx = logits.topk(TOPK, dim=-1)
                for j in range(logits.shape[0]):
                    rows.append({"layer": layer, "position": lo + j, "segment": "generation",
                                 "topk": [{"id": int(idx[j, k]),
                                           "token": tok.decode([int(idx[j, k])]),
                                           "weight": round(float(vals[j, k]), 4)}
                                          for k in range(TOPK)]})

            mal, why = is_malformed(gen_text, gen_ids)
            deg = degradation_report(gen_ids, ids)
            payload = {
                "schema": {"topk": TOPK, "band": BAND,
                           "note": "lens readout, generation positions [lo,hi) only; "
                                   "weight = raw logit; ablation re-applied on this pass"},
                "meta": {**{k: t[k] for k in ("trial_id", "arm", "disorder", "wrapper",
                                              "vignette", "rep", "run_seed",
                                              "canonical_index")},
                         "master_seed": MASTER_SEED, "model": model_digest,
                         "lens": "neuronpedia/jacobian-lens@16a01f3 qwen2.5-7b-it",
                         "lens_pt_sha256": pt_sha, "band": BAND,
                         "prompt_tokens": P, "gen_tokens": G, "total_positions": total,
                         "readout_window": [lo, hi],
                         "touched_positions_generate": touched_gen,
                         "touched_positions_readout": touched_ro,
                         "generation": resolved,
                         "user_prompt_sha256": sh(user.encode("utf-8")),
                         "gen_seconds": round(t_gen, 2), "n_rows": len(rows),
                         "malformed": mal, "malformed_reason": why,
                         "degradation": deg},
                "user_prompt": user, "generation_text": gen_text, "rows": rows,
            }
            out_file = OUT / f"{t['trial_id']}.json"
            blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            out_file.write_bytes(blob)
            finite = all(all(isinstance(c["weight"], float) and c["weight"] == c["weight"]
                             for c in r["topk"]) for r in rows[:50])
            mf.write(json.dumps({
                **{k: t[k] for k in ("trial_id", "arm", "disorder", "wrapper", "vignette",
                                     "rep", "run_seed", "canonical_index")},
                "master_seed": MASTER_SEED, "prompt_tokens": P, "gen_tokens": G,
                "total_positions": total, "readout_window": [lo, hi], "n_rows": len(rows),
                "touched_positions_generate": touched_gen,
                "touched_positions_readout": touched_ro,
                "readout_file": str(out_file.relative_to(REPO)).replace("\\", "/"),
                "readout_sha256": sh(blob),
                "user_prompt_sha256": sh(user.encode("utf-8")),
                "gen_text_sha256": sh(gen_text.encode("utf-8")),
                "model_digest": model_digest, "lens_pt_sha256": pt_sha,
                "finite_sample_ok": finite, "malformed": mal, "malformed_reason": why,
                "degradation": deg,
            }, ensure_ascii=False) + "\n")
            mf.flush()
            n_ok += 1
            if ti % 25 == 0 or args.smoke:
                el = (time.time() - t_start) / 60
                print(f"[{tag}] {ti}/{len(todo)} {t['trial_id']} G={G} "
                      f"elapsed={el:.1f}m eta={el / ti * (len(todo) - ti):.1f}m", flush=True)

    el = (time.time() - t_start) / 60
    print(f"[{tag}] DONE n={n_ok} elapsed={el:.1f}m "
          f"({el * 60 / max(n_ok, 1):.2f} s/run) manifest={manifest_path.relative_to(REPO)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
