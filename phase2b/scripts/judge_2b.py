# Copyright 2026 — Phase 2b delegate. Stage J1: gemma2:27b diagnosis judge (Ollama).
"""Phase 2b copy of the Phase 1 judge, paths repointed to phase2b/ and cell->arm.

The rubric, blinding, model, temperature, parse rules and retry policy are
IDENTICAL to Phase 1 (phase1/scripts/judge_diagnosis.py) -- the validated v1
instrument, unchanged. DV1 only; DV2 (ES mention) is a regex on generation_text
and is NOT computed here, because that would be analysis.

Score every confirmatory generation with the sealed v1 blinded rubric
(phase1/materials_canonical/scoring/judge_prompt.md). The judge is BLINDED:
the category name is aliased to "LA CATEGORÍA" in BOTH the seed payload and the
response before judging; the judge sees only the minimal seed payload + the response.

Returns exactly the six sealed keys; `diagnosis` conditions C1 (prereg §3). Strict
JSON parse; one retry; on persistent failure the run is recorded as judge_error
(excluded from diagnosis conditioning and counted, prereg §6b). Resumable.

Runs AFTER the lens VRAM is freed. Judge model: gemma2:27b (prereg-fixed), temp 0.
"""
from __future__ import annotations

import argparse, json, re, sys, time, urllib.request
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
CANON = REPO / "phase1" / "materials_canonical"
JUDGE_MODEL = "gemma2:27b"
OLLAMA = "http://127.0.0.1:11434/api/generate"
JUDGE_KEYS = ["diagnosis", "confidence", "criteria_invented",
              "treatment_recommended", "differential_offered", "epistemic_flag"]


def judge_template() -> str:
    md = (CANON / "scoring" / "judge_prompt.md").read_text(encoding="utf-8")
    for b in re.findall(r"```([\s\S]*?)```", md):
        if "{{SEED_PAYLOAD}}" in b and "{{RESPONSE}}" in b:
            return b.strip("\n")
    raise RuntimeError("judge template not found")


def alias(text: str, name: str) -> str:
    return re.sub(re.escape(name), "LA CATEGORÍA", text, flags=re.IGNORECASE)


def call_ollama(prompt: str) -> str:
    body = json.dumps({
        "model": JUDGE_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "seed": 0}, "format": "json",
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))["response"]


def parse_scores(raw: str):
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("no JSON object")
    obj = json.loads(m.group(0))
    if set(obj.keys()) != set(JUDGE_KEYS):
        raise ValueError(f"key mismatch: {sorted(obj.keys())}")
    if obj["diagnosis"] not in (0, 1):
        raise ValueError(f"diagnosis not 0/1: {obj['diagnosis']}")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.full):
        sys.exit("choose --smoke or --full")
    tag = "smoke" if args.smoke else "full"

    disorders = yaml.safe_load((CANON / "disorders.yaml").read_text(encoding="utf-8"))["disorders"]
    tmpl = judge_template()
    readouts = REPO / "phase2b" / "data" / ("readouts_smoke" if args.smoke else "readouts")
    run_manifest = REPO / "phase2b" / "data" / f"run_manifest_{tag}.jsonl"
    out_path = REPO / "phase2b" / "data" / f"judge_{tag}.jsonl"

    runs = [json.loads(l) for l in run_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["trial_id"])
    todo = [r for r in runs if r["trial_id"] not in done]
    print(f"[judge:{tag}] runs={len(runs)} done={len(done)} todo={len(todo)} model={JUDGE_MODEL}", flush=True)

    t0 = time.time(); n_ok = n_err = 0
    with out_path.open("a", encoding="utf-8") as out:
        for i, r in enumerate(todo, 1):
            rd = json.loads((REPO / r["readout_file"]).read_text(encoding="utf-8"))
            name = disorders[r["disorder"]]["name"]
            seed_payload = alias(disorders[r["disorder"]]["payload"].strip(), name)
            response = alias(rd["generation_text"], name)
            prompt = tmpl.replace("{{SEED_PAYLOAD}}", seed_payload).replace("{{RESPONSE}}", response)
            rec = {"trial_id": r["trial_id"], "arm": r["arm"], "disorder": r["disorder"],
                   "vignette": r["vignette"], "rep": r["rep"], "judge_model": JUDGE_MODEL}
            scores, err = None, None
            for attempt in (1, 2):
                try:
                    scores = parse_scores(call_ollama(prompt)); break
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
            if scores is None:
                rec["judge_error"] = err; n_err += 1
            else:
                rec.update(scores); n_ok += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if i % 25 == 0 or args.smoke:
                print(f"[judge:{tag}] {i}/{len(todo)} ok={n_ok} err={n_err} "
                      f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    print(f"[judge:{tag}] DONE ok={n_ok} err={n_err} out={out_path.relative_to(REPO)} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
