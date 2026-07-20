# Copyright 2026 — Phase 0 delegate. Stage 0.2 fetch + verify (no VRAM).
"""Download the pinned model + lens to a project-local (gitignored) HF cache,
compute sha256 for provenance, and run the issue-#6 torch.isfinite() guard on
the lens CPU-side (map_location='cpu' — never touches the GPU).

Pins (from PROVENANCE.md):
  model : Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28
  lens  : neuronpedia/jacobian-lens @ 16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a
          (tag qwen-n1000), file qwen2.5-7b-it/jlens/Salesforce-wikitext/...
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Hard guard: this script must never allocate VRAM (colleague's run is pinned).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "phase0" / "data" / "hf_cache"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REV = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
LENS_SUBDIR = "qwen2.5-7b-it"  # only fetch the Qwen2.5-7B-it lens, not all 38 dirs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    result: dict = {"pins": {"model": f"{MODEL_REPO}@{MODEL_REV}",
                             "lens": f"{LENS_REPO}@{LENS_REV}"}}

    print(f"[1/4] snapshot_download model {MODEL_REPO}@{MODEL_REV[:8]} ...", flush=True)
    model_path = Path(snapshot_download(
        repo_id=MODEL_REPO, revision=MODEL_REV, cache_dir=str(CACHE),
    ))
    print(f"      -> {model_path}", flush=True)

    print(f"[2/4] snapshot_download lens {LENS_REPO}@{LENS_REV[:8]} "
          f"(allow {LENS_SUBDIR}/**) ...", flush=True)
    lens_root = Path(snapshot_download(
        repo_id=LENS_REPO, revision=LENS_REV, cache_dir=str(CACHE),
        allow_patterns=[f"{LENS_SUBDIR}/**"],
    ))
    print(f"      -> {lens_root}", flush=True)

    # locate lens .pt + sidecars
    pts = sorted(lens_root.rglob("*_jacobian_lens.pt"))
    csvs = sorted(lens_root.rglob("*convergence.csv"))
    cfgs = sorted(lens_root.rglob("config.yaml"))
    if not pts:
        print("ERROR: no *_jacobian_lens.pt found under lens snapshot", file=sys.stderr)
        return 2
    lens_pt = pts[0]

    print("[3/4] sha256 (model shards + lens files) ...", flush=True)
    model_hashes = {}
    for f in sorted(model_path.rglob("*")):
        if f.is_file() and f.suffix in {".safetensors", ".json", ".txt"}:
            model_hashes[f.name] = sha256(f)
    lens_hashes = {p.name: sha256(p) for p in (pts + csvs + cfgs)}

    print("[4/4] issue-#6 torch.isfinite() guard on lens (CPU, map_location=cpu) ...",
          flush=True)
    obj = torch.load(str(lens_pt), map_location="cpu", weights_only=False)
    # The lens .pt is a state dict / object; walk every tensor it contains.
    tensors = []

    def collect(x):
        if isinstance(x, torch.Tensor):
            tensors.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                collect(v)
        elif hasattr(x, "state_dict") and callable(getattr(x, "state_dict")):
            try:
                collect(x.state_dict())
            except Exception:
                pass
        elif hasattr(x, "__dict__"):
            collect(vars(x))

    collect(obj)
    n_tensors = len(tensors)
    n_bad = 0
    per_layer_finite = True
    for t in tensors:
        if t.is_floating_point() or t.is_complex():
            if not bool(torch.isfinite(t).all()):
                per_layer_finite = False
                n_bad += 1
    verdict = "PASS" if per_layer_finite else "FAIL"
    print(f"      tensors scanned: {n_tensors}; non-finite tensors: {n_bad}; "
          f"ISFINITE={verdict}", flush=True)

    result.update({
        "model_path": str(model_path),
        "lens_root": str(lens_root),
        "lens_pt": str(lens_pt),
        "lens_pt_sha256": lens_hashes.get(lens_pt.name),
        "model_footprint_bytes": dir_bytes(model_path),
        "lens_footprint_bytes": dir_bytes(lens_root),
        "model_hashes": model_hashes,
        "lens_hashes": lens_hashes,
        "isfinite": {"tensors": n_tensors, "non_finite_tensors": n_bad,
                     "verdict": verdict},
    })
    out = REPO_ROOT / "phase0" / "data" / "stage02_fetch_manifest.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nMANIFEST -> {out}")
    print(f"model footprint : {result['model_footprint_bytes']/1e9:.2f} GB")
    print(f"lens  footprint : {result['lens_footprint_bytes']/1e6:.2f} MB")
    print(f"lens .pt sha256 : {result['lens_pt_sha256']}")
    print(f"ISFINITE guard  : {verdict}")
    return 0 if per_layer_finite else 3


if __name__ == "__main__":
    raise SystemExit(main())
