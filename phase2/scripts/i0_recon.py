#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2 / Stage I0 — reconnaissance for the activation-addition instrument.

Cold re-derivation. NO GPU (CPU-only linear algebra), NO generation, NO conditions,
NO token-set scoring of any run. Produces exactly what PREREG_PHASE2 s0 lists as
I0-dependent:

  C. capability of `jlens` verified IN CODE (not from prose):
     C1 per-layer J_l exposed as [d_model, d_model]
     C2 the readout composition is lm_head(final_norm(h @ J_l.T))
     C3 a forward hook can REPLACE a block's output (the injection mechanism)
     C4 the Qwen2 decoder block returns a bare Tensor in the pinned transformers
  D. the target direction u_F from the 11 sealed Set F SURVIVOR tokens
  E. Tikhonov inverse v_hat_l per layer, lambda by the PREREG s3.2(c) rule
  F. cos_l = cos(J_l v_hat_l, u_F) for every layer in the band 17-26

Writes phase2/data/i0_recon.json and prints a summary. The narrative artifact
phase2/I0_RECON.md is written from that JSON by hand at commit time.

Run as a file:  .venv/Scripts/python.exe phase2/scripts/i0_recon.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys

import torch
import transformers
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "jacobian-lens"))

HF = ROOT / "phase0" / "data" / "hf_cache"
LENS_PT = (HF / "models--neuronpedia--jacobian-lens" / "snapshots"
           / "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a" / "qwen2.5-7b-it"
           / "jlens" / "Salesforce-wikitext" / "Qwen2.5-7B-Instruct_jacobian_lens.pt")
MODEL = (HF / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots"
         / "a09a35458c702b33eeacc393d103063234e8bc28")
SCREEN = ROOT / "phase0" / "data" / "phase1_seal_screening_A1.json"
OUT = ROOT / "phase2" / "data" / "i0_recon.json"

LENS_SHA_PIN = "3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29"
BAND = list(range(17, 27))                      # PREREG_PHASE2 s3.2, inherited
LAMBDA_LADDER = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]   # s3.2(c)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ capability

def capability_checks() -> dict:
    """C1-C4. Every claim is asserted mechanically, not read off documentation."""
    out: dict = {"transformers": transformers.__version__, "torch": torch.__version__}

    # C3 -- a forward hook returning a value REPLACES the module output.
    # This is the mechanism the intervention needs; ActivationRecorder is
    # record-only (jlens/hooks.py), so I1 must add its own injecting hook.
    lin = torch.nn.Linear(4, 4, bias=False)
    torch.nn.init.eye_(lin.weight)
    probe = torch.ones(1, 4)
    bump = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    handle = lin.register_forward_hook(lambda m, i, o: o + bump)
    got = lin(probe)
    handle.remove()
    out["C3_forward_hook_replaces_output"] = bool(
        torch.allclose(got, probe + bump) and torch.allclose(lin(probe), probe)
    )

    # C4 -- what does a Qwen2 decoder block actually return in the pinned version?
    import inspect

    from transformers.models.qwen2 import modeling_qwen2 as mq
    src = inspect.getsource(mq.Qwen2DecoderLayer.forward)
    last = [ln.strip() for ln in src.strip().splitlines() if ln.strip().startswith("return")]
    out["C4_decoder_return_stmt"] = last[-1] if last else None
    out["C4_returns_bare_tensor"] = last[-1] == "return hidden_states" if last else False

    # C2 -- the readout composition, asserted from the vendored source.
    src_unembed = inspect.getsource(
        __import__("jlens.hf", fromlist=["HFLensModel"]).HFLensModel.unembed)
    out["C2_unembed_applies_final_norm"] = "_final_norm" in src_unembed
    out["C2_unembed_src"] = " ".join(src_unembed.split())
    return out


# ------------------------------------------------------------------ inputs

def load_lens() -> tuple[dict[int, torch.Tensor], dict]:
    from jlens.lens import JacobianLens

    got = sha256(LENS_PT)
    lens = JacobianLens.load(str(LENS_PT))
    meta = {"lens_pt": str(LENS_PT.relative_to(ROOT)), "sha256": got,
            "sha256_pin": LENS_SHA_PIN, "sha256_match": got == LENS_SHA_PIN,
            "d_model": lens.d_model, "n_prompts": lens.n_prompts,
            "source_layers": lens.source_layers}
    # C1 -- per-layer J is [d_model, d_model]
    shapes = {l: list(lens.jacobians[l].shape) for l in BAND}
    meta["C1_band_shapes_ok"] = all(s == [lens.d_model, lens.d_model]
                                    for s in shapes.values())
    meta["C1_band_shapes"] = {str(k): v for k, v in shapes.items()}
    meta["band_in_source_layers"] = all(l in lens.source_layers for l in BAND)
    # transport() convention check: transport(h,l) == h @ J.T  (i.e. J @ h)
    l0 = BAND[0]
    h = torch.randn(3, lens.d_model)
    meta["C1_transport_is_J_at_h"] = bool(torch.allclose(
        lens.transport(h, l0), h @ lens.jacobians[l0].T, atol=1e-4))
    return {l: lens.jacobians[l].float() for l in BAND}, meta


def load_head() -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return (W_U [vocab, d_model], final_norm gain g [d_model], meta)."""
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    index = json.loads((MODEL / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wmap = index["weight_map"]
    tied = bool(cfg.get("tie_word_embeddings", False))
    head_key = "model.embed_tokens.weight" if tied else "lm_head.weight"
    norm_key = "model.norm.weight"

    def get(key: str) -> torch.Tensor:
        with safe_open(str(MODEL / wmap[key]), framework="pt") as f:
            return f.get_tensor(key).float()

    W_U = get(head_key)
    g = get(norm_key)
    meta = {"model_dir": str(MODEL.relative_to(ROOT)),
            "model_digest": "Qwen/Qwen2.5-7B-Instruct@a09a3545",
            "tie_word_embeddings": tied, "head_key": head_key,
            "W_U_shape": list(W_U.shape), "norm_gain_shape": list(g.shape),
            "norm_gain_min": float(g.min()), "norm_gain_max": float(g.max()),
            "norm_gain_mean": float(g.mean()), "norm_gain_std": float(g.std())}
    return W_U, g, meta


def f_survivors() -> list[dict]:
    d = json.loads(SCREEN.read_text(encoding="utf-8"))["F_disclosure_fictional"]
    return [r for c in d["concepts"] for r in c["realized"] if r["status"] == "SURVIVES"]


# ------------------------------------------------------------------ solve

def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def sweep(J: torch.Tensor, u: torch.Tensor) -> dict:
    """cos(J v_hat, u) for every lambda on the ladder, at one layer.

    v_hat = (J^T J + lam I)^-1 J^T u, then unit-normalized (s3.2b: magnitude
    lives in alpha, not in lambda). lambda is scaled by the mean eigenvalue of
    J^T J so the ladder is comparable across layers (s3.2c).
    """
    d = J.shape[0]
    G = J.T @ J
    mean_eig = float(torch.diagonal(G).mean())          # tr(G)/d == mean eigenvalue
    rhs = J.T @ u
    eye = torch.eye(d, dtype=J.dtype)
    res = {}
    for lam in LAMBDA_LADDER:
        v = torch.linalg.solve(G + (lam * mean_eig) * eye, rhs)
        v = unit(v)
        Jv = J @ v
        res[repr(lam)] = {
            "cos": float(torch.dot(unit(Jv), u)),
            "norm_Jv": float(Jv.norm()),
        }
    sv = torch.linalg.svdvals(J)
    return {"mean_eig_GtG": mean_eig, "by_lambda": res,
            "sv_max": float(sv[0]), "sv_min": float(sv[-1]),
            "sv_median": float(sv[d // 2]),
            "cond": float(sv[0] / sv[-1]),
            "eff_rank_99": int((torch.cumsum(sv, 0) / sv.sum() < 0.99).sum() + 1)}


def select_lambda(per_layer: dict[int, dict]) -> tuple[float, float]:
    """s3.2(c): one lambda for the whole band, maximizing min_l cos_l.
    Ties -> the larger lambda."""
    best_lam, best_min = None, -2.0
    for lam in LAMBDA_LADDER:                    # ascending -> ties keep the larger
        m = min(per_layer[l]["by_lambda"][repr(lam)]["cos"] for l in BAND)
        if m >= best_min:
            best_min, best_lam = m, lam
    return best_lam, best_min


def main() -> int:
    torch.manual_seed(0)
    cap = capability_checks()
    Js, lens_meta = load_lens()
    if not lens_meta["sha256_match"]:
        print("LENS SHA MISMATCH", lens_meta, file=sys.stderr)
        return 2
    W_U, g, head_meta = load_head()
    surv = f_survivors()
    ids = [r["id"] for r in surv]

    # ---- D. target direction(s) in the final-layer basis -------------------
    # The readout is lm_head(final_norm(x)) = W_U @ (g * x / rms(x)).  The
    # F-token logits are therefore driven by <g * sum_t W_U[t], x>, so the
    # gain-corrected target is the mathematically correct one for the
    # registered estimator. PREREG_PHASE2 s3.1 as drafted omitted the gain.
    # BOTH are computed; the PI decides at freeze which s3.1 says.
    raw_sum = W_U[ids].sum(dim=0)
    u_raw = unit(raw_sum)
    u_gain = unit(g * raw_sum)
    targets = {"u_raw_no_gain": u_raw, "u_gain_corrected": u_gain}
    target_meta = {
        "n_survivors": len(surv),
        "survivors": [{"id": r["id"], "piece": r["piece"], "langs": r["langs"]}
                      for r in surv],
        "cos_raw_vs_gain": float(torch.dot(u_raw, u_gain)),
    }

    # ---- G. reachability ceiling of the loading estimator ------------------
    # The dumped "weight" is the raw LOGIT (run_confirmatory.py: logits.topk,
    # no softmax), read out AFTER final_norm. Since final_norm rescales any x to
    # ||g_free|| with ||x/rms(x)|| = sqrt(d), the summed F logit at one position
    # is bounded:  sum_t W_U[t].(g*y) = (g*sum_t W_U[t]).y  <=  sqrt(d)*||g*sum_t W_U[t]||.
    # That is a HARD analytic ceiling on the s2 loading estimator, independent of
    # alpha: driving the residual harder rotates y toward the target but cannot
    # exceed full alignment. Decision-relevant for the s3.4 2x/10x/50x targets.
    d_model = lens_meta["d_model"]
    ceil_per_pos = float((d_model ** 0.5) * (g * raw_sum).norm())
    natural = 0.0825                       # RESULTS_PHASE1 App. A1, without-mention
    reach = {
        "readout_weight_is": "raw logit (topk of logits, no softmax) after final_norm",
        "ceiling_per_position_summed_F_logit": ceil_per_pos,
        "natural_reference_loading": natural,
        "targets_2x_10x_50x": [2 * natural, 10 * natural, 50 * natural],
        "ceiling_over_natural": ceil_per_pos / natural,
        "note": ("ceiling assumes EVERY generation position in the band is driven to "
                 "perfect alignment AND every F token stays inside top-10; the "
                 "achievable loading is far below it. Reported as a necessary "
                 "condition, not an expectation."),
    }
    print(f"\n[reachability] per-position summed-F-logit ceiling = {ceil_per_pos:.2f}"
          f"  ({ceil_per_pos / natural:.0f}x the natural {natural})"
          f"   50x target = {50 * natural:.3f}")

    results = {}
    for name, u in targets.items():
        per_layer = {l: sweep(Js[l], u) for l in BAND}
        lam, cmin = select_lambda(per_layer)
        results[name] = {
            "per_layer": {str(l): per_layer[l] for l in BAND},
            "selected_lambda": lam,
            "band_min_cos": cmin,
            "cos_at_selected": {str(l): per_layer[l]["by_lambda"][repr(lam)]["cos"]
                                for l in BAND},
        }
        print(f"\n=== target {name} ===  selected lambda={lam:g}  "
              f"min_l cos={cmin:.4f}")
        for l in BAND:
            row = " ".join(f"{per_layer[l]['by_lambda'][repr(x)]['cos']:.3f}"
                           for x in LAMBDA_LADDER)
            print(f"  L{l:>2}  cond={per_layer[l]['cond']:9.1f}  cos[lam]: {row}")

    out = {
        "stage": "Phase 2 / Stage I0 reconnaissance",
        "generated_by": "phase2/scripts/i0_recon.py",
        "host": platform.node(),
        "gpu_used": False,
        "capability": cap,
        "lens": lens_meta,
        "head": head_meta,
        "target": target_meta,
        "lambda_ladder": LAMBDA_LADDER,
        "lambda_scaling": "lam * mean_eigenvalue(J^T J), per layer",
        "band": BAND,
        "reachability": reach,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
