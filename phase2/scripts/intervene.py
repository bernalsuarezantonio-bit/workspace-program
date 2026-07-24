#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2 / Stage I1 — the activation-addition instrument (shared module).

Implements PREREG_PHASE2 @ tag `prereg-phase2-v1` (commit 72bf075), frozen values:

  target      u_F = u_gain = unit( g * sum_{t in F_survivors} W_U[t] )     (s3.1, A-2)
  direction   v_hat_l = unit( (J_l^T J_l + lam*mean_eig*I)^-1 J_l^T u_F ), lam = 0.1  (s3.2, A-1)
  dose        h_{l,pos} += k * rho_l * v_hat_l   for pos in GENERATION positions,
              re-added independently at each layer l in 17..26                (s3.4)
  control     A4_rand: c_l * r_l with ||c_l r_l|| == ||k3 rho_l v_hat_l||,
              r_l a FIXED unit vector per layer from RAND_SEED = 20260722     (s2)

The injecting hook is required because jlens's ActivationRecorder is record-only
(I0 finding, phase2/I0_RECON.md s1 C3). Registration order matters: the injector
is registered BEFORE the recorder so the recorded residual is the POST-addition
value -- i.e. what downstream layers and the lens readout actually see.

No experiment logic here; runners import this.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]

# ---- frozen constants from the tagged prereg -------------------------------
BAND = list(range(17, 27))
LAMBDA = 0.1                     # s3.2(c), amendment A-1 accepted
RAND_SEED = 20260722             # s2, A4_rand
K_LADDER = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]        # s3.4, fixed pre-pilot
NATURAL_F = 0.0825               # RESULTS_PHASE1 App. A1 (without-mention)
TARGETS = {"2x": 2 * NATURAL_F, "10x": 10 * NATURAL_F, "50x": 50 * NATURAL_F}
SKIP_FIRST_N = 16                # R4

LENS_PT_SHA = "3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29"
SCREEN = REPO / "phase0" / "data" / "phase1_seal_screening_A1.json"
RHO_JSON = REPO / "phase2" / "data" / "rho_layers.json"


def f_survivor_ids() -> list[int]:
    """The 11 sealed Set F SURVIVOR token ids (seal A1). Matching by id is
    equivalent to Phase 1's folded-form match over exactly these pieces."""
    d = json.loads(SCREEN.read_text(encoding="utf-8"))["F_disclosure_fictional"]
    ids = [r["id"] for c in d["concepts"] for r in c["realized"] if r["status"] == "SURVIVES"]
    assert len(ids) == 11, f"expected 11 Set F survivors, got {len(ids)}"
    return ids


def load_rho() -> dict[int, float]:
    r = json.loads(RHO_JSON.read_text(encoding="utf-8"))
    return {l: float(r["rho_l"][str(l)]) for l in BAND}


def build_vhat(lens, W_U: torch.Tensor, gain: torch.Tensor,
               ids: list[int]) -> tuple[dict[int, torch.Tensor], dict]:
    """v_hat_l for every band layer, at the frozen lambda, plus the landing report."""
    u = gain * W_U[ids].sum(dim=0)
    u = (u / u.norm()).float()
    vhat, report = {}, {}
    for l in BAND:
        J = lens.jacobians[l].float()
        G = J.T @ J
        mean_eig = float(torch.diagonal(G).mean())
        v = torch.linalg.solve(G + (LAMBDA * mean_eig) * torch.eye(J.shape[0]), J.T @ u)
        v = v / v.norm()
        Jv = J @ v
        vhat[l] = v
        report[str(l)] = {"cos": float(torch.dot(Jv / Jv.norm(), u)),
                          "norm_Jv": float(Jv.norm())}
    return vhat, {"lambda": LAMBDA, "target": "u_gain", "per_layer": report,
                  "band_min_cos": min(v["cos"] for v in report.values())}


def build_rand(d_model: int) -> dict[int, torch.Tensor]:
    """One FIXED unit vector per layer, identical across all runs/vignettes/reps."""
    gen = torch.Generator().manual_seed(RAND_SEED)
    out = {}
    for l in BAND:
        r = torch.randn(d_model, generator=gen)
        out[l] = r / r.norm()
    return out


class Injector:
    """Adds a per-layer vector to the residual at GENERATION positions.

    mode='generate': skip each layer's FIRST forward call (the prompt pass) and
        add to every position of all later calls -- with a KV cache those are
        exactly the generated tokens.
    mode='full': one pass over the whole sequence; add at positions
        [prompt_len, end). `end` MUST be set to total-1 to reproduce generation
        faithfully -- see the asymmetry note below.

    Registered BEFORE ActivationRecorder so recorded residuals are post-addition.

    GENERATION ASYMMETRY (mechanical, verified by verify_injection.py V5).
    With a KV cache, the token at the LAST generated position is never fed back
    as an input, so its residual is never computed during generation and cannot
    receive the addition. Generation therefore injects at positions
    [P, total-1) -- exactly total-1-P of them -- while a naive teacher-forced
    pass would cover one position more. This is inherent, not a bug: the last
    token's residual has no downstream effect. Both the injection window and the
    readout window are set to [P, total-1) so that what is measured is exactly
    what the model computed.
    """

    def __init__(self, blocks, deltas: dict[int, torch.Tensor], *, mode: str,
                 prompt_len: int | None = None, end: int | None = None) -> None:
        self.blocks, self.deltas, self.mode = blocks, deltas, mode
        self.prompt_len, self.end = prompt_len, end
        self.calls: dict[int, int] = {l: 0 for l in deltas}
        self.added: dict[int, int] = {l: 0 for l in deltas}
        self._handles: list = []
        if mode == "full" and prompt_len is None:
            raise ValueError("mode='full' needs prompt_len")

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            n = self.calls[layer]
            self.calls[layer] = n + 1
            d = self.deltas[layer].to(t.device, t.dtype)
            if self.mode == "generate":
                if n == 0:
                    return None                      # prompt pass: untouched
                t = t + d
                self.added[layer] += int(t.shape[-2])
            else:
                p = self.prompt_len
                e = self.end if self.end is not None else int(t.shape[-2])
                if t.shape[-2] <= p or e <= p:
                    return None
                t = t.clone()
                t[..., p:e, :] = t[..., p:e, :] + d
                self.added[layer] += int(e - p)
            return t if torch.is_tensor(output) else (t, *output[1:])
        return hook

    def __enter__(self) -> Injector:
        for l in self.deltas:
            self._handles.append(self.blocks[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def deltas_for(arm: str, k: float, k_top: float, vhat, rand, rho) -> dict[int, torch.Tensor]:
    """alpha_l * direction_l for the requested arm. A4_rand is norm-matched to k_top."""
    if arm == "base":
        return {}
    if arm == "rand":
        return {l: (k_top * rho[l]) * rand[l] for l in BAND}
    return {l: (k * rho[l]) * vhat[l] for l in BAND}


@torch.no_grad()
def f_loading(model, lens, acts: dict[int, torch.Tensor], ids: list[int],
              lo: int, hi: int, topk: int = 10) -> float:
    """PREREG s2 estimator restricted to [lo, hi): mean over band layers of the
    mean over positions of the summed top-k logits of Set F operative tokens.

    Matches Phase 1 exactly: a token contributes only when it ranks in the top-k
    at that (layer, position) -- the fixed instrument property (s2).
    """
    if hi <= lo:
        return float("nan")
    id_t = torch.tensor(ids)
    per_layer = []
    for l in BAND:
        h = acts[l][0].float()[lo:hi]
        logits = model.unembed(lens.transport(h, l)).float().cpu()
        vals, idx = logits.topk(topk, dim=-1)
        hit = (idx.unsqueeze(-1) == id_t.view(1, 1, -1)).any(-1)      # [pos, topk]
        per_layer.append(float((vals * hit).sum(-1).mean()))
    return sum(per_layer) / len(per_layer)


def is_malformed(gen_text: str, gen_ids: list[int]) -> tuple[bool, str]:
    """PREREG s7, mechanical, applied before any judging."""
    if len(gen_ids) == 0:
        return True, "zero_generated_tokens"
    try:
        gen_text.encode("utf-8").decode("utf-8")
    except UnicodeError:
        return True, "non_utf8"
    n = len(gen_ids)
    for size in range(1, 11):                       # n-gram of <= 10 tokens
        if size > n:
            break
        counts: dict[tuple, int] = {}
        for i in range(n - size + 1):
            g = tuple(gen_ids[i:i + size])
            counts[g] = counts.get(g, 0) + 1
        best = max(counts.values())
        if best * size > 0.5 * n:
            return True, f"repetition_{size}gram_x{best}"
    return False, ""
