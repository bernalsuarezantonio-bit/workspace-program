#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b — the projection-ablation instrument (shared module).

    h' = h - s * (h . v_hat_l) v_hat_l     for pos in GENERATION positions,
                                           per layer l in 17..26

s = 1.0 full ablation, s = 0.5 partial (sensitivity arm).

Unlike Phase 2's addition, a projection can only REMOVE a component: it cannot
force tokens into the output distribution, which is the failure mode that closed
Phase 2 (CLOSURE.md lesson #6). The Phase 2 direction construction (u_gain,
Tikhonov lambda = 0.1) and the verified hook mechanics -- including the KV-cache
generation asymmetry, windows [P, total-1) -- carry forward unchanged.
"""

from __future__ import annotations

import math
from collections import Counter

import torch


class Projector:
    """Removes the v_hat component from the residual at GENERATION positions.

    Same mechanics and same windows as Phase 2's Injector (verified at 5453270):
    mode='generate' skips each layer's first forward call (the prompt pass);
    mode='full' ablates positions [prompt_len, end) with end = total-1.
    Registered BEFORE ActivationRecorder so recorded residuals are post-ablation.
    """

    def __init__(self, blocks, vhat: dict[int, torch.Tensor], *, scale: float,
                 mode: str, prompt_len: int | None = None,
                 end: int | None = None) -> None:
        self.blocks, self.vhat, self.scale, self.mode = blocks, vhat, scale, mode
        self.prompt_len, self.end = prompt_len, end
        self.calls: dict[int, int] = {l: 0 for l in vhat}
        self.touched: dict[int, int] = {l: 0 for l in vhat}
        self.removed_norm: dict[int, float] = {l: 0.0 for l in vhat}
        self._handles: list = []
        if mode == "full" and prompt_len is None:
            raise ValueError("mode='full' needs prompt_len")

    def _apply(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        comp = (t * v).sum(-1, keepdim=True)          # [..., 1]
        return t - self.scale * comp * v

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            n = self.calls[layer]
            self.calls[layer] = n + 1
            v = self.vhat[layer].to(t.device, t.dtype)
            if self.mode == "generate":
                if n == 0:
                    return None
                new = self._apply(t, v)
                self.touched[layer] += int(t.shape[-2])
            else:
                p = self.prompt_len
                e = self.end if self.end is not None else int(t.shape[-2])
                if t.shape[-2] <= p or e <= p:
                    return None
                new = t.clone()
                new[..., p:e, :] = self._apply(t[..., p:e, :], v)
                self.touched[layer] += int(e - p)
            self.removed_norm[layer] += float((new - t).norm())
            return new if torch.is_tensor(output) else (new, *output[1:])
        return hook

    def __enter__(self) -> Projector:
        for l in self.vhat:
            self._handles.append(self.blocks[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# --------------------------------------------------------------------------
# Extended degradation gate (CLOSURE.md lesson #6): a repetition-only detector
# passes fluent, semantically vacuous output saturated with a driven set's
# vocabulary. Any set-directed intervention needs a set-vocabulary-share term
# and a lexical-entropy term alongside the Phase 2 s7 checks.
# --------------------------------------------------------------------------

def set_vocab_share(gen_ids: list[int], set_ids: list[int]) -> float:
    """Fraction of generated tokens belonging to the driven set."""
    if not gen_ids:
        return float("nan")
    s = set(set_ids)
    return sum(1 for t in gen_ids if t in s) / len(gen_ids)


def lexical_entropy(gen_ids: list[int]) -> float:
    """Shannon entropy (nats) of the generated token distribution."""
    if not gen_ids:
        return float("nan")
    n = len(gen_ids)
    return -sum((c / n) * math.log(c / n) for c in Counter(gen_ids).values())


def degradation_report(gen_ids: list[int], set_ids: list[int]) -> dict:
    return {"set_vocab_share": set_vocab_share(gen_ids, set_ids),
            "lexical_entropy": lexical_entropy(gen_ids),
            "n_tokens": len(gen_ids)}
