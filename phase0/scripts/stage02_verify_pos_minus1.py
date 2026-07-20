# Copyright 2026 — Phase 0 delegate. Stage 0.2 closing verification (single attempt).
"""Pre-registered check (criterion fixed BEFORE running, per PI):
read the currency multi-hop prompt at position -1 (final), all fitted layers.
SUCCESS = a currency token (euro / lira / € or an obvious morphological /
multilingual variant of either) present in the top-k of some mid-to-late layer
band. To remove k-arbitrariness on this one shot, we print top-10 per layer AND
compute the explicit rank of currency candidate tokens at every layer.
No tuning, no second prompt, no retry.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import transformers

import jlens

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "phase0" / "data" / "stage02_fetch_manifest.json").read_text())
MODEL_DIR, LENS_PT = M["model_path"], M["lens_pt"]

CURRENCY = (
    "Fact: The capital of Japan is Tokyo.\n"
    "Fact: The currency used in the country shaped like a boot is"
)
# Obvious morphological / multilingual variants of euro & lira (+ symbol).
CAND = [" euro", " Euro", " euros", "euro", "Euro", "€", " €", " EUR",
        " lira", " Lira", " lire", " Lire", "lira", "Lira",
        "欧元", "里拉", " ユーロ", " リラ", " евро"]
TOPK = 10


def main() -> int:
    print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).cuda()
    hf.eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PT)
    print("lens:", repr(lens), flush=True)

    # single-token candidate ids
    cand_ids = {}
    for s in CAND:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            cand_ids[s] = ids[0]
    print("single-token currency candidates:", cand_ids, flush=True)

    t0 = time.time()
    ll, ml, ids = lens.apply(model, CURRENCY, positions=[-1])
    print(f"apply {time.time()-t0:.2f}s ; seq_len={ids.shape[-1]} ; read pos = -1\n", flush=True)

    result = {"prompt": CURRENCY, "read_pos": -1, "topk": {}, "currency_rank": {}}
    hit_layers = []
    for layer in sorted(ll):
        logits = ll[layer][0]
        top = logits.topk(TOPK).indices.tolist()
        toks = [tok.decode([t]) for t in top]
        result["topk"][layer] = toks
        # explicit rank of each currency candidate (rank 0 = argmax)
        ranks = {s: int((logits > logits[i]).sum()) for s, i in cand_ids.items()}
        best_s = min(ranks, key=ranks.get) if ranks else None
        best_r = ranks[best_s] if best_s else None
        result["currency_rank"][layer] = {"best_token": best_s, "best_rank": best_r,
                                          "all": ranks}
        flag = ""
        if best_r is not None and best_r < TOPK:
            flag = f"   <== currency '{best_s.strip()}' @ rank {best_r}"
            hit_layers.append((layer, best_s.strip(), best_r))
        print(f"L{layer:>2}: {toks}   [best currency: {best_s.strip() if best_s else None} r={best_r}]{flag}")

    print("\n=== currency in top-{}: layers {} ===".format(
        TOPK, [(l, s, r) for l, s, r in hit_layers]))
    verdict = "GREEN" if hit_layers else "NOT-GREEN"
    result["hit_layers"] = hit_layers
    result["verdict"] = verdict
    print("VERDICT:", verdict)
    out = REPO / "phase0" / "data" / "stage02_verify_minus1.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
