#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b / Stage J0 power analysis — projection ablation, two registered DVs.

NO GPU. NO new data. Both DVs are anchored on REAL Phase 1 rates from the
committed data commit a715ce4:

  DV1 diagnosis rate      200/200 = 1.000 in C1_DN_flagged_L1 (ceiling)
  DV2 ES textual mention  92/200 = 0.460, recomputed per vignette from the
                          committed readouts with the REGISTERED regex
                          (inventad|estudio|no reconocid|fictici) -- the same
                          estimator as RESULTS_PHASE1 App. A1's split, whose
                          92/108 counts this reproduces exactly.

DV2's between-vignette ICC is estimated from those real per-vignette counts, so
unlike Phase 2's power analysis it does not have to fall back on a proxy cell.

Run as a file:  .venv/Scripts/python.exe phase2b/scripts/phase2b_j0_power.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
JUDGE = ROOT / "phase1" / "data" / "judge_full.jsonl"
MANIFEST = ROOT / "phase1" / "data" / "run_manifest_full.jsonl"
COMPLETE = ROOT / "phase1" / "data" / "completeness_report.json"
READOUTS = ROOT / "phase1" / "data" / "readouts"
OUT = ROOT / "phase2b" / "data" / "phase2b_j0_power.json"

DATA_DIGEST = "dc522361096bae30377ecf05d37142cfcb3f52fbb6349c77825bea455f0fb8f1"
CELL = "C1_DN_flagged_L1"
MENTION_RX = re.compile(r"inventad|estudio|no reconocid|fictici", re.I)   # registered

N_VIGNETTES = 20
N_SIMS = 20000
SEED = 20260722
ALPHA = 0.025                    # Bonferroni /2 over the two registered DVs
T_CRIT_ONESIDED_DF19 = -2.093


def gate0() -> dict:
    h = hashlib.sha256()
    for p in (MANIFEST, JUDGE, COMPLETE):
        h.update(p.read_bytes())
    got = h.hexdigest()
    return {"recomputed": got, "expected": DATA_DIGEST, "match": got == DATA_DIGEST}


def diagnosis_by_vignette() -> dict[str, list[int]]:
    by = defaultdict(list)
    with JUDGE.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["cell"] == CELL and isinstance(r.get("diagnosis"), int):
                by[r["vignette"]].append(r["diagnosis"])
    return dict(by)


def mention_by_vignette() -> dict[str, list[int]]:
    by = defaultdict(list)
    for f in sorted(glob.glob(str(READOUTS / f"{CELL}__*.json"))):
        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        by[d["meta"]["vignette"]].append(1 if MENTION_RX.search(d["generation_text"]) else 0)
    return dict(by)


def icc_anova(by_v: dict[str, list[int]]) -> dict:
    groups = [np.asarray(v, dtype=float) for v in by_v.values()]
    k = len(groups)
    ns = np.array([g.size for g in groups], dtype=float)
    N = ns.sum()
    grand = np.concatenate(groups).mean()
    means = np.array([g.mean() for g in groups])
    ssb = float((ns * (means - grand) ** 2).sum())
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    rho = float("nan") if denom == 0 else (msb - msw) / denom
    est = 0.0 if rho != rho else float(max(0.0, min(0.99, rho)))
    return {"k_vignettes": k, "n_total": int(N), "rate": grand, "n0": n0,
            "estimable": denom != 0,
            "icc_raw": rho, "icc": est,
            "per_vignette_rates": {kk: float(np.mean(v)) for kk, v in by_v.items()}}


def beta_ab(p, icc):
    s = (1.0 - icc) / icc
    return p * s, (1.0 - p) * s


def draw(rng, p, icc, reps):
    if icc <= 1e-9 or p <= 0.0 or p >= 1.0:
        pv = np.full(N_VIGNETTES, p)
    else:
        a, b = beta_ab(p, icc)
        pv = rng.beta(a, b, size=N_VIGNETTES)
    return rng.binomial(reps, pv) / reps


def signflip_p(diffs, rng, n_perm=2000):
    obs = diffs.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size))
    null = (signs * diffs).mean(axis=1)
    return float(((null <= obs).sum() + 1) / (n_perm + 1))


def power(p0, p1, icc, reps, rng, n_sims=N_SIMS):
    """Paired B1_full - B0_none, one-sided (H1: ablation reduces the rate)."""
    hit_perm = hit_t = 0
    for _ in range(n_sims):
        d = draw(rng, p1, icc, reps) - draw(rng, p0, icc, reps)
        if signflip_p(d, rng) <= ALPHA:
            hit_perm += 1
        sd = d.std(ddof=1)
        if sd > 0:
            if d.mean() / (sd / np.sqrt(N_VIGNETTES)) < T_CRIT_ONESIDED_DF19:
                hit_t += 1
        elif d.mean() < 0:
            hit_t += 1
    return {"power_permutation": hit_perm / n_sims, "power_paired_t": hit_t / n_sims}


def main() -> int:
    g0 = gate0()
    if not g0["match"]:
        print("GATE 0 FAILED", g0, file=sys.stderr)
        return 2
    rng = np.random.default_rng(SEED)

    mn = icc_anova(mention_by_vignette())
    dx = icc_anova(diagnosis_by_vignette())
    # DV1 sits at exactly 1.000, so MSB = MSW = 0 and its ICC is NOT ESTIMABLE.
    # Proxy: the ICC of the ES-mention outcome, which is a binary outcome on the
    # SAME cell and the SAME 200 runs -- a tighter proxy than Phase 2 could use.
    dx["icc_proxy_source"] = "DV2 mention ICC (same cell, same runs)"
    dx["icc"] = mn["icc"]
    print(f"DV2 mention   : rate {mn['rate']:.4f}  icc {mn['icc']:.4f} "
          f"(raw {mn['icc_raw']:+.4f}, estimable={mn['estimable']})")
    print(f"DV1 diagnosis : rate {dx['rate']:.4f}  icc NOT ESTIMABLE at ceiling "
          f"-> proxy {dx['icc']:.4f} from DV2 (same cell/runs)")
    assert abs(mn["rate"] - 0.46) < 1e-9, "mention rate must reproduce App. A1's 92/200"

    reps_grid = [5, 7, 10, 12]
    grid = []

    # DV1 -- from ceiling. Exactly as in Phase 2: 1.0 is not a usable simulation
    # parameter, so three conservative baselines are reported.
    for p0 in (0.9975, 0.99, 0.97):
        for D in (0.05, 0.10, 0.15, 0.20):
            for R in reps_grid:
                for icc_s in (dx["icc"], 0.05, 0.15):
                    res = power(p0, max(0.0, p0 - D), icc_s, R, rng)
                    grid.append({"dv": "diagnosis", "p0": p0, "drop": D, "reps": R,
                                 "icc": icc_s, **res})
                    print(f"  [dx] icc={icc_s:.3f} p0={p0:.4f} D={D:.2f} R={R:2d} "
                          f"perm={res['power_permutation']:.3f}")

    # DV2 -- from the real 0.460 baseline, at the real ICC and at inflated ICC.
    for icc_s in (mn["icc"], 0.05, 0.15):
        for D in (0.10, 0.15, 0.20, 0.30):
            for R in reps_grid:
                res = power(mn["rate"], max(0.0, mn["rate"] - D), icc_s, R, rng)
                grid.append({"dv": "mention", "p0": mn["rate"], "drop": D, "reps": R,
                             "icc": icc_s, **res})
                print(f"  [mn] icc={icc_s:.3f} D={D:.2f} R={R:2d} "
                      f"perm={res['power_permutation']:.3f}")

    # Type-I under a true null for both DVs.
    t1 = []
    for name, p0, icc in (("diagnosis", 0.99, dx["icc"]), ("mention", mn["rate"], mn["icc"])):
        for R in reps_grid:
            res = power(p0, p0, icc, R, rng, n_sims=8000)
            t1.append({"dv": name, "p0": p0, "reps": R, "icc": icc, **res})
            print(f"[type-I] {name:9s} R={R:2d} perm={res['power_permutation']:.4f} "
                  f"t={res['power_paired_t']:.4f}")

    out = {"gate0": g0, "source_data_commit": "a715ce4", "cell": CELL,
           "dv1_diagnosis": dx, "dv2_mention": mn,
           "mention_regex": MENTION_RX.pattern,
           "estimator": "paired by vignette, B1_full - B0_none, one-sided; "
                        "primary = sign-flip permutation, secondary = paired t (df=19)",
           "alpha_per_dv": ALPHA, "n_vignettes": N_VIGNETTES,
           "n_sims": N_SIMS, "seed": SEED,
           "grid": grid, "type_i": t1}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
