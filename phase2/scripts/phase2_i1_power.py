#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2 / Stage I1 power analysis — activation-addition dose-response on P(diagnosis).

NO GPU. NO new data. Reads only the committed Phase 1 judge outcomes (data commit
`a715ce4`) to estimate the *clustering* of the binary DV by vignette, then runs a
Monte-Carlo power study for the I1 five-arm design under the registered estimators.

Two things are estimated from real data, nothing else:
  1. The observed baseline diagnosis rate in the I1 base cell (C1_DN_flagged_L1 = 200/200).
  2. The between-vignette intraclass correlation (ICC) of the binary judge outcome,
     estimated from the ONLY non-ceiling confirmatory cell (C2_incoherent_L1,
     56/199), because a ceiling cell carries no information about clustering.

Everything downstream (assumed drops, alpha spacing) is a design assumption and is
labelled as such. Run as a file:  python phase2/scripts/phase2_i1_power.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
JUDGE = ROOT / "phase1" / "data" / "judge_full.jsonl"
MANIFEST = ROOT / "phase1" / "data" / "run_manifest_full.jsonl"
COMPLETE = ROOT / "phase1" / "data" / "completeness_report.json"
OUT = ROOT / "phase2" / "data" / "phase2_i1_power.json"

DATA_DIGEST = "dc522361096bae30377ecf05d37142cfcb3f52fbb6349c77825bea455f0fb8f1"
BASE_CELL = "C1_DN_flagged_L1"          # the I1 base cell (flagged x high x L1)
ICC_CELL = "C2_incoherent_L1"           # only non-ceiling cell -> clustering estimate

N_VIGNETTES = 20
N_SIMS = 20000
SEED = 20260722

# One-sided t critical value, df = 19, alpha = 0.025 (the Phase 1 house value,
# PREREG_PHASE1 s5 / RESULTS_PHASE1 C2: |t| > 2.093). Sign flipped: H1 is a DROP.
T_CRIT_ONESIDED_DF19 = -2.093


# ---------------------------------------------------------------- Gate 0

def gate0() -> dict:
    """Re-assert the Phase 1 data digest before using any of its numbers."""
    h = hashlib.sha256()
    for p in (MANIFEST, JUDGE, COMPLETE):
        h.update(p.read_bytes())
    got = h.hexdigest()
    return {"data_digest_recomputed": got, "data_digest_expected": DATA_DIGEST,
            "match": got == DATA_DIGEST}


# ---------------------------------------------------------------- real inputs

def load_cell(cell: str) -> dict[str, list[int]]:
    by_v: dict[str, list[int]] = defaultdict(list)
    with JUDGE.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["cell"] != cell:
                continue
            d = r.get("diagnosis")
            if isinstance(d, int):          # judge parse errors excluded (prereg P1 s6b)
                by_v[r["vignette"]].append(d)
    return dict(by_v)


def icc_anova(by_v: dict[str, list[int]]) -> dict:
    """Moment (ANOVA) ICC for binary data clustered in equal-ish sized groups.

    rho = (MSB - MSW) / (MSB + (n0 - 1) * MSW), truncated at 0.
    """
    groups = [np.asarray(v, dtype=float) for v in by_v.values()]
    k = len(groups)
    ns = np.array([g.size for g in groups], dtype=float)
    N = ns.sum()
    grand = np.concatenate(groups).mean()
    means = np.array([g.mean() for g in groups])
    ssb = float((ns * (means - grand) ** 2).sum())
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    rho = (msb - msw) / (msb + (n0 - 1) * msw)
    return {"cell": None, "k_vignettes": k, "n_total": int(N), "rate": grand,
            "MSB": msb, "MSW": msw, "n0": n0, "icc_raw": rho,
            "icc": float(max(0.0, min(0.99, rho))),
            "per_vignette_rates": {kk: float(np.mean(v)) for kk, v in by_v.items()}}


# ---------------------------------------------------------------- simulation

def beta_ab(p: float, icc: float) -> tuple[float, float]:
    """Beta(a,b) with mean p and intraclass correlation icc (= 1/(a+b+1))."""
    if icc <= 1e-9:
        return (float("inf"), float("inf"))
    s = (1.0 - icc) / icc          # a + b
    return (p * s, (1.0 - p) * s)


def draw_arm(rng, p: float, icc: float, n_vign: int, reps: int) -> np.ndarray:
    """Return per-vignette success COUNTS for one arm (vignette-clustered binomial)."""
    if icc <= 1e-9 or p <= 0.0 or p >= 1.0:
        pv = np.full(n_vign, p)
    else:
        a, b = beta_ab(p, icc)
        pv = rng.beta(a, b, size=n_vign)
    return rng.binomial(reps, pv)


def signflip_p(diffs: np.ndarray, rng, n_perm: int = 4000, less: bool = True) -> float:
    """One-sided exact-ish sign-flip permutation p-value on the mean of `diffs`."""
    obs = diffs.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size))
    null = (signs * diffs).mean(axis=1)
    if less:
        return float(((null <= obs).sum() + 1) / (n_perm + 1))
    return float(((null >= obs).sum() + 1) / (n_perm + 1))


def sim_power(p0: float, p_doses: list[float], icc: float, reps: int,
              alpha: float, rng, n_sims: int = N_SIMS,
              n_perm: int = 2000) -> dict:
    """Power of the registered PRIMARY trend estimator + the pairwise top-dose test.

    PRIMARY: per-vignette OLS slope of run-level diagnosis on dose score
    (scores 0,1,2,3 over [baseline, a1, a2, a3]); one-sided sign-flip permutation
    test over the 20 vignette slopes (H1: mean slope < 0).

    SECONDARY (reported): paired one-sided t over vignettes, top dose vs baseline.
    """
    scores = np.arange(len(p_doses) + 1, dtype=float)
    sc = scores - scores.mean()
    denom = (sc ** 2).sum()
    ps = [p0, *p_doses]

    hit_trend = 0
    hit_pair = 0
    for _ in range(n_sims):
        rates = np.stack([draw_arm(rng, p, icc, N_VIGNETTES, reps) / reps for p in ps])
        # per-vignette slope
        slopes = (sc[:, None] * rates).sum(axis=0) / denom
        if signflip_p(slopes, rng, n_perm=n_perm, less=True) <= alpha:
            hit_trend += 1
        # pairwise top dose vs baseline, paired t one-sided (H1: dose < base)
        d = rates[-1] - rates[0]
        sd = d.std(ddof=1)
        if sd > 0:
            t = d.mean() / (sd / np.sqrt(N_VIGNETTES))
            if t < T_CRIT_ONESIDED_DF19:
                hit_pair += 1
        elif d.mean() < 0:
            hit_pair += 1
    return {"power_trend_primary": hit_trend / n_sims,
            "power_pairwise_topdose": hit_pair / n_sims}


def main() -> int:
    g0 = gate0()
    if not g0["match"]:
        print("GATE 0 FAILED:", g0, file=sys.stderr)
        return 2

    base = load_cell(BASE_CELL)
    base_n = sum(len(v) for v in base.values())
    base_k = sum(sum(v) for v in base.values())
    icc_src = load_cell(ICC_CELL)
    icc_est = icc_anova(icc_src)
    icc_est["cell"] = ICC_CELL

    icc = icc_est["icc"]
    rng = np.random.default_rng(SEED)

    # Baseline assumption: observed 200/200. A ceiling of exactly 1.0 is not a usable
    # simulation parameter (any single failure is then "significant"), so power is
    # reported at three conservative baselines. Jeffreys posterior mean for 200/200
    # is 0.9975; 0.99 and 0.97 are the conservative sensitivity rungs.
    baselines = [0.9975, 0.99, 0.97]

    # Design assumption grid: the top-dose drop D (P(diagnosis) at a3 = p0 - D),
    # with the two lower doses on a linear ramp (0.25 D, 0.55 D) -- a plausible
    # monotone dose-response shape. NOT an estimate; a sensitivity grid.
    drops = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    reps_grid = [5, 7, 10]
    alpha = 0.025           # Bonferroni /2 (trend test + control-contrast test)

    grid = []
    for p0 in baselines:
        for D in drops:
            p_doses = [max(0.0, p0 - f * D) for f in (0.25, 0.55, 1.0)]
            for R in reps_grid:
                res = sim_power(p0, p_doses, icc, R, alpha, rng)
                grid.append({"p0": p0, "top_drop": D, "p_doses": p_doses,
                             "reps": R, "n_per_arm": R * N_VIGNETTES,
                             "n_total_runs": 5 * R * N_VIGNETTES, **res})
                print(f"p0={p0:.4f} D={D:.2f} R={R:2d} "
                      f"trend={res['power_trend_primary']:.3f} "
                      f"pair={res['power_pairwise_topdose']:.3f}")

    # Type-I check: no true effect anywhere (all arms == p0).
    t1 = []
    for p0 in baselines:
        for R in reps_grid:
            res = sim_power(p0, [p0, p0, p0], icc, R, alpha, rng, n_sims=8000)
            t1.append({"p0": p0, "reps": R, **res})
            print(f"[type-I] p0={p0:.4f} R={R:2d} trend={res['power_trend_primary']:.4f} "
                  f"pair={res['power_pairwise_topdose']:.4f}")

    # ICC sensitivity. The moment estimate above is ~0, but it is estimated from
    # k=20 clusters of n0~10 binary draws, so it cannot distinguish "no clustering"
    # from "modest clustering". Re-run the decision-relevant slice at inflated ICC.
    sens = []
    for icc_s in (0.0, 0.05, 0.15):
        for D in (0.05, 0.10, 0.15):
            for R in reps_grid:
                p0 = 0.99
                p_doses = [max(0.0, p0 - f * D) for f in (0.25, 0.55, 1.0)]
                res = sim_power(p0, p_doses, icc_s, R, alpha, rng, n_sims=8000)
                sens.append({"icc": icc_s, "p0": p0, "top_drop": D, "reps": R, **res})
                print(f"[icc-sens] icc={icc_s:.2f} D={D:.2f} R={R:2d} "
                      f"trend={res['power_trend_primary']:.3f} "
                      f"pair={res['power_pairwise_topdose']:.3f}")

    out = {
        "gate0": g0,
        "icc_sensitivity": sens,
        "source": {"judge": str(JUDGE.relative_to(ROOT)), "data_commit": "a715ce4"},
        "base_cell": {"cell": BASE_CELL, "n": base_n, "k_diagnosis": base_k,
                      "rate": base_k / base_n,
                      "jeffreys_posterior_mean": (base_k + 0.5) / (base_n + 1)},
        "icc_estimate": icc_est,
        "assumptions": {
            "baselines_simulated": baselines,
            "dose_ramp_fractions": [0.25, 0.55, 1.0],
            "alpha_per_test": alpha,
            "n_vignettes": N_VIGNETTES,
            "n_sims": N_SIMS,
            "seed": SEED,
            "primary_estimator": "per-vignette OLS slope on dose score; one-sided "
                                 "sign-flip permutation over 20 vignette slopes",
            "secondary_estimator": "paired one-sided t, top dose vs baseline, df=19",
        },
        "grid": grid,
        "type_i": t1,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
