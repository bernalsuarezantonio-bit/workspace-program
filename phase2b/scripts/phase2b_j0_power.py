#!/usr/bin/env python3
# Copyright 2026.
"""Phase 2b / Stage J0 power analysis — projection ablation, 3 arms, 2 DVs.

NO GPU. NO new data. Both DVs are anchored on REAL Phase 1 rates from the
committed data commit a715ce4 (Gate 0 re-asserted in-process):

  DV1 diagnosis rate      200/200 = 1.000 in C1_DN_flagged_L1 (at ceiling)
  DV2 ES textual mention  92/200 = 0.460, recomputed per vignette from the
                          committed readouts with the REGISTERED regex
                          (inventad|estudio|no reconocid|fictici) -- the same
                          estimator that produced RESULTS_PHASE1 App. A1's
                          split, whose 92/108 counts this reproduces exactly.

Arms (PI, 2026-07-22): B0_none / B1_full / B3_rand. B2_half is REMOVED; the
random-direction projection B3_rand is the specificity control.

Two contrasts per DV, both paired by vignette and one-sided:
  intervention  B1 - B0   "does removing this direction change the channel?"
  specificity   B1 - B3   "is the change specific to the F direction?"
=> 4 registered tests, Bonferroni 0.05/4 = 0.0125. Power is also reported at
0.025 so the PI can weigh a 2-test structure at freeze.

The specificity contrast needs an assumption about how much of the effect a
random direction reproduces: gamma in {0, 0.25, 0.5}, i.e. p(B3) = p0 - gamma*D.
gamma = 0 is a fully specific effect. Labelled as an assumption, not an estimate.

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
N_SIMS = 8000
N_PERM = 1000
SEED = 20260722
T_CRIT_ONESIDED_DF19 = -2.093
ALPHAS = (0.0125, 0.025)          # 4-test and 2-test Bonferroni structures
N_ARMS = 3
SEC_PER_RUN = 12.3                # Phase 1 measured: 7.41 gen + 3.37 judge + ~1.5 readout


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
    return {"k_vignettes": k, "n_total": int(N), "rate": float(grand), "n0": float(n0),
            "estimable": bool(denom != 0),
            "icc_raw": (None if rho != rho else float(rho)),
            "icc": (0.0 if rho != rho else float(max(0.0, min(0.99, rho)))),
            "per_vignette_rates": {kk: float(np.mean(v)) for kk, v in by_v.items()}}


def draw(rng, p, icc, reps, n_sims):
    """[n_sims, N_VIGNETTES] arm rates, vignette-clustered."""
    if icc <= 1e-9 or p <= 0.0 or p >= 1.0:
        pv = np.full((n_sims, N_VIGNETTES), p)
    else:
        s = (1.0 - icc) / icc
        pv = rng.beta(p * s, (1.0 - p) * s, size=(n_sims, N_VIGNETTES))
    return rng.binomial(reps, pv) / reps


def power_from_diffs(diffs, rng, alphas):
    """Vectorized one-sided sign-flip permutation + paired t over [n_sims, 20]."""
    obs = diffs.mean(axis=1)                                   # [n_sims]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_PERM, N_VIGNETTES))
    null = (signs @ diffs.T) / N_VIGNETTES                     # [N_PERM, n_sims]
    p_perm = ((null <= obs).sum(axis=0) + 1) / (N_PERM + 1)
    sd = diffs.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, obs / (sd / np.sqrt(N_VIGNETTES)), np.where(obs < 0, -np.inf, 0.0))
    out = {"power_paired_t": float((t < T_CRIT_ONESIDED_DF19).mean())}
    for a in alphas:
        out[f"power_permutation_a{a}"] = float((p_perm <= a).mean())
    return out


def main() -> int:
    g0 = gate0()
    if not g0["match"]:
        print("GATE 0 FAILED", g0, file=sys.stderr)
        return 2
    rng = np.random.default_rng(SEED)

    mn = icc_anova(mention_by_vignette())
    dx = icc_anova(diagnosis_by_vignette())
    # DV1 sits at exactly 1.000, so MSB = MSW = 0 and its ICC is NOT ESTIMABLE.
    # Proxy: the ICC of the ES-mention outcome -- a binary outcome on the SAME
    # cell and the SAME 200 runs, a tighter proxy than Phase 2 could use.
    dx["icc_proxy_source"] = "DV2 mention ICC (same cell, same 200 runs)"
    dx["icc_used"] = mn["icc"]
    assert abs(mn["rate"] - 0.46) < 1e-9, "mention rate must reproduce App. A1's 92/200"
    print(f"DV2 mention   rate {mn['rate']:.4f}  icc {mn['icc']:.4f} "
          f"(raw {mn['icc_raw']:+.4f}, estimable={mn['estimable']})")
    print(f"DV1 diagnosis rate {dx['rate']:.4f}  icc NOT ESTIMABLE at ceiling "
          f"-> proxy {dx['icc_used']:.4f}")

    reps_grid = [5, 7, 10, 12]
    iccs = [0.0, 0.05, 0.15]
    gammas = [0.0, 0.25, 0.5]        # fraction of the effect a random direction reproduces
    grid = []

    def run_cell(dv, p0, D, icc, R, gamma):
        pB0 = draw(rng, p0, icc, R, N_SIMS)
        pB1 = draw(rng, max(0.0, p0 - D), icc, R, N_SIMS)
        pB3 = draw(rng, max(0.0, p0 - gamma * D), icc, R, N_SIMS)
        inter = power_from_diffs(pB1 - pB0, rng, ALPHAS)
        spec = power_from_diffs(pB1 - pB3, rng, ALPHAS)
        row = {"dv": dv, "p0": p0, "drop": D, "icc": icc, "reps": R, "gamma": gamma,
               "n_per_arm": R * N_VIGNETTES, "n_total_runs": N_ARMS * R * N_VIGNETTES,
               "intervention": inter, "specificity": spec}
        grid.append(row)
        return row

    print("\n--- DV1 diagnosis (from ceiling) ---")
    for p0 in (0.9975, 0.99, 0.97):
        for D in (0.05, 0.10, 0.15, 0.20):
            for icc in iccs:
                for R in reps_grid:
                    for gm in gammas:
                        r = run_cell("diagnosis", p0, D, icc, R, gm)
                        if icc == 0.05 and gm == 0.0 and p0 == 0.99:
                            print(f"  p0={p0} D={D:.2f} icc={icc} R={R:2d} "
                                  f"inter={r['intervention']['power_permutation_a0.0125']:.3f} "
                                  f"spec={r['specificity']['power_permutation_a0.0125']:.3f}")

    print("\n--- DV2 mention (real baseline 0.460) ---")
    for D in (0.10, 0.15, 0.20, 0.30):
        for icc in iccs:
            for R in reps_grid:
                for gm in gammas:
                    r = run_cell("mention", mn["rate"], D, icc, R, gm)
                    if icc == 0.05 and gm == 0.0:
                        print(f"  D={D:.2f} icc={icc} R={R:2d} "
                              f"inter={r['intervention']['power_permutation_a0.0125']:.3f} "
                              f"spec={r['specificity']['power_permutation_a0.0125']:.3f}")

    # Type-I: no true effect anywhere (all arms at p0).
    t1 = []
    for name, p0, icc in (("diagnosis", 0.99, dx["icc_used"]), ("mention", mn["rate"], mn["icc"])):
        for R in reps_grid:
            a = draw(rng, p0, icc, R, N_SIMS)
            b = draw(rng, p0, icc, R, N_SIMS)
            res = power_from_diffs(a - b, rng, ALPHAS)
            t1.append({"dv": name, "p0": p0, "icc": icc, "reps": R, **res})
            print(f"[type-I] {name:9s} R={R:2d} "
                  f"a0.0125={res['power_permutation_a0.0125']:.4f} "
                  f"a0.025={res['power_permutation_a0.025']:.4f}")

    budget = {str(R): {"n_total_runs": N_ARMS * R * N_VIGNETTES,
                       "est_hours": N_ARMS * R * N_VIGNETTES * SEC_PER_RUN / 3600.0}
              for R in reps_grid}
    print("\nbudget (3 arms x 20 vignettes x R, at 12.3 s/run):")
    for R in reps_grid:
        print(f"  R={R:2d}  {budget[str(R)]['n_total_runs']:4d} runs  "
              f"{budget[str(R)]['est_hours']:.2f} h")

    out = {"gate0": g0, "source_data_commit": "a715ce4", "cell": CELL,
           "arms": ["B0_none", "B1_full", "B3_rand"],
           "contrasts": {"intervention": "B1_full - B0_none",
                         "specificity": "B1_full - B3_rand"},
           "dv1_diagnosis": dx, "dv2_mention": mn,
           "mention_regex": MENTION_RX.pattern,
           "estimator": "paired by vignette, one-sided; primary = sign-flip permutation, "
                        "secondary = paired t (df=19)",
           "alphas": list(ALPHAS), "gammas_assumed": gammas,
           "gamma_meaning": "fraction of the intervention effect that a random-direction "
                            "projection reproduces; an ASSUMPTION, not an estimate",
           "n_vignettes": N_VIGNETTES, "n_sims": N_SIMS, "n_perm": N_PERM, "seed": SEED,
           "sec_per_run": SEC_PER_RUN, "budget": budget,
           "grid": grid, "type_i": t1}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
