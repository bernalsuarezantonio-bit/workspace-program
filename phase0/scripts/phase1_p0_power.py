# Copyright 2026 — Phase 0/1 delegate. Stage P0 power analysis (NO GPU, condition-free).
"""Estimates between-rep variance of the preregistered Set A / Set F loadings from
the 20-rep condition-free nightly calibration data (v12), then Monte-Carlo powers
the two Phase-1 contrasts under the preregistered aggregation. This uses ONLY
condition-free technical-calibration data (explicitly legitimate per PI). No
conditions, no comparisons of conditions, no diagnostic-token counting.

Aggregation (preregistered): loading of set S in a run = for each layer L in the
primary band 17-26, mean over GENERATION positions (R4) of the summed readout
weights of S's operative tokens (folded match per R3, SURVIVORS only — echoes
barred), then mean over the band. Language-tagged breakdown reported.

Power model: within-item (paired-by-vignette) contrast over the 20 'high'
vignettes; R reps averaged per (vignette,cell). Per-vignette paired difference
~ N(effect, 2*sigma_rep^2 / R) under the (flagged) assumption of negligible
vignette x cell interaction. Paired t-test, n=20, df=19, alpha=0.025
(Bonferroni/2). C2 one-sided (directional), C1 two-sided (discovery). Power is
scale-free (depends on standardized effect delta = effect/sigma_rep and R), so
the grid holds for both sets; sigma_rep is reported for non-degeneracy and raw
anchoring. LIMITATION (flagged for freeze): vignette x cell interaction variance
is NOT estimable from one calibration vignette -> the R from the rule is a FLOOR.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[2]
NIGHTLY = REPO / "phase0" / "data" / "pilot_readouts" / "nightly"
SCREEN = json.loads((REPO / "phase0" / "data" / "phase1_seal_screening_A1.json").read_text(encoding="utf-8"))

BAND = list(range(17, 27))        # primary band layers 17-26 inclusive
ALPHA = 0.025
TCRIT_1SIDED = 2.093              # t_{0.975, 19}
TCRIT_2SIDED = 2.433             # t_{0.9875, 19}
N_VIGNETTES = 20
RNG_SEED = 0
DELTAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
RS = [3, 5, 8, 10, 15]
GPU_SEC_PER_RUN = 6.3
CELLS = 4


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def operative(setname):
    """folded -> lang for SURVIVORS only."""
    out = {}
    for c in SCREEN[setname]["concepts"]:
        for r in c["realized"]:
            if r["status"] == "SURVIVES":
                out[r["folded"]] = r["langs"]
    return out


def loading_for_rep(rep_rows, op_folded):
    """Return (total_loading, en_loading, es_loading) for one rep under the
    preregistered aggregation. n_cells = |BAND| * n_gen_positions."""
    band = set(BAND)
    gen_positions = set()
    tot = en = es = 0.0
    for row in rep_rows:
        if row["kind"] != "lens" or row["layer"] not in band or row["segment"] != "generation":
            continue
        gen_positions.add(row["position"])
        for e in row["topk"]:
            f = fold(e["token"])
            lang = op_folded.get(f)
            if lang is not None:
                w = e["weight"]
                tot += w
                if "EN" in lang:
                    en += w
                if "ES" in lang:
                    es += w
    n_cells = len(BAND) * max(1, len(gen_positions))
    return tot / n_cells, en / n_cells, es / n_cells


def power_mc(delta, R, two_sided, n_sim=40000, rng=None):
    """Scale-free MC power: 20 paired diffs ~ N(delta, 2/R); one-sample t vs 0."""
    rng = rng or np.random.default_rng(RNG_SEED)
    sd = np.sqrt(2.0 / R)
    d = rng.normal(delta, sd, size=(n_sim, N_VIGNETTES))
    m = d.mean(axis=1)
    s = d.std(axis=1, ddof=1)
    t = m / (s / np.sqrt(N_VIGNETTES))
    if two_sided:
        return float(np.mean(np.abs(t) > TCRIT_2SIDED))
    return float(np.mean(t > TCRIT_1SIDED))


def main() -> int:
    opA, opF = operative("A_generic_nosological"), operative("F_disclosure_fictional")
    print(f"operative Set A: {len(opA)} folded keys ; Set F: {len(opF)} folded keys")

    loadA, loadF, langA = [], [], []
    for i in range(1, 21):
        rows = json.loads((NIGHTLY / f"v12_rep{i:02d}.json").read_text(encoding="utf-8"))["rows"]
        a, aen, aes = loading_for_rep(rows, opA)
        f, fen, fes = loading_for_rep(rows, opF)
        loadA.append(a); loadF.append(f); langA.append((aen, aes))

    loadA, loadF = np.array(loadA), np.array(loadF)
    def desc(x): return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                         "min": float(x.min()), "max": float(x.max()),
                         "cv": float(x.std(ddof=1) / x.mean()) if x.mean() else None}
    A_desc, F_desc = desc(loadA), desc(loadF)
    aen = np.array([x[0] for x in langA]); aes = np.array([x[1] for x in langA])

    print("\n=== between-rep loading (condition-free v12, n=20) ===")
    print(f"Set A: mean={A_desc['mean']:.4f} sd={A_desc['sd']:.4f} cv={A_desc['cv']} "
          f"[EN mean {aen.mean():.4f} | ES mean {aes.mean():.4f}]")
    print(f"Set F: mean={F_desc['mean']:.4f} sd={F_desc['sd']:.4f} cv={F_desc['cv']}")

    print("\n=== power grid (n=20 paired vignettes, alpha=0.025) ===")
    rng = np.random.default_rng(RNG_SEED)
    grid = {"C1_two_sided": {}, "C2_one_sided": {}}
    print("delta \\ R      " + "   ".join(f"R={R}" for R in RS))
    for label, two in [("C1_two_sided", True), ("C2_one_sided", False)]:
        print(f"-- {label} --")
        for delta in DELTAS:
            row = {}
            for R in RS:
                row[R] = round(power_mc(delta, R, two, rng=rng), 3)
            grid[label][delta] = row
            print(f"  d={delta:<4}   " + "  ".join(f"{row[R]:.3f}" for R in RS))

    # pre-fixed rule: smallest R with power>=0.80 at medium (delta=0.5) for BOTH, within 3h
    def smallest_R(delta=0.5):
        for R in RS:
            gpu_min = CELLS * N_VIGNETTES * R * GPU_SEC_PER_RUN / 60
            if grid["C1_two_sided"][delta][R] >= 0.80 and grid["C2_one_sided"][delta][R] >= 0.80 \
               and (gpu_min + 30) <= 180:  # +judge, <=3h
                return R, gpu_min
        return None, None
    R_chosen, gpu_min = smallest_R(0.5)
    print(f"\n=== RULE: smallest R with power>=0.80 at delta=0.5 (medium) BOTH contrasts, "
          f"<=3h GPU ===\n  R = {R_chosen}  (~{gpu_min:.0f} min lens + ~30 min judge)")

    out = {
        "aggregation_band": BAND, "alpha": ALPHA, "n_vignettes": N_VIGNETTES,
        "rng_seed": RNG_SEED, "tcrit": {"one_sided": TCRIT_1SIDED, "two_sided": TCRIT_2SIDED},
        "loading_A": A_desc, "loading_F": F_desc,
        "loading_A_lang_mean": {"EN": float(aen.mean()), "ES": float(aes.mean())},
        "per_rep_loading_A": loadA.tolist(), "per_rep_loading_F": loadF.tolist(),
        "power_grid": {k: {str(d): v for d, v in g.items()} for k, g in grid.items()},
        "R_from_rule": R_chosen, "R_gpu_minutes_lens": gpu_min,
        "gpu_budget_note": {f"R={R}": round(CELLS*N_VIGNETTES*R*GPU_SEC_PER_RUN/60, 1) for R in RS},
        "limitation": "vignette x cell interaction variance not estimable from one "
                      "calibration vignette; R_from_rule is a FLOOR. F loading on the "
                      "condition-free vignette is expected near-floor (no disclosure).",
    }
    (REPO / "phase0" / "data" / "phase1_p0_power.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {REPO / 'phase0' / 'data' / 'phase1_p0_power.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
