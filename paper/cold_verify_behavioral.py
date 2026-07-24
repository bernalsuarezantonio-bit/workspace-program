#!/usr/bin/env python3
# Copyright 2026.
"""Study 1 (behavioral) — COLD re-verification of the confirmatory analysis.

Independent re-derivation of H1-H4 from the committed data of the reification-
gradient repo, NOT reusing its analysis code or its library (statsmodels is not
installed here; the logit + cluster-robust sandwich is reimplemented in numpy).
Any number that fails to reproduce the committed RESULTS.md is reported as a
DISCREPANCY and nothing is reconciled.

Gate 0 (git-level, done out of band and recorded in the report):
  tag prereg-v1 -> commit 4b2464f (PLAN.md present) -> data commit 770fa9c
  (7200 raw responses) -> scoring c4a5ce8 (scored_full.jsonl) -> RESULTS 64166cd.

Inputs (committed blobs, extracted read-only; sha256 recorded):
  responses_770fa9c.jsonl  git-blob 44f3407  sha256 d3e5bf67...  (raw run)
  scored_c4a5ce8.jsonl     git-blob a0ca7cf  sha256 fb85367d...  (judge-scored)

Model C (PLAN.md s5 + amendments A2/A4, from phase6/analyze_confirmatory.py):
  DV = diagnosis; formula = diagnosis ~ C(disorder, ref='DN_plausible') * level_c
  with level_c = level_num - 3; inference = logit MLE + cluster-robust SE by
  vignette (60 clusters); one-sided tests; alpha = 0.0125.

Run:  .venv/Scripts/python.exe paper/cold_verify_behavioral.py <scratch_beh_dir>
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LEVELS = ["L1_forum", "L2_coach_blog", "L3_wiki", "L4_preprint", "L5_pseudodsm"]
DISORDERS_REF = "DN_plausible"
OTHERS = ["DN_flagged", "incoherent", "real_anchor"]     # alphabetical, ref removed
ALPHA = 0.05 / 4

# Committed RESULTS.md values (phase6/RESULTS.md @ HEAD) to reproduce.
EXPECTED = {
    "n_scored": 7200, "n_analytic": 7169, "n_excluded": 31,
    "H1": {"coef": -0.0391, "se": 0.0173, "OR": 0.962, "z": -2.257, "p": 0.988},
    "H2": {"coef": +0.4514, "se": 0.0348, "OR": 1.571, "z": +12.964, "p": 1.000},
    "H3": {"coef": +0.0810, "se": 0.0290, "OR": 1.084, "z": +2.793, "p": 0.997},
    "H4": {"coef": -0.1570, "se": 0.0385, "OR": 0.855, "z": -4.075, "p": 0.0001},
    "JT": {"z": -1.180, "p": 0.881},
    "robustness_fraction": 0.00,
    "means": {"mistral-small3.1:24b": {"DN_flagged": 0.554, "incoherent": 0.490,
                                       "DN_plausible": 0.607, "real_anchor": 0.451},
              "qwen2.5:32b": {"DN_flagged": 0.419, "incoherent": 0.218,
                              "DN_plausible": 0.445, "real_anchor": 0.184}},
    "H1_anchor_own_slope": 0.042,
}
TOL = {"coef": 5e-3, "se": 5e-3, "OR": 5e-3, "z": 0.05, "p": 5e-3, "rate": 5e-3}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------ logistic regression (IRLS)

def fit_logit(X, y, iters=100, tol=1e-12):
    n, k = X.shape
    b = np.zeros(k)
    for _ in range(iters):
        eta = X @ b
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1.0 - p)
        WX = X * W[:, None]
        H = X.T @ WX
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b, p


def cluster_robust_cov(X, y, p, groups):
    """Sandwich cov for MLE logit with clustering, matching statsmodels'
    cov_type='cluster' default correction G/(G-1) * (N-1)/(N-K)."""
    n, k = X.shape
    W = p * (1.0 - p)
    bread = np.linalg.inv(X.T @ (X * W[:, None]))
    u = y - p
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        idx = groups == g
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    corr = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    return bread @ meat @ bread * corr


def phi_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def phi_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def one_sided(b, se, direction):
    z = b / se
    p = phi_sf(z) if direction == "+" else phi_cdf(z)
    return {"coef": b, "se": se, "z": z, "p_one_sided": p, "OR": math.exp(b),
            "OR_CI95": [math.exp(b - 1.96 * se), math.exp(b + 1.96 * se)],
            "significant": bool(p < ALPHA)}


def jonckheere(groups):
    k = len(groups)
    ns = [len(g) for g in groups]
    N = sum(ns)
    U = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            a = groups[i][:, None]
            bb = groups[j][None, :]
            U += float((a < bb).sum() + 0.5 * (a == bb).sum())
    EU = (N ** 2 - sum(n ** 2 for n in ns)) / 4.0
    allv = np.concatenate(groups)
    _, tc = np.unique(allv, return_counts=True)
    T = sum(t * (t - 1) * (2 * t + 5) for t in tc)
    varU = (N * (N - 1) * (2 * N + 5)
            - sum(n * (n - 1) * (2 * n + 5) for n in ns) - T) / 72.0
    z = (U - EU) / math.sqrt(varU)
    return {"U": U, "z": z, "p_one_sided": phi_sf(z), "significant": bool(phi_sf(z) < ALPHA)}


def cmp(name, got, exp, kind, out):
    ok = abs(got - exp) <= TOL[kind]
    out.append({"name": name, "got": got, "expected": exp, "kind": kind,
                "abs_diff": abs(got - exp), "tol": TOL[kind], "MATCH": bool(ok)})
    return ok


def main() -> int:
    beh = Path(sys.argv[1])
    resp_p = beh / "responses_770fa9c.jsonl"
    scored_p = beh / "scored_c4a5ce8.jsonl"

    prov = {"responses_sha256": sha256(resp_p), "scored_sha256": sha256(scored_p),
            "responses_blob": "44f3407fd09892a54e2a6029291f919d5f3aefa4",
            "scored_blob": "a0ca7cf00fbe5be35c6e54dfb8c1e0d45aea500d"}

    # ---- completeness on the DATA commit (raw responses) ----
    raw = load(resp_p)
    # the file ends with a run-end footer record (no "model" key); split it off
    footers = [r for r in raw if "model" not in r]
    resp = [r for r in raw if "model" in r]
    cell_key = lambda r: (r["model"], r["disorder"], r["level"], r["vignette"], r["rep"])
    keys = [cell_key(r) for r in resp]
    comp = {"n_lines_total": len(raw), "n_responses": len(resp),
            "footer": footers[0] if footers else None,
            "footer_expected_cells": footers[0].get("expected_cells") if footers else None,
            "n_duplicate_keys": len(keys) - len(set(keys))}
    by_cond_level = Counter((r["disorder"], r["level"]) for r in resp)
    comp["n_condition_level_cells"] = len(by_cond_level)
    comp["cond_level_counts"] = {f"{d}|{l}": c for (d, l), c in sorted(by_cond_level.items())}
    comp["cond_level_min"] = min(by_cond_level.values())
    comp["cond_level_max"] = max(by_cond_level.values())

    # ---- analytic dataset (scored) ----
    scored = load(scored_p)
    df = [r for r in scored if not r.get("error")]
    n_excluded = len(scored) - len(df)

    lv = {l: i + 1 for i, l in enumerate(LEVELS)}
    y = np.array([int(r["diagnosis"]) for r in df], dtype=float)
    level_c = np.array([lv[r["level"]] - 3.0 for r in df], dtype=float)
    dis = [r["disorder"] for r in df]
    vign = np.array([r["vignette"] for r in df])
    fam = [r["model"].split("/")[-1] for r in df]

    # design: intercept + 3 disorder dummies (ref DN_plausible) + level_c + 3 interactions
    cols = {"Intercept": np.ones(len(df))}
    for o in OTHERS:
        cols[o] = np.array([1.0 if d == o else 0.0 for d in dis])
    cols["level_c"] = level_c
    for o in OTHERS:
        cols[f"{o}:level_c"] = cols[o] * level_c
    names = ["Intercept", *OTHERS, "level_c", *[f"{o}:level_c" for o in OTHERS]]
    X = np.column_stack([cols[n] for n in names])

    b, p = fit_logit(X, y)
    cov = cluster_robust_cov(X, y, p, vign)
    se = np.sqrt(np.diag(cov))
    par = dict(zip(names, b))
    separ = dict(zip(names, se))

    H1 = one_sided(par["level_c"], separ["level_c"], "+")
    H2 = one_sided(par["incoherent:level_c"], separ["incoherent:level_c"], "-")
    H3 = one_sided(par["real_anchor:level_c"], separ["real_anchor:level_c"], "-")
    H4 = one_sided(par["DN_flagged"], separ["DN_flagged"], "-")
    anchor_own = par["level_c"] + par["real_anchor:level_c"]

    # JT on DN_plausible
    dnp = [r for r in df if r["disorder"] == "DN_plausible"]
    jt_groups = [np.array([int(r["diagnosis"]) for r in dnp if lv[r["level"]] == l])
                 for l in range(1, 6)]
    JT = jonckheere(jt_groups)

    # robustness ordering per family
    fam_means, order_ok = {}, {}
    for f in sorted(set(fam)):
        m = {}
        for d in ["DN_flagged", "incoherent", "DN_plausible", "real_anchor"]:
            vals = [int(r["diagnosis"]) for r, ff in zip(df, fam)
                    if ff == f and r["disorder"] == d]
            m[d] = float(np.mean(vals))
        fam_means[f] = m
        order_ok[f] = bool(m["DN_flagged"] < m["incoherent"] < m["DN_plausible"] < m["real_anchor"])
    frac = float(np.mean([order_ok[f] for f in order_ok]))

    # ---- compare against committed RESULTS.md ----
    checks = []
    cmp("n_scored", len(scored), EXPECTED["n_scored"], "coef", checks)
    cmp("n_analytic", len(df), EXPECTED["n_analytic"], "coef", checks)
    cmp("n_excluded", n_excluded, EXPECTED["n_excluded"], "coef", checks)
    for h, res in (("H1", H1), ("H2", H2), ("H3", H3), ("H4", H4)):
        e = EXPECTED[h]
        cmp(f"{h}.coef", res["coef"], e["coef"], "coef", checks)
        cmp(f"{h}.se", res["se"], e["se"], "se", checks)
        cmp(f"{h}.OR", res["OR"], e["OR"], "OR", checks)
        cmp(f"{h}.z", res["z"], e["z"], "z", checks)
        cmp(f"{h}.p", res["p_one_sided"], e["p"], "p", checks)
    cmp("JT.z", JT["z"], EXPECTED["JT"]["z"], "z", checks)
    cmp("JT.p", JT["p_one_sided"], EXPECTED["JT"]["p"], "p", checks)
    cmp("robustness_fraction", frac, EXPECTED["robustness_fraction"], "rate", checks)
    cmp("H1_anchor_own_slope", anchor_own, EXPECTED["H1_anchor_own_slope"], "coef", checks)
    for f, mm in EXPECTED["means"].items():
        for d, ev in mm.items():
            cmp(f"mean[{f}][{d}]", fam_means[f][d], ev, "rate", checks)

    all_match = all(c["MATCH"] for c in checks)
    n_fail = sum(1 for c in checks if not c["MATCH"])

    out = {
        "stage": "Study 1 cold re-verification",
        "provenance": prov,
        "gate0_git": {
            "tag": "prereg-v1", "tag_commit": "4b2464f", "data_commit": "770fa9c",
            "scoring_commit": "c4a5ce8", "results_commit": "64166cd",
            "note": "chain and ancestry verified out of band with git merge-base"},
        "completeness": comp,
        "analytic": {"n_scored": len(scored), "n_analytic": len(df),
                     "n_excluded_malformed": n_excluded},
        "rederived": {
            "H1": H1, "H2": H2, "H3": H3, "H4": H4,
            "H1_jonckheere_terpstra": JT,
            "H3_anchor_own_slope_logodds": anchor_own,
            "robustness": {"by_family_means": fam_means, "order_ok": order_ok,
                           "fraction_preserving": frac},
            "all_params": par, "all_se": separ},
        "comparison_vs_RESULTS_md": checks,
        "n_checks": len(checks), "n_mismatch": n_fail,
        "ALL_MATCH": bool(all_match),
    }
    outp = Path(__file__).resolve().parent / "cold_verification_behavioral.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"completeness: n={comp['n_responses']} dup={comp['n_duplicate_keys']} "
          f"cells={comp['n_condition_level_cells']} "
          f"cond-level n in [{comp['cond_level_min']},{comp['cond_level_max']}]")
    print(f"analytic: scored={len(scored)} analytic={len(df)} excluded={n_excluded}")
    for h, res in (("H1", H1), ("H2", H2), ("H3", H3), ("H4", H4)):
        print(f"  {h}: coef={res['coef']:+.4f} OR={res['OR']:.3f} z={res['z']:+.3f} "
              f"p={res['p_one_sided']:.4f} sig={res['significant']}")
    print(f"  JT: z={JT['z']:+.3f} p={JT['p_one_sided']:.4f}")
    print(f"  robustness fraction={frac:.2f}")
    print(f"\n{len(checks)} checks, {n_fail} mismatch -> ALL_MATCH={all_match}")
    if not all_match:
        print("DISCREPANCIES (not reconciled):")
        for c in checks:
            if not c["MATCH"]:
                print(f"  {c['name']}: got {c['got']} vs expected {c['expected']} "
                      f"(diff {c['abs_diff']:.4g} > tol {c['tol']})")
    print("wrote", outp.name)
    return 0 if all_match else 4


if __name__ == "__main__":
    raise SystemExit(main())
