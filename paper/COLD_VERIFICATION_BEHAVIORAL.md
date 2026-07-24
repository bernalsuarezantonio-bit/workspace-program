# Cold verification — Study 1 (behavioral, reification-gradient)

**Recorded:** 2026-07-24, paper stage W0. **Independent cold re-derivation** of the behavioral confirmatory analysis from the committed data of the read-only reification-gradient clone. The analysis code and its statistical library (`statsmodels`) were **not** reused: the logit + cluster-robust sandwich was reimplemented in numpy, so this is a genuine second derivation, not a re-run. Script `paper/cold_verify_behavioral.py`; machine-readable `paper/cold_verification_behavioral.json`.

**Standing rule applied:** if any number failed to reproduce the committed `phase6/RESULTS.md`, it would be reported here as a **discrepancy and nothing reconciled**. **None did — all 35 checks match.**

---

## Gate 0 — provenance chain (git-level, verified out of band)

| link | value | check |
|---|---|---|
| tag `prereg-v1` → tag object | `1e67b02e…` | ✓ |
| tag → commit | **`4b2464f`** | ✓ |
| tagger | `Antonio Bernal <bernalsuarezantonio@gmail.com>` (subject "preregistration freeze") | ✓ |
| `PLAN.md` present at `4b2464f` | `git cat-file -e 4b2464f:PLAN.md` | ✓ |
| data commit descends from tag | `git merge-base --is-ancestor 4b2464f 770fa9c` → true | ✓ |
| data commit | **`770fa9c`** ("data: confirmatory run (prereg-v1)") | ✓ |
| scoring commit | `c4a5ce8` ("data(6a): complete judge scoring to 7200") | ✓ |
| results commit | `64166cd` ("results(6): confirmatory analysis + … RESULTS.md") | ✓ |

**Input blobs (extracted read-only; content sha256 recorded):**

| file | git blob | sha256 | rows |
|---|---|---|---|
| `resultados_tirada_real/responses.jsonl` @ `770fa9c` | `44f3407` | `d3e5bf67…` | 7 200 + 1 footer |
| `phase6/scored_full.jsonl` @ `c4a5ce8` | `a0ca7cf` | `fb85367d…` | 7 200 |

The `scored_full.jsonl` blob is byte-identical at `c4a5ce8` and at HEAD.

---

## 1. Completeness of the run (data commit `770fa9c`)

| check | value |
|---|---|
| response records (excluding the run-end footer) | **7 200** |
| run-end footer `expected_cells` | **7 200** (matches) |
| duplicate (model×disorder×level×vignette×rep) keys | **0** |
| condition×level cells | **20** (4 disorders × 5 levels) |
| n per condition×level cell | **360 = 360** (min = max; 60 vignettes × 3 reps × 2 models) |

## 2. Analytic dataset (scored, `c4a5ce8`)

| check | re-derived | RESULTS.md | match |
|---|---|---|---|
| scored responses | 7 200 | 7 200 | ✓ |
| analytic rows | **7 169** | 7 169 | ✓ |
| excluded (persistent malformed judge JSON) | **31** (0.43 %) | 31 | ✓ |

## 3. The four confirmatory tests — re-derived vs committed

Model C, re-derived: logit MLE (IRLS) of `diagnosis ~ C(disorder, ref='DN_plausible') * level_c`, `level_c = level_num − 3`, **cluster-robust SE by vignette** (60 clusters, sandwich with the `G/(G−1)·(N−1)/(N−K)` correction), one-sided tests, α = 0.0125.

| H | term | coef (re-derived / committed) | OR | z | p (1-sided) | verdict | match |
|---|---|---|---|---|---|---|---|
| **H1** | DN_plausible slope (`level_c`), >0? | **−0.0391 / −0.0391** | 0.962 | −2.257 | 0.988 | **not significant** | ✓ |
| **H2** | `incoherent:level_c` (inc − DN slope), <0? | **+0.4514 / +0.4514** | 1.571 | +12.964 | 1.000 | **not significant** | ✓ |
| **H3** | `real_anchor:level_c` (anchor − DN slope), <0? | **+0.0810 / +0.0810** | 1.084 | +2.793 | 0.997 | **not significant** | ✓ |
| **H4** | DN_flagged main effect at matched level, <0? | **−0.1570 / −0.1570** | 0.855 | −4.075 | <0.0001 | **SIGNIFICANT** | ✓ |

**H1 Jonckheere–Terpstra on DN_plausible:** re-derived **z = −1.180, p = 0.881** vs committed z = −1.180, p = 0.881 ✓ (not significant). Anchor's own slope (H3 auxiliary): re-derived **+0.042** log-odds/level vs committed +0.042 ✓.

Every SE reproduces to < 0.005 despite the independent sandwich implementation (H1 0.0173, H2 0.0348, H3 0.0290, H4 0.0385).

## 4. Robustness ordering (PLAN §5, "primary output")

Expected order `DN_flagged < incoherent < DN_plausible < real_anchor`; observed P(diagnosis) means per family, re-derived and matching RESULTS.md to three decimals:

| family | DN_flagged | incoherent | DN_plausible | real_anchor | preserves order? |
|---|---|---|---|---|---|
| mistral-small3.1:24b | 0.554 | 0.490 | 0.607 | 0.451 | **No** |
| qwen2.5:32b | 0.419 | 0.218 | 0.445 | 0.184 | **No** |

**Fraction preserving the order = 0.00 (0/2)** — matches. (Observed order in both families: `real_anchor < incoherent < DN_flagged < DN_plausible`.)

---

## Verdict

**35 / 35 checks match; 0 discrepancies.** The committed behavioral `RESULTS.md` (`phase6/RESULTS.md` @ `64166cd`) reproduces exactly from the committed data under an independent re-implementation — completeness (7 200 / 0 duplicates), analytic Ns (7 169 / 31 excluded), all four H1–H4 coefficients, ORs, cluster-robust z's and one-sided p's, the Jonckheere–Terpstra trend, and the 0/2 robustness fraction. Confirmatory verdicts stand as committed: **H1 not supported, H2 not supported (opposite-signed), H3 not supported, H4 supported.** No number required reconciliation.

*The re-derivation used the reification-gradient repo read-only; no write, commit, or push was made against that clone (standing hard rule). Interpretation of these results is the PI's.*
