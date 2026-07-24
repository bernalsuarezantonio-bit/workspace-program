#!/usr/bin/env python3
# Copyright 2026.
"""Build the paper's canonical NUMBERS manifest from committed artifacts.

REGISTERED RULE: this manifest is the SINGLE SOURCE of numbers for the manuscript.
Any figure in the paper text that does not appear here is, by definition, an error.

Every entry carries: id, value, (source_file, commit, repo). Numbers are read from
committed machine-readable artifacts where they exist; the few that live only in a
committed Markdown table (recognition probe, judge alphas, robustness means) are
transcribed with their exact source file + commit and re-checked against the
cold-verification JSON where overlap exists.

  workspace-program artifacts: read from the working tree; each entry's `commit`
    is the last commit that touched that file (git log -1).
  reification-gradient artifacts: read read-only from the sibling clone via
    `git -C <clone> cat-file blob <commit>:<path>`; NO write against that clone.

Run:  .venv/Scripts/python.exe paper/build_numbers_manifest.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WP = Path(__file__).resolve().parents[1]                 # workspace-program
BEH = WP.parent / "reification-gradient"                 # read-only sibling clone
OUT_JSON = WP / "paper" / "NUMBERS.json"
OUT_MD = WP / "paper" / "NUMBERS.md"


def wp_commit(rel: str) -> str:
    r = subprocess.run(["git", "-C", str(WP), "log", "-1", "--format=%h", "--", rel],
                       capture_output=True, text=True)
    return r.stdout.strip()


def wp_json(rel: str) -> dict:
    return json.loads((WP / rel).read_text(encoding="utf-8"))


def beh_blob(commit: str, path: str) -> str:
    r = subprocess.run(["git", "-C", str(BEH), "cat-file", "blob", f"{commit}:{path}"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"cat-file failed for {commit}:{path}: {r.stderr}")
    return r.stdout


def beh_json(commit: str, path: str) -> dict:
    return json.loads(beh_blob(commit, path))


ENTRIES: list[dict] = []


def add(study, eid, value, source, commit, repo, note=""):
    ENTRIES.append({"study": study, "id": eid, "value": value,
                    "source_file": source, "commit": commit, "repo": repo, "note": note})


def main() -> int:
    # ============================ STUDY 1 (behavioral) ============================
    R = "reification-gradient"
    conf = beh_json("64166cd", "phase6/confirmatory_results.json")
    add("S1", "s1.n_analytic", conf["n_analytic"], "phase6/confirmatory_results.json",
        "64166cd", R, "7200 scored - 31 malformed")
    add("S1", "s1.n_excluded_malformed", conf["n_excluded_malformed"],
        "phase6/confirmatory_results.json", "64166cd", R)
    add("S1", "s1.alpha", conf["alpha"], "phase6/confirmatory_results.json", "64166cd", R,
        "Bonferroni 0.05/4")

    for h in ("H1", "H2", "H3", "H4"):
        tm = conf[h].get("test_model") or conf[h].get("contrast_DN_vs_anchor")
        add("S1", f"s1.{h}.coef", round(tm["coef"], 4), "phase6/confirmatory_results.json",
            "64166cd", R, conf[h]["term"] if "term" in conf[h] else h)
        add("S1", f"s1.{h}.OR", round(tm["OR"], 3), "phase6/confirmatory_results.json",
            "64166cd", R)
        add("S1", f"s1.{h}.OR_CI95", [round(x, 3) for x in tm["OR_CI95"]],
            "phase6/confirmatory_results.json", "64166cd", R)
        add("S1", f"s1.{h}.z", round(tm["z"], 3), "phase6/confirmatory_results.json",
            "64166cd", R)
        add("S1", f"s1.{h}.p_one_sided", round(tm["p_one_sided"], 4),
            "phase6/confirmatory_results.json", "64166cd", R)
        add("S1", f"s1.{h}.significant", tm["significant_at_alpha"],
            "phase6/confirmatory_results.json", "64166cd", R)
    jt = conf["H1"]["jonckheere_terpstra"]
    add("S1", "s1.H1.jonckheere_z", round(jt["z"], 3), "phase6/confirmatory_results.json",
        "64166cd", R)
    add("S1", "s1.H1.jonckheere_p", round(jt["p_one_sided"], 3),
        "phase6/confirmatory_results.json", "64166cd", R)

    # stratified P(diagnosis) by condition x level (observed, all models)
    strat = {}
    for row in conf["predicted_and_observed"]:
        strat[f"{row['disorder']}|{row['level']}"] = {"p_obs": round(row["p_obs"], 4),
                                                       "n": row["n_obs"]}
    add("S1", "s1.pdiagnosis_by_condition_level", strat,
        "phase6/confirmatory_results.json", "64166cd", R,
        "observed P(diagnosis), pooled over both models; 20 cond x level cells")

    rob = conf["robustness_ordering"]
    add("S1", "s1.robustness_fraction_preserving", rob["fraction_preserving"],
        "phase6/confirmatory_results.json", "64166cd", R,
        "fraction of families preserving DN_flagged<incoherent<DN_plausible<real_anchor")
    for fam in ("mistral-small3.1:24b", "qwen2.5:32b"):
        add("S1", f"s1.robustness_means.{fam}", rob["by_family"][f"{fam}_means"],
            "phase6/confirmatory_results.json", "64166cd", R)

    # judge validation, Krippendorff alpha per variable (committed MD table)
    add("S1", "s1.judge_alpha.diagnosis.human_judge", 0.830,
        "phase4/validation/AGREEMENT_RESULTS.md", "fd3ff52", R, "CI95 [0.604,1.000], raw 93.8%")
    add("S1", "s1.judge_alpha.diagnosis.judge_corater", 0.958,
        "phase4/validation/AGREEMENT_RESULTS.md", "fd3ff52", R, "raw 98.0%")
    for var, a, raw in (("criteria_invented", -0.076, "31.8%"),
                        ("confidence", -0.188, "50.8%"),
                        ("treatment_recommended", 0.350, "73.8%"),
                        ("differential_offered", 0.717, "99.4%"),
                        ("epistemic_flag", 0.547, "84.2%")):
        add("S1", f"s1.judge_alpha.{var}.judge_corater", a,
            "phase4/validation/AGREEMENT_RESULTS.md", "fd3ff52", R, f"raw {raw}")

    # recognition probe (the only 66.7% in the corpus: anchor_dpdr, mistral, 20/30)
    add("S1", "s1.recognition.anchor_dpdr.mistral", round(20 / 30, 4),
        "phase2/recognition_probe/tabla_resumen.md", "5382d9b", R,
        "20/30 = 0.6667 (66.7%); the DPDR real-anchor recognized by mistral-small3.1:24b")
    add("S1", "s1.recognition.anchor_dpdr.qwen", round(21 / 30, 4),
        "phase2/recognition_probe/tabla_resumen.md", "5382d9b", R, "21/30 = 0.70")
    add("S1", "s1.recognition.DN.both_families", 0.0,
        "phase2/recognition_probe/tabla_resumen.md", "5382d9b", R,
        "disprosexia narrativa recognized 0/30 in both families")

    # ====================== STUDY 2 (Phase 1 representational) ====================
    W = "workspace-program"
    r1 = wp_json("phase1/data/results_phase1.json")
    c1c = wp_commit("phase1/data/results_phase1.json")
    add("S2", "s2.C1.flagged_mean_F", round(r1["C1"]["flagged_mean_F"], 4),
        "phase1/data/results_phase1.json", c1c, W, "Set F loading, DN_flagged x high x L1")
    add("S2", "s2.C1.plausible_mean_F", round(r1["C1"]["plausible_mean_F"], 4),
        "phase1/data/results_phase1.json", c1c, W)
    add("S2", "s2.C1.mean_diff", round(r1["C1"]["test_result"]["mean_diff"], 4),
        "phase1/data/results_phase1.json", c1c, W)
    add("S2", "s2.C1.t", round(r1["C1"]["test_result"]["t"], 3),
        "phase1/data/results_phase1.json", c1c, W, "paired t(19), two-sided")
    add("S2", "s2.C1.p_two_sided", r1["C1"]["test_result"]["p_two_sided"],
        "phase1/data/results_phase1.json", c1c, W)
    add("S2", "s2.C2.L4_mean_A", round(r1["C2"]["L4_mean_A"], 4),
        "phase1/data/results_phase1.json", c1c, W, "Set A loading, incoherent x high")
    add("S2", "s2.C2.L1_mean_A", round(r1["C2"]["L1_mean_A"], 4),
        "phase1/data/results_phase1.json", c1c, W)
    add("S2", "s2.C2.t", round(r1["C2"]["test_result"]["t"], 3),
        "phase1/data/results_phase1.json", c1c, W, "paired t(19), one-sided L4>L1")
    add("S2", "s2.C2.p_one_sided", r1["C2"]["test_result"]["p_one_sided_upper"],
        "phase1/data/results_phase1.json", c1c, W, "H1 L4>L1 NOT supported (opposite sign)")
    add("S2", "s2.diagnosis_rates",
        {k: round(v["rate"], 4) for k, v in r1["diagnosis_rates"].items()},
        "phase1/data/results_phase1.json", c1c, W)

    sub2 = r1["exploratory"]["sub2_F_by_stratum"]
    add("S2", "s2.mention_split.with_mention_n", sub2["n_with"],
        "phase1/data/results_phase1.json", c1c, W, "flagged x diagnosis=1, textual mention")
    add("S2", "s2.mention_split.without_mention_n", sub2["n_without"],
        "phase1/data/results_phase1.json", c1c, W)
    add("S2", "s2.mention_split.with_mention_meanF", round(sub2["flagged_with_mention_meanF"], 4),
        "phase1/data/results_phase1.json", c1c, W)
    add("S2", "s2.mention_split.without_mention_meanF",
        round(sub2["flagged_without_mention_meanF"], 4),
        "phase1/data/results_phase1.json", c1c, W)

    ap = wp_json("phase1/data/results_phase1_appendix.json")
    apc = wp_commit("phase1/data/results_phase1_appendix.json")
    a1 = ap["A1_registered_decisive_cell"]["paired_test"]
    add("S2", "s2.A1.mean_diff", round(a1["mean_diff"], 4),
        "phase1/data/results_phase1_appendix.json", apc, W,
        "flagged-without-mention vs plausible, Set F, paired t(19) two-sided")
    add("S2", "s2.A1.ci95", [round(x, 4) for x in a1["ci95"]],
        "phase1/data/results_phase1_appendix.json", apc, W)
    add("S2", "s2.A1.t", round(a1["t"], 3), "phase1/data/results_phase1_appendix.json", apc, W)
    add("S2", "s2.A1.p_two_sided", a1["p_two_sided"],
        "phase1/data/results_phase1_appendix.json", apc, W)
    a2 = ap["A2_new_spanish_mask"]["paired_test"]
    add("S2", "s2.A2.mean_diff", round(a2["mean_diff"], 4),
        "phase1/data/results_phase1_appendix.json", apc, W,
        "Spanish-surface emission mask, Set F EN-concept, paired t(19)")
    add("S2", "s2.A2.ci95", [round(x, 4) for x in a2["ci95"]],
        "phase1/data/results_phase1_appendix.json", apc, W)
    add("S2", "s2.A2.t", round(a2["t"], 3), "phase1/data/results_phase1_appendix.json", apc, W)
    add("S2", "s2.A2.p_two_sided", a2["p_two_sided"],
        "phase1/data/results_phase1_appendix.json", apc, W)

    # ================= STUDY 3 (Phase 2 amplification + Phase 2b ablation) ========
    pil = wp_json("phase2/data/pilot_calibration.json")
    pc = wp_commit("phase2/data/pilot_calibration.json")
    add("S3", "s3.pilot.reference_alpha0_meanF", round(pil["reference_alpha0"]["0.0"]["mean_f_loading"], 4),
        "phase2/data/pilot_calibration.json", pc, W, "alpha=0 reference on neutral pilot material")
    add("S3", "s3.pilot.k0.05_meanF", round(pil["frozen_ladder"]["0.05"]["mean_f_loading"], 3),
        "phase2/data/pilot_calibration.json", pc, W, "lowest rung; 331.8x natural 0.0825")
    add("S3", "s3.pilot.k0.05_malformed_rate", pil["frozen_ladder"]["0.05"]["malformed_rate"],
        "phase2/data/pilot_calibration.json", pc, W)
    add("S3", "s3.pilot.k0.1_malformed_rate", pil["frozen_ladder"]["0.1"]["malformed_rate"],
        "phase2/data/pilot_calibration.json", pc, W, "100% malformed")
    add("S3", "s3.pilot.k_max", pil["selection"]["k_max"],
        "phase2/data/pilot_calibration.json", pc, W)
    add("S3", "s3.pilot.rule_satisfiable", pil["selection"]["rule_satisfiable"],
        "phase2/data/pilot_calibration.json", pc, W, "False -> Phase 2 closed instrument-negative")
    add("S3", "s3.pilot.three_distinct_doses", pil["selection"]["three_distinct_doses"],
        "phase2/data/pilot_calibration.json", pc, W)

    j0 = wp_json("phase2b/data/ablation_effect.json")["summary"]
    jc = wp_commit("phase2b/data/ablation_effect.json")
    add("S3", "s3.J0b.mean_F_base", round(j0["mean_F_base"], 4),
        "phase2b/data/ablation_effect.json", jc, W, "instruct-lens F loading, no ablation (circular est.)")
    add("S3", "s3.J0b.mean_F_ablated", round(j0["mean_F_ablated"], 4),
        "phase2b/data/ablation_effect.json", jc, W)
    add("S3", "s3.J0b.F_reduction_pct", round(j0["F_reduction_pct"], 1),
        "phase2b/data/ablation_effect.json", jc, W)
    add("S3", "s3.J0b.generations_changed", f"{j0['n_vignettes'] - j0['identical_generations']}/{j0['n_vignettes']}",
        "phase2b/data/ablation_effect.json", jc, W, "greedy, seed-matched")
    add("S3", "s3.J0b.malformed_ablated", j0["malformed_ablated"],
        "phase2b/data/ablation_effect.json", jc, W, "0/20")

    r2 = wp_json("phase2b/data/results_phase2b.json")
    r2c = wp_commit("phase2b/data/results_phase2b.json")
    for a in ("B0_none", "B1_full", "B3_rand"):
        add("S3", f"s3.2b.rate.{a}",
            {"diagnosis": r2["arm_overall"][a]["dv1_diagnosis"],
             "mention": r2["arm_overall"][a]["dv2_mention"], "n": r2["arm_overall"][a]["n"]},
            "phase2b/data/results_phase2b.json", r2c, W)
    for t in r2["confirmatory"]:
        add("S3", f"s3.2b.{t['test']}",
            {"contrast": t["contrast"], "estimate": t["estimate_mean_diff"],
             "ci95": [round(x, 4) for x in t["ci95"]],
             "p": round(t["p_permutation_one_sided"], 4), "verdict": t["verdict"]},
            "phase2b/data/results_phase2b.json", r2c, W, "alpha=0.0125, sign-flip permutation")
    add("S3", "s3.2b.joint_cell", r2["joint_table_cell"]["registered_reading"],
        "phase2b/data/results_phase2b.json", r2c, W)

    # ================================ emit =======================================
    manifest = {
        "title": "Program numbers manifest — single source of truth for the manuscript",
        "registered_rule": "This manifest is the ONLY source of numbers for the paper. "
                           "Any figure in the text not present here is, by definition, an error.",
        "repos": {"workspace-program": str(WP), "reification-gradient": str(BEH) + " (read-only)"},
        "n_entries": len(ENTRIES),
        "entries": ENTRIES,
    }
    OUT_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown
    lines = ["# NUMBERS — canonical figures for the manuscript", "",
             "> **Registered rule:** this manifest is the **single source of numbers** for the "
             "paper. Any figure in the manuscript text that does not appear here is, by "
             "definition, an error. Generated by `paper/build_numbers_manifest.py` from committed "
             "artifacts of both repos.", "",
             f"**{len(ENTRIES)} entries.** `reification-gradient` read read-only; no write against "
             "that clone.", ""]
    for study, title in (("S1", "Study 1 — behavioral (reification-gradient)"),
                         ("S2", "Study 2 — representational J-lens (workspace-program Phase 1)"),
                         ("S3", "Study 3 — causal amplification + ablation (Phase 2 / 2b)")):
        lines += [f"## {title}", "", "| id | value | source | commit |", "|---|---|---|---|"]
        for e in ENTRIES:
            if e["study"] != study:
                continue
            v = e["value"]
            vs = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            if len(vs) > 90:
                vs = vs[:87] + "…"
            note = f" — {e['note']}" if e["note"] else ""
            lines.append(f"| `{e['id']}` | {vs}{note} | `{e['source_file']}` | `{e['commit']}` |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(ENTRIES)} entries -> {OUT_MD.relative_to(WP)} + {OUT_JSON.relative_to(WP)}")
    by_study = {s: sum(1 for e in ENTRIES if e["study"] == s) for s in ("S1", "S2", "S3")}
    print("by study:", by_study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
