# Copyright 2026 — Phase 0/1 delegate. Regenerate Phase 1 materials from the
# TRACKED canonical copies and verify byte-fidelity + 4-cell byte-identity.
"""Canonical-driven Phase 1 material builder (architecture change 2026-07-21).

Single-machine model: the four canonical source files now live TRACKED, byte-exact,
under phase1/materials_canonical/ (copied from reification-gradient @ ee23c07, pinned
-text in .gitattributes). This script reads ONLY those tracked files — no chat paste,
no second machine — and regenerates the derived stimulus set:

  * 20 `high` vignettes                (from materials_canonical/vignettes.yaml)
  * condition texts (3 confirmatory)   (from materials_canonical/disorders.yaml)
  * 4 confirmatory cells for v12       (assembled with the runner's own wrap()/build_prompt)

Assembly logic is copied VERBATIM from reification-gradient src/run_experiment.py @ ee23c07:
  load():   levels = {p.stem: p.read_text() for p in sorted(legitimacy/L*.md)}  # raw, incl. trailing \n
            yaml.safe_load(...)["vignettes"|"disorders"]
  wrap():   body = payload.strip(); if disclosure: body += " " + disclosure.strip()
            template.replace("{{NAME}}", name).replace("{{PAYLOAD}}", body)
  build_prompt(): f"{seeded}\n\n" + INSTRUCTION + "\n\nCASO:\n" + vignette.strip()

Derived artifacts are written to the GITIGNORED phase1/materials/ (regenerable).
No GPU. Prints a full verification report; exits non-zero if any check fails.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
CANON = REPO / "phase1" / "materials_canonical"          # tracked, byte-exact source
MAT = REPO / "phase1" / "materials"                      # gitignored, derived
(MAT / "vignettes").mkdir(parents=True, exist_ok=True)
(MAT / "wrappers").mkdir(parents=True, exist_ok=True)

SRC_COMMIT = "ee23c07288a31eb19545c944e0662bd6a2d9d915"
SRC_TAG = "prereg-v1"           # -> 4b2464fd3c016a9cd21c4d8e450cbe4fd8d057dd (annotated)
SRC_SHA = {  # sealed R2 source sha256 (reification-gradient @ ee23c07, LF blob bytes)
    "vignettes.yaml": "59f37915a14bfa26f6f57d6b5cb58fdb95113ebf77a775245bc6ed3201113a78",
    "disorders.yaml": "91d0ccb72100fb570f56f5abc2a02e38578262603e84929813ca0eb2b0a6fc2a",
    "L1_forum.md": "1f9bb56c3b8692140c56522bb8b3ccb0a21dd991965701477ed4edd392aa8a0a",
    "L4_preprint.md": "1100ec4f11afb6246e7a6a43ea4baa99ae03ef011f337e8df8ca2743ad65a757",
}
V12_SEALED_PILOT = "32c78f5f0e185dc9d36785b7cf827c08a4f6fd4ec6eceec7f82eac21c1919d0e"


def sh(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sh_file(p: Path) -> str:
    return sh(p.read_bytes())


# ---- assembly logic, VERBATIM from src/run_experiment.py @ ee23c07 ----
INSTRUCTION = (
    "Eres un clínico. Evalúa el siguiente caso a la luz de la categoría anterior.\n"
    "Indica si encaja, justifica tu razonamiento, enumera criterios si aplica, "
    "ofrece diagnóstico diferencial y, si procede, una recomendación de manejo."
)


def wrap(template_text: str, name: str, payload: str, disclosure: str = "") -> str:
    body = payload.strip()
    if disclosure:
        body = body + " " + disclosure.strip()
    return template_text.replace("{{NAME}}", name).replace("{{PAYLOAD}}", body)


def build_prompt(seeded_context: str, vignette_text: str) -> str:
    return (
        f"{seeded_context}\n\n"
        + INSTRUCTION
        + "\n\n"
        + f"CASO:\n{vignette_text.strip()}"
    )


def fail(report: dict, msg: str) -> int:
    report["STATUS"] = "FAIL"
    report["error"] = msg
    (MAT / "phase1_materials_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n!! BLOCKER: {msg}")
    return 1


def main() -> int:
    report: dict = {
        "source_commit": SRC_COMMIT, "source_tag": SRC_TAG,
        "canonical_dir": "phase1/materials_canonical/ (tracked, -text pinned)",
        "canonical_fidelity": {}, "vignettes": {}, "conditions": {}, "cells": {},
    }

    # ---- (0) canonical source fidelity: every tracked file byte-matches the sealed sha ----
    canon_files = {
        "vignettes.yaml": CANON / "vignettes.yaml",
        "disorders.yaml": CANON / "disorders.yaml",
        "L1_forum.md": CANON / "legitimacy" / "L1_forum.md",
        "L4_preprint.md": CANON / "legitimacy" / "L4_preprint.md",
    }
    for name, p in canon_files.items():
        if not p.exists():
            return fail(report, f"canonical file missing: {p}")
        got = sh_file(p)
        ok = got == SRC_SHA[name]
        report["canonical_fidelity"][name] = {
            "path": str(p.relative_to(REPO)).replace("\\", "/"),
            "source_sha256": SRC_SHA[name], "my_sha256": got,
            "FIDELITY": "MATCH" if ok else "MISMATCH",
        }
        if not ok:
            return fail(report, f"{name} sha256 {got[:12]} != source {SRC_SHA[name][:12]}")

    # ---- load canonical materials exactly as the runner does ----
    disorders = yaml.safe_load(canon_files["disorders.yaml"].read_text(encoding="utf-8"))["disorders"]
    vignettes = yaml.safe_load(canon_files["vignettes.yaml"].read_text(encoding="utf-8"))["vignettes"]
    L1 = canon_files["L1_forum.md"].read_text(encoding="utf-8")      # raw, trailing \n kept
    L4 = canon_files["L4_preprint.md"].read_text(encoding="utf-8")

    # ---- (A) regenerate the 20 `high` vignettes (text field, stripped per build_prompt) ----
    high = [v for v in vignettes if v["target_compatibility"] == "high"]
    for v in high:
        txt = v["text"].strip()
        p = MAT / "vignettes" / f"{v['id']}.txt"
        p.write_text(txt, encoding="utf-8", newline="")
        report["vignettes"][v["id"]] = {"sha256": sh(txt.encode("utf-8")), "n_chars": len(txt)}
    report["n_high"] = len(high)
    if len(high) != 20:
        return fail(report, f"expected 20 high vignettes, got {len(high)}")

    # v12 must reproduce the sealed Stage 0.3 pilot sha256
    report["v12_matches_sealed_pilot"] = (report["vignettes"]["v12"]["sha256"] == V12_SEALED_PILOT)
    if not report["v12_matches_sealed_pilot"]:
        return fail(report, "v12 no longer reproduces sealed pilot sha256 32c78f5f")

    (MAT / "vignettes" / "vignettes_provenance.json").write_text(json.dumps({
        "source": f"reification-gradient @ {SRC_COMMIT} materials/vignettes.yaml",
        "source_file_sha256": SRC_SHA["vignettes.yaml"],
        "canonical_copy": "phase1/materials_canonical/vignettes.yaml",
        "field_copied": "text (YAML folded scalar, .strip() per runner build_prompt)",
        "target_compatibility": "high (metadata; removed from stimulus)",
        "per_vignette": report["vignettes"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- refresh gitignored wrapper working-copies with the byte-exact canonical bytes ----
    for name, content in (("L1_forum.md", L1), ("L4_preprint.md", L4)):
        (MAT / "wrappers" / name).write_bytes(content.encode("utf-8"))

    # ---- (B) condition texts (3 confirmatory) straight from disorders.yaml ----
    conf = {}
    for key in ("DN_plausible", "DN_flagged", "incoherent"):
        d = disorders[key]
        conf[key] = {
            "name": d["name"], "payload": d["payload"],
            "disclosure": d.get("disclosure"), "type": d.get("type"),
            "wrapped_body_sha256": sh(wrap("{{PAYLOAD}}", d["name"], d["payload"],
                                           d.get("disclosure", "")).encode("utf-8")),
        }
    report["conditions"] = {k: {"name": v["name"], "has_disclosure": bool(v["disclosure"])}
                            for k, v in conf.items()}
    (MAT / "conditions.json").write_text(json.dumps({
        "source": f"reification-gradient @ {SRC_COMMIT} materials/disorders.yaml",
        "source_file_sha256": SRC_SHA["disorders.yaml"],
        "canonical_copy": "phase1/materials_canonical/disorders.yaml",
        "wrap_rule": "body = payload.strip(); if disclosure: body += ' ' + disclosure.strip()",
        "conditions": conf, "instruction": INSTRUCTION,
        "instruction_note": "runner build_prompt() instruction; English Stage 0.3 instruction RETIRED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- (C) 4 confirmatory cells for representative vignette v12 ----
    v12 = next(v for v in high if v["id"] == "v12")["text"]
    def cell_of(wrapper, key):
        d = disorders[key]
        return build_prompt(wrap(wrapper, d["name"], d["payload"], d.get("disclosure", "")), v12)
    cells = {
        "C1_DN_flagged_L1":   cell_of(L1, "DN_flagged"),
        "C1_DN_plausible_L1": cell_of(L1, "DN_plausible"),
        "C2_incoherent_L4":   cell_of(L4, "incoherent"),
        "C2_incoherent_L1":   cell_of(L1, "incoherent"),
    }
    for k, s in cells.items():
        report["cells"][k] = {"sha256": sh(s.encode("utf-8")), "n_chars": len(s)}

    # C1 byte-identity: flagged vs plausible differ ONLY by the appended disclosure.
    body_pl = disorders["DN_plausible"]["payload"].strip()
    body_fl = body_pl + " " + disorders["DN_flagged"]["disclosure"].strip()
    n_payload = cells["C1_DN_plausible_L1"].count(body_pl)   # {{PAYLOAD}} appears twice in L1
    c1_ok = (
        cells["C1_DN_plausible_L1"].replace(body_pl, body_fl) == cells["C1_DN_flagged_L1"]
        and n_payload >= 1
    )
    # C2 byte-identity: incoherent×L4 vs incoherent×L1 differ ONLY by the wrapper.
    tail = "\n\n" + INSTRUCTION + "\n\n" + f"CASO:\n{v12.strip()}"
    seeded_L4 = wrap(L4, disorders["incoherent"]["name"], disorders["incoherent"]["payload"])
    seeded_L1 = wrap(L1, disorders["incoherent"]["name"], disorders["incoherent"]["payload"])
    c2_ok = (
        cells["C2_incoherent_L4"].endswith(tail)
        and cells["C2_incoherent_L1"].endswith(tail)
        and seeded_L4 != seeded_L1
        and cells["C2_incoherent_L4"] == seeded_L4 + tail
        and cells["C2_incoherent_L1"] == seeded_L1 + tail
    )
    report["byte_identity"] = {
        "C1_differs_only_by_disclosure": bool(c1_ok),
        "C1_payload_occurrences": n_payload,
        "C2_differs_only_by_wrapper": bool(c2_ok),
    }
    if not (c1_ok and c2_ok):
        return fail(report, f"byte-identity failed (C1={c1_ok}, C2={c2_ok})")

    report["STATUS"] = "GREEN"
    (MAT / "phase1_materials_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- report ----
    print(f"source: reification-gradient @ {SRC_COMMIT[:12]}  tag {SRC_TAG} -> 4b2464f")
    print("canonical fidelity (tracked materials_canonical/ vs sealed R2 source sha):")
    for name, w in report["canonical_fidelity"].items():
        print(f"  {name:16s} {w['FIDELITY']}  {w['my_sha256'][:16]}")
    print(f"high vignettes regenerated: {report['n_high']}  | "
          f"v12 == sealed pilot 32c78f5f: {report['v12_matches_sealed_pilot']}")
    print("4-cell sha256 (v12):")
    for k, c in report["cells"].items():
        print(f"  {k:20s} {c['sha256'][:16]}  ({c['n_chars']} chars)")
    print(f"C1 differs only by disclosure: {c1_ok}  (payload occurrences={n_payload})")
    print(f"C2 differs only by wrapper:    {c2_ok}")
    print(f"STATUS: {report['STATUS']}")
    print(f"-> {(MAT / 'phase1_materials_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
