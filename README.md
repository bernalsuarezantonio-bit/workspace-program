# workspace-program — "Held but not heeded" (representational & causal studies)

When a language model writes *"note: this category is not a recognized disorder"* and, in the same
breath, diagnoses someone with it — is that first sentence a report of something the model holds, or
just text it emits? This repository holds the **representational (Study 2)** and **causal (Study 3)**
arms of a three-study preregistered program that answers that question causally for one clean case.

Using a pre-fitted **verbalizable-workspace (Jacobian) lens** on `Qwen2.5-7B-Instruct`, the program
shows that the "this is fiction" status is **genuinely held** in the model's workspace during
diagnostic generation (Study 2), yet **behaviorally inert**: projection-ablating that representation
**cuts written acknowledgments by ~40%** (p = 0.0007, direction-specific) **while diagnosis stays at
100%** in every arm and vignette (Study 3). The held representation of fictional status feeds a
dedicated **verbal channel with no measurable authority over behavior**.

- **Manuscript:** [`paper/preprint_held_but_not_heeded_v2.md`](paper/preprint_held_but_not_heeded_v2.md)
- **Study 1 (behavioral, 7,200 runs)** lives in the sibling repo **`reification-gradient`** — the
  fabricated category "narrative dysprosexia" and the behavioral phenomenon it establishes. This repo
  *cold-verifies* Study 1's numbers (see `paper/COLD_VERIFICATION_BEHAVIORAL.md`) but does not contain
  its run harness.

Vendored third-party code (`vendor/jacobian-lens`, Apache-2.0) is **gitignored** and not distributed
here; provenance and pins are in `PROVENANCE.md`.

---

## Repository structure

```
workspace-program/
├── PROVENANCE.md            Chain-of-custody spine: every stage, hash, digest, seed, and incident
├── PREREG_PHASE1.md         Preregistration — Study 2 (representational)   → tag prereg-phase1-v1
├── PREREG_PHASE2.md         Preregistration — Study 3 amplification arm    → tag prereg-phase2-v1
├── PREREG_PHASE2B.md        Preregistration — Study 3 ablation arm         → tag prereg-phase2b-v1
├── phase1_token_sets_SEALED.md   Sealed concept/token sets + matching rules R1–R5 (pre-data)
├── RESULTS_PHASE1.md        Study 2 results
├── LICENSE / LICENSE-CC-BY-4.0.txt / CITATION.cff
│
├── paper/                   THE MANUSCRIPT and its verification apparatus
│   ├── preprint_held_but_not_heeded_v2.md   Full three-study write-up
│   ├── NUMBERS.{md,json}       Single source of truth: every number ↔ its committed artifact
│   ├── HISTORY_REWRITE_MAP.md  old→new commit/tag correspondence (pre-publication rewrite)
│   ├── COLD_VERIFICATION_BEHAVIORAL.md   Cold re-check of Study 1 (35/35 match)
│   ├── references.md / references_candidates.md   Verified bibliography
│   ├── mentions_behavioral*.{json,md}     Acknowledgment-mention analysis + verbatims
│   ├── figures/                fig1–fig3 (PDF + PNG + per-figure CSV)
│   └── *.py                    Committed scripts (figures, NUMBERS manifest, cold verify)
│
├── phase0/                  Recon + env setup (iMac → RTX 5090) + feasibility pilots + seals
│   ├── reports/  logs/  scripts/  (stage01 recon, stage02 validation, stage03 pilot/nightly)
│
├── phase1/                  STUDY 2 — representational (800 lens runs)
│   ├── materials_canonical/    Byte-exact Study-1 materials (pinned -text; sealed sha256)
│   ├── scripts/  data/  logs/   run harness, judge, results, completeness report
│   └── POSTHOC_EXPLORATORY_C1_selfecho.md
│
├── phase2/                  STUDY 3a — amplification (CLOSED, instrument-negative)
│   ├── I0_RECON.md  PILOT_CALIBRATION.md  CLOSURE.md
│   └── scripts/  data/          intervene / measure_rho / verify_injection
│
└── phase2b/                 STUDY 3b — projection ablation (600 runs; the central finding)
    ├── RESULTS_PHASE2B.md
    └── scripts/  data/          ablate / gate0 / projector-verify / analyze
```

Repository↔study map: `phase1` = **Study 2**; `phase2` = **Study 3 amplification** (closed);
`phase2b` = **Study 3 ablation**.

---

## Reading order of the preregistrations

The confirmatory record is a chain of dated, tagged freezes plus the pre-data seals they depend on.
Read in this order:

1. **[`paper/preprint_held_but_not_heeded_v2.md`](paper/preprint_held_but_not_heeded_v2.md)** — the
   overview. §7 (Transparency, incidents, and chain of custody) frames everything below.
2. **[`PROVENANCE.md`](PROVENANCE.md)** — the custody spine: Phase 0 recon and environment, the
   **token-set seal chain** (A0 → A1 → C-note), pins (model/lens digests), and the **incident log**
   (see *Incident #3* — a phantom results file caught by Gate-0).
3. **[`phase1_token_sets_SEALED.md`](phase1_token_sets_SEALED.md)** — the concept sets and matching
   rules **R1–R5**, sealed (hashed, pushed for external timestamp) *before any condition-bearing
   readout existed*. This is what protects Study 2 from circularity.
4. **[`PREREG_PHASE1.md`](PREREG_PHASE1.md)** (tag `prereg-phase1-v1`, 2026-07-21) — Study 2 design
   and the two contrasts → results in [`RESULTS_PHASE1.md`](RESULTS_PHASE1.md).
5. **[`PREREG_PHASE2.md`](PREREG_PHASE2.md)** (tag `prereg-phase2-v1`, 2026-07-22) — the amplification
   arm, with named open slots gated on Stage I0 → closed instrument-negative in
   [`phase2/CLOSURE.md`](phase2/CLOSURE.md).
6. **[`PREREG_PHASE2B.md`](PREREG_PHASE2B.md)** (tag `prereg-phase2b-v1`, 2026-07-22) — the ablation
   arm and its pre-fixed joint-reading table → results in
   [`phase2b/RESULTS_PHASE2B.md`](phase2b/RESULTS_PHASE2B.md).
7. **[`paper/NUMBERS.md`](paper/NUMBERS.md)** — read last: every number in the manuscript traced back
   to the committed artifact and commit it comes from.

Read the *exact frozen* text of any preregistration with `git show <tag>:<file>` (see below).

---

## Verifying the chain of custody, step by step

The program's rule (manuscript §7): every confirmatory element was frozen in version control **before
the corresponding data existed**, and every reported number derives from a committed artifact whose
input digest is recomputed cold — the **Gate-0** chain `tag → preregistration blob → data commit →
dataset digest`. Each step below is checkable from a clone.

**1 — Each preregistration was frozen before its data.** The prereg tag must be an ancestor of (and
dated before) its confirmatory-data commit.

```bash
# Study 2 (phase1): tag prereg-phase1-v1 → data commit 8046a12 (800 runs, digest dc522361)
git merge-base --is-ancestor prereg-phase1-v1 8046a12 && echo "OK: phase1 prereg precedes data"
git log -1 --format='tag  %ci' prereg-phase1-v1^{commit}   # 2026-07-21 10:12
git log -1 --format='data %ci' 8046a12                     # 2026-07-21 16:14

# Study 3b (phase2b): tag prereg-phase2b-v1 → data commit 317ddb9 (600 runs, digest aa56df8d)
git merge-base --is-ancestor prereg-phase2b-v1 317ddb9 && echo "OK: phase2b prereg precedes data"
```

**2 — The preregistration blobs are the ones cited.** The sha256 of each frozen prereg is recorded in
`PROVENANCE.md` / `paper/HISTORY_REWRITE_MAP.md` and survives the history rewrite unchanged.

```bash
git show prereg-phase1-v1:PREREG_PHASE1.md   | shasum -a 256   # → bedbcc78…
git show prereg-phase2b-v1:PREREG_PHASE2B.md | shasum -a 256   # → f17ac365…
```

**3 — The token sets were sealed pre-data.** The seal chain is a three-link hash ladder; the final
sealed file is LF-pinned so its hash is stable across platforms.

```bash
grep -nE '3689ac85|9530aceb|cfce4742' PROVENANCE.md     # A0 → A1 → C-note seal chain
cat .gitattributes                                      # phase1_token_sets_SEALED.md text eol=lf
```
The sealed rules guarantee **no confirmatory token appears anywhere in any stimulus** (substring-strict
echo exclusion, R2) — the anti-circularity core of Study 2.

**4 — The canonical Study-1 materials are byte-exact.** They are pinned `-text` so their sealed sha256
never drift; they are copied from `reification-gradient @ ee23c07`.

```bash
grep -n 'materials_canonical' .gitattributes            # pinned -text (no renormalization)
git ls-files phase1/materials_canonical/
```

**5 — Models and lens are digest-pinned.** Generators/judge weights (sha256) and the lens `.pt`
(sha256, plus a `torch.isfinite()` guard on load) are recorded in `PROVENANCE.md`.

```bash
grep -nE 'sha256|digest|isfinite|581d398|16a01f3' PROVENANCE.md | head
```

**6 — History-rewrite integrity (translation table).** A pre-publication rewrite (2026-07-24) remapped
commit authorship and redacted personal identifiers; **dates, messages, file contents, preregistration
blobs, and all data digests were preserved**. The old↔new correspondence is the translation table
**[`paper/HISTORY_REWRITE_MAP.md`](paper/HISTORY_REWRITE_MAP.md)**.

The **sealed preregistrations deliberately keep the commit/tag hashes that were current at their freeze**
— editing a sealed, externally-timestamped document would break its seal, so those pre-rewrite hashes
are *correct, not errors*. To resolve any hash a sealed document cites against today's history, look it
up in the map's "old" column to find its "new" equivalent.

Only the **live (non-sealed) publication artifacts** — the manuscript, `NUMBERS.{md,json}`, and
`references*.md` — are kept citing post-rewrite, resolvable hashes. This sweep expects **zero** retired
hashes in them:

```bash
olds=$(awk -F'`' '/^\| `[0-9a-f]{7}` \| `[0-9a-f]{7}` \|$/ && $2 != $4 {print $2}' \
  paper/HISTORY_REWRITE_MAP.md | paste -sd'|' -)
grep -nHE "\b($olds)\b" paper/preprint_held_but_not_heeded_v2.md paper/NUMBERS.md \
  paper/NUMBERS.json paper/references.md paper/references_candidates.md \
  || echo "clean: no retired hash in the live publication artifacts"
```
(Any hit is a retired hash in a live artifact — update it to the "new" value via the map. Hashes that
fail `git cat-file -e` but are *not* in the map are cross-repository references to `reification-gradient`
or upstream revisions, not errors.)

**7 — Every number traces to an artifact.** `paper/NUMBERS.md` is the single source of truth; figures
were produced by committed scripts reading only committed CSVs; Study 1's numbers were cold-verified
here.

```bash
grep -nE 'commit|digest|artifact' paper/NUMBERS.md | head
git log --oneline | grep -iE 'cold verification|NUMBERS|figures'
```

**8 — Completion claims require an artifact (incident-hardened).** During the program, three interim
reports (delivered through a conversational interface) were found on mechanical verification to describe
work that had not occurred — caught by Gate-0 before contaminating any result (manuscript §7;
`PROVENANCE.md` *Incident #3*). The standing rule: no claim of completion or result is accepted, by
human or model, without a repository artifact and hash.

---

## Ethics & scope

Synthetic-only, sandbox-only. The fabricated category "narrative dysprosexia" and all synthetic
disorder content exist for controlled in-silico measurement and are never presented as real clinical
claims.

## Licensing & citation

Dual-licensed: **MIT** for code, **CC-BY-4.0** for materials, data, preregistrations, and the
manuscript — see [`LICENSE`](LICENSE) for the exact scope of each.

**How to cite.** Until the preprint is posted with a DOI, this README is the canonical citation source.
Cite the study and this repository as:

> Bernal Suárez, A. (2026). *Held but not heeded: causal dissociation of verbal acknowledgment and
> diagnostic behavior in a language model* [preprint, in preparation]. Code, data, and preregistrations:
> https://github.com/bernalsuarezantonio-bit/workspace-program

The machine-readable `CITATION.cff` is temporarily disabled (kept as `CITATION.cff.disabled`) and will
be re-enabled with the real DOI once the preprint is published.
