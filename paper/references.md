# References — verification record

Every `[verify]` / bracketed reference in `preprint_held_but_not_heeded_v1.md` checked against primary sources on the web (2026-07-24). **Rule applied: anything not confirmed against an original is marked `NO ENCONTRADA` and is never completed from memory.** Author lists, titles, and identifiers below are transcribed from the cited source URL, not recalled.

**Status legend:** ✅ CONFIRMADA (authors + title + identifier all verified) · ⚠️ PARCIAL (some fields verified, at least one not) · ❌ NO ENCONTRADA (no confirmable primary source).

| # | status | verified identifier |
|---|---|---|
| [1] | ✅ CONFIRMADA | arXiv:2305.04388 |
| [2] | ✅ CONFIRMADA | arXiv:2307.13702 |
| [3] | ❌ NO ENCONTRADA | — |
| [4] | ✅ CONFIRMADA | Transformer Circuits Thread, 2026-07-06 · arXiv:2607.15495 |
| [5] | ✅ CONFIRMADA | arXiv:2308.10248 |
| [6] | ✅ CONFIRMADA | arXiv:2310.01405 |
| [7] | ✅ CONFIRMADA | arXiv:2603.22161 |
| [8] | ✅ CONFIRMADA | Oxford: Clarendon Press, 1995 |
| [9] | ⚠️ PARCIAL | repo `neuronpedia/jacobian-lens` confirmed; revision label "qwen-n1000" NOT confirmed |

---

## [1] ✅ CONFIRMADA

- **Authors (verified):** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman.
- **Title (verified, exact):** *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*.
- **Identifier (verified):** arXiv:2305.04388 (submitted 2023-05-07; NeurIPS 2023).
- **Source:** https://arxiv.org/abs/2305.04388
- **Manuscript entry as written:** correct; no change needed.

## [2] ✅ CONFIRMADA

- **Authors (verified):** Tamera Lanham, Anna Chen, Ansh Radhakrishnan, … Jan Brauner, Samuel R. Bowman, Ethan Perez (28 authors; lead **Tamera Lanham**). The manuscript's "Lanham, T., et al." is correct.
- **Title (verified, exact):** *Measuring Faithfulness in Chain-of-Thought Reasoning*.
- **Identifier (verified):** arXiv:2307.13702 (2023-07-17). **This was the field flagged `[verify citation details]` — now supplied.**
- **Source:** https://arxiv.org/abs/2307.13702 · https://dblp.org/rec/journals/corr/abs-2307-13702.html

## [3] ❌ NO ENCONTRADA

- **Manuscript placeholder:** *"[2026 source of 'verbalization is a witness' formulation — verify and cite exactly]"* — used in the introduction for the phrase *verbalization … "a witness, not the phenomenon"*.
- **Outcome:** no primary source with confirmable authors/title/identifier could be located for that exact formulation. Web search surfaced only thematically-adjacent 2026 evaluation-awareness papers (e.g. arXiv:2605.05835, 2605.23055, 2606.29196); none is demonstrably the source of the quoted wording, and the search engine's attribution was self-retracted. A partial-string match ("witness … not the phenomenon itself") was reported but could not be tied to a specific paper.
- **Action required (author):** either locate the exact source and cite it, or rewrite the sentence to drop the attributed quotation. **Not filled from memory.**

## [4] ✅ CONFIRMADA

- **Authors (verified from the primary page):** Wes Gurnee\*, Nicholas Sofroniew\*, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, Jack Lindsey\*† (\*core contributor; †correspondence). **This replaces the manuscript's placeholder "[Anthropic interpretability team]".**
- **Title (verified, exact):** *Verbalizable Representations Form a Global Workspace in Language Models*.
- **Identifier (verified):** Transformer Circuits Thread, published **2026-07-06** (Anthropic); also arXiv:2607.15495.
- **Source:** https://transformer-circuits.pub/2026/workspace/index.html · https://arxiv.org/abs/2607.15495

## [5] ✅ CONFIRMADA

- **Authors (verified):** Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Ulisse Mini, Monte MacDiarmid.
- **Title (verified, exact):** *Activation Addition: Steering Language Models Without Optimization*.
- **Identifier (verified):** arXiv:2308.10248 (2023).
- **Source:** https://arxiv.org/abs/2308.10248
- **Note:** manuscript gave short title "Activation Addition"; full title supplied above.

## [6] ✅ CONFIRMADA

- **Authors (verified):** Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, et al. (lead **Andy Zou**).
- **Title (verified, exact):** *Representation Engineering: A Top-Down Approach to AI Transparency*.
- **Identifier (verified):** arXiv:2310.01405 (2023-10). Code: github.com/andyzoujm/representation-engineering.
- **Source:** https://arxiv.org/abs/2310.01405

## [7] ✅ CONFIRMADA

- **Authors (verified, two independent sources):** Dharshan Kumaran, Nathaniel Daw, Simon Osindero, Petar Veličković, Viorica Patraucean (Google DeepMind). **This replaces the manuscript's placeholder "[Authors]".**
- **Title (verified, exact):** *Causal Evidence that Language Models use Confidence to Drive Behavior*.
- **Identifier (verified):** arXiv:2603.22161 (submitted 2026-03-23; v2 2026-05-19).
- **Source:** https://arxiv.org/abs/2603.22161

## [8] ✅ CONFIRMADA

- **Author (verified):** Ian Hacking.
- **Title (verified, exact):** *The looping effects of human kinds*. In D. Sperber, D. Premack & A. J. Premack (Eds.), *Causal Cognition: A Multidisciplinary Debate*.
- **Identifier (verified):** Oxford: Clarendon Press, 1995 (chapter, pp. 351–383).
- **Source:** https://academic.oup.com/book/26284/chapter/194529638 · https://philpapers.org/rec/HACTLE-2
- **Manuscript entry as written:** correct.

## [9] ⚠️ PARCIAL

- **Confirmed:** the repository **`neuronpedia/jacobian-lens`** exists on HuggingFace and hosts pre-fitted Jacobian lenses, **including a `qwen2.5-7b-it` lens** (fitted at the paper's n=1000 scale over wikitext, per the release notes). This matches the project's own pin (`neuronpedia/jacobian-lens`, lens `.pt` for `qwen2.5-7b-it`).
- **NOT confirmed:** a named revision/release literally called **"qwen-n1000"**. The HF repo page shows only the `main` branch; no release tag `qwen-n1000` was found. The project's committed provenance pins the concrete HF **revision `16a01f3`** (with a recorded lens `.pt` sha256), which is verifiable, but that is our internal git-revision pin, **not** a confirmed upstream release identifier named "qwen-n1000".
- **Source:** https://huggingface.co/neuronpedia/jacobian-lens · https://huggingface.co/neuronpedia/jacobian-lens/tree/main
- **Action required (author):** cite the repository with the **actual pinned revision** (e.g. the HF commit hash used, `16a01f3` per project provenance) rather than the unconfirmed label "qwen-n1000", or confirm that release name directly with Neuronpedia. **Not filled from memory.**

---

## Verification method

- arXiv items: fetched the abstract page (or cross-checked via search + dblp) for exact title, author list, and identifier.
- [4]: author list read from the primary Transformer Circuits page, not a search snippet.
- [7]: author list corroborated by two independent sources (arXiv abstract fetch + search) before recording.
- [3], [9]: multiple distinct search formulations attempted; the unconfirmed fields are named explicitly and left blank rather than guessed.

*No citation is recorded here unless its stated fields were seen on the linked source. Bracketed placeholders that could not be resolved remain unresolved by design.*
