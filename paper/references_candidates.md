# Reference candidates — W3b bibliography expansion

**Task:** search and verify candidate references (not to draft text, not to decide inclusion). Every entry below was checked against a **primary source** on the web (arXiv abstract page, publisher DOI, journal/ACL/JMLR page); authors, exact title, year, and identifier transcribed from the URL shown. **Rule applied: nothing recorded from memory — an item appears here only if its identifier was seen in a real tool result.** 2025–2026 items are beyond the assistant's training and were confirmed by search only.

**Legend:** ★ = imprescindible (highly-cited / seminal / directly load-bearing) · ○ = optional/supporting. Selection is the PI's; ★ is a recommendation, not a decision. **Hook** = Intro / Discusión-implicaciones / Métodos / Lecciones-de-instrumento.

The 8 references already in the manuscript ([1] Turpin, [2] Lanham, [4] Gurnee et al. J-lens, [5] Turner ActAdd, [6] Zou RepE, [7] Kumaran confidence, [8] Hacking 1995, [9] Neuronpedia lens) are **not** repeated here; these are additions.

---

## Área 1 — Faithfulness de chain-of-thought / self-explanation (más allá de Turpin y Lanham)

- ★ **Radhakrishnan, A., et al. (2023). Question Decomposition Improves the Faithfulness of Model-Generated Reasoning.** arXiv:2307.11768 · https://arxiv.org/abs/2307.11768 — *directly shows a method that raises CoT faithfulness; the constructive counterpart to Turpin/Lanham's negative results.* **Hook:** Intro (§1 faithfulness).
- ★ **Lyu, Q., et al. (2023). Faithful Chain-of-Thought Reasoning.** arXiv:2301.13379 · https://arxiv.org/abs/2301.13379 — *faithful-by-construction reasoning (translate→solve); frames the "window vs wallpaper" question.* **Hook:** Intro.
- ○ **Bentham, O., Stringham, N., et al. (2024). Chain-of-Thought Unfaithfulness as Disguised Accuracy.** arXiv:2402.14897 · https://arxiv.org/abs/2402.14897 — *critiques Lanham's faithfulness metric (correlates with accuracy); relevant caveat when we cite faithfulness measures.* **Hook:** Discusión §5.1 / Lecciones-de-instrumento.
- ○ **Arcuschin, I., Janiak, J., Krzyzanowski, R., Rajamanoharan, S., Nanda, N., Conmy, A. (2025). Chain-of-Thought Reasoning In The Wild Is Not Always Faithful.** arXiv:2503.08679 · https://arxiv.org/abs/2503.08679 — *unfaithfulness on natural, non-adversarial prompts (up to 13%); strengthens "saying ≠ doing" without artificial bias.* **Hook:** Intro / Discusión §5.1.

## Área 2 — Sicofancia y complacencia (incl. RLHF como causa)

- ★ **Sharma, M., Tong, M., Korbak, T., … Perez, E. (2023). Towards Understanding Sycophancy in Language Models.** arXiv:2310.13548 (ICLR 2024) · https://arxiv.org/abs/2310.13548 — *traces sycophancy to human-preference/RLHF data; the seminal causal-attribution paper.* **Hook:** Intro (§1 "pure sycophancy") / Discusión §5.3 (RLHF).
- ★ **Perez, E., et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations.** arXiv:2212.09251 (Findings of ACL 2023) · https://arxiv.org/abs/2212.09251 — *documents sycophancy as an inverse-scaling RLHF behavior; also a model-written-eval methodology reference.* **Hook:** Intro / Métodos.
- ○ **Wei, J., Huang, D., Lu, Y., Zhou, D., Le, Q. V. (2023). Simple synthetic data reduces sycophancy in large language models.** arXiv:2308.03958 · https://arxiv.org/abs/2308.03958 — *scaling+instruction-tuning increase sycophancy; a lightweight fix; supports the "train behavior directly" argument.* **Hook:** Discusión §5.3 (control).

## Área 3 — Alucinación, calibración y expresión de incertidumbre

- ★ **Ji, Z., Lee, N., Frieske, R., … Fung, P. (2023). Survey of Hallucination in Natural Language Generation.** arXiv:2202.03629 · ACM Computing Surveys 55(12) · https://arxiv.org/abs/2202.03629 — *the standard hallucination survey; anchors the fabrication framing.* **Hook:** Intro / Discusión.
- ○ **Kadavath, S., Conerly, T., Askell, A., … (2022). Language Models (Mostly) Know What They Know.** arXiv:2207.05221 · https://arxiv.org/abs/2207.05221 — *models are often calibrated and can predict their own correctness; contrast case for "knowledge that is represented."* **Hook:** Discusión §5.1.
- ○ **Lin, S., Hilton, J., Evans, O. (2022). Teaching Models to Express Their Uncertainty in Words.** arXiv:2205.14334 · https://arxiv.org/abs/2205.14334 — *verbalized calibration; relevant to acknowledgment-as-channel.* **Hook:** Discusión / Lecciones-de-instrumento.

## Área 4 — LLM-as-judge: validación, sesgos, acuerdo con humanos

- ★ **Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** arXiv:2306.05685 (NeurIPS 2023 D&B) · https://arxiv.org/abs/2306.05685 — *establishes LLM-judge ↔ human agreement (~80–85%); the reference our judge-validation stands on.* **Hook:** Métodos (§2.1 validation).
- ★ **Panickssery, A., Bowman, S. R., Feng, S. (2024). LLM Evaluators Recognize and Favor Their Own Generations.** arXiv:2404.13076 (NeurIPS 2024) · https://arxiv.org/abs/2404.13076 — *self-preference bias; justifies our third-family judge + fourth-family co-rater design.* **Hook:** Métodos / Lecciones-de-instrumento (#2).
- ○ **Wang, P., Li, L., Chen, L., et al. (2023). Large Language Models are not Fair Evaluators.** arXiv:2305.17926 (ACL 2024) · https://arxiv.org/abs/2305.17926 — *position/order bias in LLM judging; supports blinding and validation.* **Hook:** Métodos / Lecciones-de-instrumento.

## Área 5 — Activation steering / representation engineering / ablación causal (métodos y límites)

- ★ **Li, K., Patel, O., et al. (2023). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.** arXiv:2306.03341 · https://arxiv.org/abs/2306.03341 — *canonical activation-editing intervention; direct precedent for our Study 3 steering.* **Hook:** Intro / Métodos (Study 3).
- ★ **Rimsky, N., Gabrieli, N., et al. (2023). Steering Llama 2 via Contrastive Activation Addition.** arXiv:2312.06681 (ACL 2024) · https://arxiv.org/abs/2312.06681 — *contrastive steering-vector method; the additive-steering family whose limits we report.* **Hook:** Métodos / Lecciones-de-instrumento (#7).
- ○ **Meng, K., Bau, D., et al. (2022). Locating and Editing Factual Associations in GPT (ROME).** arXiv:2202.05262 (NeurIPS 2022) · https://arxiv.org/abs/2202.05262 — *causal localization + editing; precedent for direction-level causal tests.* **Hook:** Intro / Métodos.

## Área 6 — Introspección y auto-reporte en LLMs *(área DELGADA — ver nota)*

- ★ **Binder, F. J., Chua, J., Korbak, T., … Evans, O. (2024). Looking Inward: Language Models Can Learn About Themselves by Introspection.** arXiv:2410.13787 · https://arxiv.org/abs/2410.13787 — *directly asks whether models report internal states; the closest prior to our "window or wallpaper" framing.* **Hook:** Intro (§1) / Discusión.
- ○ **Ji-An, L., Xiong, H.-D., Wilson, R. C., Mattar, M. G., Benna, M. K. (2025). Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations.** arXiv:2505.13763 · https://arxiv.org/abs/2505.13763 — *low-dimensional "metacognitive space"; models monitor only a subset of activations — resonates with our weak-but-real workspace trace.* **Hook:** Discusión (§5.1).

## Área 7 — Entrenamiento de rechazo, disclaimers y efecto sobre la conducta

- ★ **Arditi, A., Obeso, O., Syed, A., Paleka, D., Rimsky, N., Gurnee, W., Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction.** arXiv:2406.11717 · https://arxiv.org/abs/2406.11717 — *a safety behavior mediated by one direction; the cleanest precedent for "a behavior lives on a single ablatable direction."* **Hook:** Intro / Discusión §5.3 / Métodos.
- ○ **Röttger, P., Kirk, H. R., Vidgen, B., Attanasio, G., Bianchi, F., Hovy, D. (2023). XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in LLMs.** arXiv:2308.01263 (NAACL 2024) · https://arxiv.org/abs/2308.01263 — *over-refusal from disclaimer-like training; supports "caveat production ≠ behavior."* **Hook:** Discusión §5.3.
- ○ **Cui, J., et al. (2024). OR-Bench: An Over-Refusal Benchmark for Large Language Models.** arXiv:2405.20947 · https://arxiv.org/abs/2405.20947 — *scale of over-refusal; same argument.* **Hook:** Discusión §5.3.

## Área 8 — Confianza del usuario en avisos/advertencias de IA (factor humano)

- ★ **Vasconcelos, H., Jörke, M., Grunde-McLaughlin, M., Gerstenberg, T., Bernstein, M. S., Krishna, R. (2023). Explanations Can Reduce Overreliance on AI Systems During Decision-Making.** arXiv:2212.06823 · Proc. ACM HCI (CSCW) · DOI 10.1145/3579605 · https://dl.acm.org/doi/10.1145/3579605 — *overreliance persists despite explanations; grounds "users treat model acknowledgments as evidence."* **Hook:** Discusión §5.3 (implications).
- ○ **Buçinca, Z., Malaya, M. B., Gajos, K. Z. (2021). To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI in AI-Assisted Decision-Making.** Proc. ACM HCI 5(CSCW1) · https://www.eecs.harvard.edu/~kgajos/papers/2021/bucinca2021trust.shtml — *overreliance and its mitigation; human-factors backing.* **Hook:** Discusión §5.3.
- ○ **Bansal, G., Wu, T., Zhou, J., … Weld, D. (2021). Does the Whole Exceed its Parts? The Effect of AI Explanations on Complementary Team Performance.** arXiv:2006.14779 · CHI '21 · DOI 10.1145/3411764.3445717 · https://arxiv.org/abs/2006.14779 — *AI explanations don't reliably improve human+AI outcomes; trust-in-warnings evidence.* **Hook:** Discusión §5.3.

## Área 9 — Prerregistro, reproducibilidad y crisis de replicación en ML/NLP

- ★ **Card, D., Henderson, P., Khandelwal, U., Jia, R., Mahowald, K., Jurafsky, D. (2020). With Little Power Comes Great Responsibility.** arXiv:2010.06595 · EMNLP 2020 · https://aclanthology.org/2020.emnlp-main.745/ — *underpowered NLP experiments are common; directly justifies our power analysis.* **Hook:** §7 Transparencia / Métodos (power).
- ○ **Pineau, J., Vincent-Lamarre, P., et al. (2021). Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program).** arXiv:2003.12206 · JMLR 22, 20-303 · https://www.jmlr.org/papers/v22/20-303.html — *the field-standard reproducibility reference; backs our chain-of-custody protocol.* **Hook:** §7 / Métodos.
- ○ **Gorman, K., Bedrick, S. (2019). We Need to Talk about Standard Splits.** ACL 2019 · https://aclanthology.org/P19-1267/ — *rankings don't survive random splits; supports our multi-vignette / preregistered-split discipline.* **Hook:** §7 / Métodos.

## Área 10 — LLMs en salud mental / contextos clínicos

- ★ **Moore, J., Grabb, D., Agnew, W., Klyman, K., Chancellor, S., Ong, D. C., Haber, N. (2025). Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers.** ACM FAccT 2025, pp. 599–627 · DOI 10.1145/3715275.3732039 · https://dl.acm.org/doi/10.1145/3715275.3732039 (also arXiv:2504.18412, https://arxiv.org/abs/2504.18412) — *LLMs mishandle serious mental-health cases; the strongest "clinical-opinion-requested is the deployment default" support.* **Hook:** Intro (motivation) / Discusión §5.3.
- ★ **Stade, E. C., Stirman, S. W., Ungar, L. H., … Eichstaedt, J. C. (2024). Large language models could change the future of behavioral healthcare: a proposal for responsible development and evaluation.** npj Mental Health Research 3:12 · DOI 10.1038/s44184-024-00056-z · https://www.nature.com/articles/s44184-024-00056-z — *responsible-evaluation roadmap for clinical LLMs; frames the stakes.* **Hook:** Intro / Discusión §5.3.
- ○ **"The opportunities and risks of large language models in mental health." arXiv:2403.14814** · https://arxiv.org/abs/2403.14814 — *⚠ PARCIAL: título + id confirmados por el listado de arXiv; **lista de autores no verificada individualmente** en esta pasada. Requiere una comprobación más antes de citar.* **Hook:** Discusión §5.3.

## Área 11 — Sobrediagnóstico, expansión diagnóstica y validez del DSM

- ★ **Frances, A. (2013). Saving Normal: An Insider's Revolt Against Out-of-Control Psychiatric Diagnosis, DSM-5, Big Pharma, and the Medicalization of Ordinary Life.** William Morrow · ISBN 9780062229250 · https://www.amazon.com/Saving-Normal-Out-Control-Medicalization/dp/0062229257 — *DSM-5 diagnostic inflation from the DSM-IV task-force chair; the anchor for "an invented disorder with a fitting description is functionally a disorder."* **Hook:** Intro (reification motivation) / Discusión §5.3.
- ○ **Moynihan, R., Doust, J., Henry, D. (2012). Preventing overdiagnosis: how to stop harming the healthy.** BMJ 344 (2012) · PubMed 22645185 · https://pubmed.ncbi.nlm.nih.gov/22645185/ — *overdiagnosis as harm; the wider-definition mechanism.* **Hook:** Intro / Discusión §5.3.
- ○ **Insel, T., Cuthbert, B., Garvey, M., et al. (2010). Research Domain Criteria (RDoC): Toward a New Classification Framework for Research on Mental Disorders.** Am. J. Psychiatry 167(7):748–751 · DOI 10.1176/appi.ajp.2010.09091379 · https://psychiatryonline.org/doi/abs/10.1176/appi.ajp.2010.09091379 — *institutional response to DSM validity concerns; category-validity context.* **Hook:** Discusión §5.3.
- ○ **Batstra, L., Frances, A. (2012). Diagnostic Inflation: Causes and a Suggested Cure.** J. Nervous and Mental Disease 200(6) · https://journals.lww.com/jonmd/fulltext/2012/06000/Diagnostic_Inflation__Causes_and_a_Suggested_Cure.4.aspx — *mechanisms of diagnostic expansion + a proposed brake.* **Hook:** Discusión §5.3.

## Área 12 — Hacking / filosofía de la clasificación (más allá de 1995)

- ★ **Hacking, I. (2007). Kinds of People: Moving Targets.** British Academy Lecture. Proceedings of the British Academy 151, 285–318 · DOI 10.5871/bacad/9780197264249.003.0010 · https://doi.org/10.5871/bacad/9780197264249.003.0010 — *the mature statement of looping effects / making up people; upgrades the lone 1995 citation.* **Hook:** Intro (§1 Hacking) / Discusión §5.3.
- ○ **Hacking, I. (1999). The Social Construction of What?** Harvard University Press · ISBN 9780674004122 · https://www.cambridge.org/core/journals/canadian-journal-of-philosophy/article/abs/ian-hacking-the-social-construction-of-what-cambridge-ma-harvard-university-press-1999-pp-x-261/5CC1DD9DA5029C0622CE395CFA517D78 — *interactive vs indifferent kinds; the constructivism framing.* **Hook:** Intro / Discusión.
- ○ **Hacking, I. (2006). Making Up People.** London Review of Books 28(16) · https://www.lrb.co.uk/the-paper/v28/n16/ian-hacking/making-up-people — *accessible statement of the making-up-people thesis (concept orig. 1986; original venue not verified here).* **Hook:** Intro / Discusión.

---

## Resumen de verificación

**Confirmadas contra fuente primaria: 33** (todas con identificador visto en resultado real). **PARCIAL: 1** (Área 10, arXiv:2403.14814 — id/título por listado arXiv, autores sin verificar individualmente). **NO ENCONTRADA en esta pasada: 0** (solo se registró lo confirmado; los ítems no verificables no se anotaron).

**Por área (confirmadas / parciales):** 1: 4/0 · 2: 3/0 · 3: 3/0 · 4: 3/0 · 5: 3/0 · 6: 2/0 · 7: 3/0 · 8: 3/0 · 9: 3/0 · 10: 2/1 · 11: 4/0 · 12: 3/0.

**Áreas pobres en literatura sólida (información, como pediste):**
- **Área 6 (introspección/auto-reporte en LLMs)** es la más delgada: es un tema joven; solo hay ~2 trabajos claramente citables y bien establecidos (Binder 2024; Ji-An 2025). El resto que aparece es 2025–2026 muy reciente, poco citado y de calidad dispar. Si quieres una tercera, habría que aceptar un preprint reciente sin track record.
- **Área 10 (LLMs en salud mental)** tiene mucho volumen pero calidad muy desigual; los dos sólidos (Moore FAccT 2025; Stade npj 2024) son claros, pero el tercero (2403.14814) quedó PARCIAL y hay que verificar autores antes de citarlo.
- **Área 8 (confianza del usuario en avisos)** es HCI, no ML: literatura sólida pero fuera de arXiv en parte (ACM DL / DOIs); las tres que puse están bien establecidas.

**Notas de método:** para los ítems de arXiv verifiqué la página del abstract (o el listado arxiv.org/abs con corroboración en dblp/ACL/JMLR/Semantic Scholar); para libros y papers de journal usé el DOI del editor / página institucional. Autores de [Área 10 · 2403.14814] no se transcribieron por no haberlos visto en fuente primaria — marcado PARCIAL en vez de completarlo de memoria.
