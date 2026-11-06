# Interpretable-by-Design / Concept-Bottleneck CNNs & LMs — Literature Survey
**Prepared:** 2026-10-15 · **Method:** web search (Firecrawl keyless) + arXiv abstract-page verification + AAAI/ACL/PMLR/Nature pages. Every cited ID was opened on arXiv/ACL/AAAI/PMLR and its title+authors+year confirmed unless marked *[unverified venue]*.
**Scope honesty:** this is the arXiv/ACL/AAAI/PMLR-verified record, not an exhaustive crawl. Two search-result IDs could not be re-opened (2603.05629, 2606.30498) and are marked.

---

## A. Core Concept-Bottleneck Models (CBM) line

1. **Koh, Nguyen, Tang, Mussmann, Pierson, Kim, Liang (2020). Concept Bottleneck Models.** ICML 2020, PMLR 119:5338–5348; arXiv:2007.04612.
   *Claim:* Train `x→concepts→label` with annotated concepts; by construction you can intervene on predicted concepts; on x-ray/bird tasks accuracy is competitive with end-to-end nets and test-time concept correction raises accuracy.

2. **Chen, Bei, Rudin (2020). Concept Whitening for Interpretable Image Recognition.** Nature Machine Intelligence 2:772–782; arXiv:2002.01650.
   *Claim:* A whitening/orthogonality module forces each latent axis to align with *one* user concept, so axes are interpretable by construction — no post-hoc analysis, and per-axis concept alignment is enforced structurally (not audited).

3. **Espinosa Zarlenga et al. (2022). Concept Embedding Models: Beyond the Accuracy–Explainability Trade-Off.** NeurIPS 2022; arXiv:2209.09056.
   *Claim:* Replace scalar concepts with learned concept *embeddings*; recovers near-black-box accuracy while keeping interventions — explicitly targets the accuracy-vs-interpretability gap left open by Koh et al.

4. **Oikarinen, Das, Nguyen, Weng (2023). Label-Free Concept Bottleneck Models.** ICLR 2023; arXiv:2304.06129.
   *Claim:* Build a CBM bottleneck from CLIP + an LLM concept set with *no* concept annotations; accuracy close to the black box and user-editable, at the cost of CLIP-defined concepts.

5. **Kim, Jung, Park, Kim, Yoon (2023). Probabilistic Concept Bottleneck Models.** ICML 2023, PMLR 202; arXiv:2306.01574.
   *Claim:* Model concept uncertainty probabilistically (PCBM) so you know *how confident* each concept is and can acknowledge uncertain concepts — improves accuracy/robustness over deterministic CBMs.

6. **Yuksekgonul, Wang, Zou (2023). Post-hoc Concept Bottleneck Models.** ICLR 2023 (Spotlight); arXiv:2205.15480.
   *Claim:* Wrap *any* trained net with a concept bottleneck using transferred/language concepts, matching black-box accuracy and enabling global concept-level edits without retraining.

7. **Knab, Steinmann, Bartelt, Kersting, Schiele, Seidl, Schlegel, Stammer (2026). What's in the Bottle? A Survey and Roadmap of Concept Bottleneck Models.** TMLR (OpenReview IF5vnqxBEW), preprint Feb 2026.
   *Claim:* Unified taxonomy + systematic review of the CBM literature, naming the field's open challenges and component choices.

---

## B. Other interpretable-by-construction architectures

8. **Chen, Li, Tao, Barnett, Rudin, Su (2019). This Looks Like That (ProtoPNet).** NeurIPS 2019; arXiv:1806.10574.
   *Claim:* Classify by comparing image parts to learned *prototypes* from training data (prototypical-part network) — a prototype/rule-based net with built-in case-based explanation.

9. **Alvarez-Melis & Jaakkola (2018). Towards Robust Interpretability with Self-Explaining Neural Networks (SENN).** NeurIPS 2018; arXiv:1806.07538.
   *Claim:* An architecture whose output is an explicit linear combination of *human-interpretable input features* with input-dependent coefficients (the "self-explaining" rubric) — the origin of "explainable by construction."

10. **Ciravegna, Barbiero, Giannini, Gori, Lió, Maggini, Melacci (2022). Logic Explained Networks.** Artificial Intelligence 103822 (2022); arXiv:2108.05149.
    *Claim:* Networks whose inputs are human-understandable *predicates* that emit First-Order-Logic formulas over those predicates as the explanation — closest formal cousin to "rules over named concepts."

11. **Granmo (2018). The Tsetlin Machine.** arXiv:1804.01508.
    *Claim:* A propositional-logic learning automaton producing conjunctive clauses (interpretable rules with positive/negative literals); inherently interpretable, competitive on classification/NLP.

12. **Bau, Zhou, Khosla, Oliva, Torralba (2017). Network Dissection.** CVPR 2017; arXiv:1704.05796.
    *Claim:* Score alignment (IoU) between each hidden *unit* and each concept in a broad concept set, labelling units that match a concept — a per-neuron "does this unit track concept X" audit (units that don't match stay unlabeled).

13. **Kim, Wattenberg, Gilmer, Cai, Wexler, Viégas, Sayres (2018). TCAV: Testing with Concept Activation Vectors.** ICML 2018, PMLR 80; arXiv:1711.11279.
    *Claim:* Measure a model's sensitivity to a user-defined concept direction (concept importance), not per-neuron concept ownership.

---

## C. Interpretable / transparent **language** models (the closest prior art)

14. **Sun, Oikarinen, Ustun, Weng (2025). Concept Bottleneck Large Language Models (CB-LLM).** ICLR 2025; arXiv:2412.07992.
    *Claim:* Transforms a pretrained LLM into an inherently interpretable LM with a **Concept Bottleneck Layer** of interpretable neurons, for both **text classification** (competitive with black-box) and **text generation** (concept detection, controlled generation, concept unlearning). ⚠️ **Single closest match to "an LM whose answer layer sees named concepts."** But: concepts are learned unsupervised, not human-defined formulas over tokens; no per-concept honesty audit + drop.

15. **Ludan et al. (2023). Interpretable-by-Design Text Understanding with Iteratively Generated Concept Bottleneck (TBM).** arXiv:2310.19660. *[venue unverified]*
    *Claim:* Text Bottleneck Models predict a sparse set of LLM-discovered concepts then a linear layer to the label; rivals few-shot GPT-4 / finetuned DeBERTa but falls short of finetuned GPT-3.5; human eval says concepts are relevant and measurement aligns with human judgment.

16. **Bhan, Choho, Moreau, Vittaut, Chesneau, Lesot (2025). Towards Achieving Concept Completeness for Textual Concept Bottleneck Models (CT-CBM).** Findings of EMNLP 2025, ACL Anthology 2025.findings-emnlp.106; arXiv:2502.11100.
    *Claim:* Unsupervised Textual CBM that *iteratively adds* concepts until the concept basis is **complete**; optimized for concept-basis completeness and concept-detection accuracy (the "completeness" analogue of an audit — but it ADDS, it does not drop).

17. **Tan, Yuan, Cheng, Li, Wang, Liu (2023). Interpreting Pretrained Language Models via Concept Bottlenecks (C3M).** arXiv:2311.05014. *[PAKDD 2024, unverified]*
    *Claim:* Combines human-annotated + machine-generated concepts to locate hidden neurons that encode task-specific concepts (e.g. "Food") in BERT-style PLMs; helps diagnose failures and improves robustness to noisy concept labels.

18. **Rajagopal, Balachandran, Hovy, Tsvetkov (2021). SelfExplain: A Self-Explaining Architecture for Neural Text Classifiers.** ACL 2021; arXiv:2103.12279.
    *Claim:* Adds a globally-interpretable concept layer (phrase-based concepts) + a local relevance layer to a text classifier; explanations judged sufficient/trustworthy by humans without hurting accuracy.

19. **ProtoryNet (2020). Interpretable Text Classification via Prototype Trajectory.** arXiv:2007.01777. *[venue unverified]*
    *Claim:* Text analogue of ProtoPNet — classifies via a trajectory over learned prototype sentences (prototype/rule-based interpretable text net).

---

## D. Evaluation standards: do concept neurons track their concept labels?

20. **Yeh, Kim, Arik, Li, Pfister, Ravikumar (2020). On Completeness-aware Concept-Based Explanations (ConceptSHAP).** NeurIPS 2020; arXiv:1910.07969.
    *Claim:* Defines **completeness** — how sufficient a concept set is as a sufficient statistic of the prediction — and a discovery method + ConceptSHAP importance that finds a *complete* concept set.

21. **Mahinpei, Clark, Lage, Doshi-Velez, Pan (2021). Promises and Pitfalls of Black-Box Concept Learning Models (concept leakage).** arXiv:2106.13314.
    *Claim:* Shows concept representations **encode information beyond the pre-defined concepts** (leakage) and that natural mitigations don't fully work — so a "concept neuron" may be tracking a hidden rival signal, not its own label.

22. **Margeloiu, Ashman, Bhatt, Chen, Jamnik, Weller (2021). Do Concept Bottleneck Models Learn as Intended?** arXiv:2105.04289.
    *Claim:* Post-hoc analysis finds CBM concepts **do not correspond to anything semantically meaningful in input space** — a direct challenge to the assumption that concept neurons track their labels.

23. **Espinosa Zarlenga, Barbiero, Shams, Kazhdan, Bhatt, Weller, Jamnik (2023). Towards Robust Metrics for Concept Representation Evaluation.** AAAI 2023; arXiv:2301.10367.
    *Claim:* Proposes robust metrics for concept *impurity/leakage* because concept-learning representations fail to fully capture their intended concept.

24. **Aysel, Cai, Prugel-Bennett (2025). Concept-Based XAI: Metrics and Benchmarks.** arXiv:2501.19271.
    *Claim:* Introduces **concept global-importance, concept-existence, and concept-location** metrics; finds that even a CBM's most important concepts are often **not present** in the input or mis-localized.

25. **Bau et al. (2017) Network Dissection** [see #12] — the de-facto per-unit "aligns-with-a-concept or not" score (IoU threshold).

26. **Enouen & Galhotra (2026). Debugging Concept Bottleneck Models through Removal and Retraining (CBDebug).** ICLR 2026; arXiv:2509.21385.
    *Claim:* A two-step **Removal + Retraining** framework in which experts/LLMs mark *spurious* concepts and the model is retrained to stop using them; improves worst-group accuracy up to 26%. Removal is driven by spuriousness feedback, **not** by a per-concept own-label test.

27. **On the Faithfulness of Post-Hoc Concept Bottleneck Models (2026).** arXiv:2606.30498. *[abstract read; venue unverified]*
    *Claim:* Argues target accuracy is the wrong success measure and shows post-hoc CBMs can be predictively accurate yet conceptually unfaithful.

28. **Leakage and Interpretability in Concept-Based Models (2025).** arXiv:2504.14094. *[search-verified title]*
    *Claim:* Studies information leakage as a barrier to faithful concept-based interpretability.

---

## E. Latent-feature alternatives & surveys

29. **Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike, Wu (2024). Scaling and Evaluating Sparse Autoencoders.** arXiv:2406.04093.
    *Claim:* k-sparse autoencoders extract interpretable features from LM activations with clean scaling laws, plus new feature-quality metrics (feature recovery, explainability, downstream sparsity). Unsupervised/emergent — not a named-concept answer layer.

30. **Templeton et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.** Anthropic technical report (May 2024). *[not peer-reviewed]*
    *Claim:* Scales SAEs to a frontier model, extracting millions of monosemantic features and steering behavior via feature clamping.

31. **Gao, Meng, Zhou, Pan (2026). Towards Intrinsic Interpretability of LLMs: A Survey.** arXiv:2604.16042.
    *Claim:* Taxonomy of intrinsic-interpretability LLM design, including a "concept alignment" paradigm and "representational decomposability."

32. **Interpreting Language Models Through Concept Descriptions: A Survey.** BlackboxNLP 2025, ACL Anthology 2025.blackboxnlp-1.8.
    *Claim:* Reviews open-vocabulary, LLM-generated concept descriptions for LLM components (moving past fixed concept sets).

33. **Selvaraj, Guo, Kong, Kot (2024). Improving Concept Alignment in Vision-Language Concept Bottleneck Models.** arXiv:2405.01825. *[ECCV 2024, unverified]*
    *Claim:* Targets the *concept-alignment* problem in VLM-CBMs — mismatch between a concept's name and what the neuron actually detects.

34. **Bouniot & Akata (2026). Rethinking Concept Bottleneck Models: From Pitfalls to Solutions.** arXiv:2603.05629. *[search-verified title/venue line; arXiv HTML failed to render]*

Supporting / classic post-hoc:
- **Ribeiro, Singh, Guestrin (2018). Anchors: High-Precision Model-Agnostic Explanations.** AAAI 2018 — if-then rules with precision guarantees.
- **Kim, Wattenberg, Gilmer, Cai, Wexler, Viégas, Sayres (2018). TCAV** [see #13].
- **Bau et al. (2017). Network Dissection** [see #12].

---

## SYNTHESIS — prior art vs. no-direct-match

### Clearly PRIOR ART (our pipeline reuses these ideas)
- **Answer layer that only sees concepts** → this is the *definition* of a CBM (Koh 2020) and, for **language**, explicitly built in **CB-LLM (Sun 2025)**, **TBM (Ludan 2023)**, **CT-CBM (Bhan 2025)**, **C3M (Tan 2023)**.
- **Named human-defined concepts as the inputs to rules** → Koh 2020 (human-specified concepts); stronger match: **LEN (Ciravegna 2022)** where inputs are human-understandable *predicates* and outputs are FOL rules.
- **Sparse rules over ≤3 concepts** → LEN, **Tsetlin Machines (Granmo 2018)** clauses, decision-rule-set learning, **SENN (Alvarez-Melis 2018)** sparse interpretable feature combination, **SelfExplain** relevance layer.
- **Concepts measured as formulas/regex over token ids ("is_a_question")** → human predicates/attributes are standard; *symbolic/regex over tokens* specifically is the LEN-style predicate approach. No paper makes the symbolic-token-regex the *entire* concept bank of a from-scratch LM that I could find.
- **Per-concept audit of "does the neuron mean its concept"** → **Network Dissection (Bau 2017)** (per-unit concept IoU), **ConceptSHAP/completeness (Yeh 2020)**, **concept-existence metric (Aysel 2025)**, **robust concept metrics (Espinosa Zarlenga 2023)**, **concept leakage (Mahinpei 2021)**, **"learn as intended" (Margeloiu 2021)**.
- **Enforcing one concept per axis** → **Concept Whitening (Chen 2020)** (structural, not audited).
- **Editing/removing concepts** → **CBDebug (Enouen 2026)**, **Post-hoc CBM (Yuksekgonul 2023)**.

### NO DIRECTLY MATCHING PUBLISHED WORK (verified gap)
- ❌ **No paper found that runs a strict per-concept audit requiring each concept neuron to correlate BEST with its own measurable label, beating all rival labels, and DROPS the concepts that fail.** The nearest are: Network Dissection (labels only units that align with *some* concept — no "beats all rivals" mutual-exclusivity test, and it doesn't gate a downstream answer layer); CBDebug (removes concepts by expert/LLM spuriousness feedback, not an intrinsic own-label test); Concept Whitening (aligns axes by construction rather than auditing and pruning). Search phrasings tried: "neuron correlates own concept label beating others", "audit removes concept if not aligned", "prune concepts by concept accuracy", etc. — no match. → **OPEN.**
- ❌ **No paper found doing this at the tiny from-scratch scale (~8M params, 13–39M tokens, 12k vocab) evaluated as next-token accuracy on dialogue** with a symbolic named-concept bottleneck. CB-LLM/CT-CBM/TBM all build on pretrained/LLM backbones at much larger scale.
- ⚠️ **Weakly adjacent:** CB-LLM (2412.07992) is the single closest — an LM with a concept-bottleneck answer layer enabling concept detection/steering/unlearning — but it uses *unsupervised learned* concepts and does **not** prune by a per-concept honesty test. CT-CBM (2502.11100) is the closest *audit-flavored* text work, but it optimizes **completeness by ADDING** concepts, not by dropping ones that fail.

### Honesty verdict (novelty-check rubric)
**Minor extension / novel combination, not a clean replication and not a claim no paper touches.**
Each *component* is prior art (CBM bottleneck ✓, LEN predicates ✓, sparse rules ✓, per-unit concept alignment ✓, concept removal ✓). The specific *combination* — a tiny from-scratch dialogue LM whose answer layer reads only ~40 hand-named symbolic (token-regex/support) concepts through ≤3-concept sparse rules, with a strict per-concept honesty audit that **drops** concepts failing to beat rival labels — is not matched by any verified paper. State it as "components recombine known techniques; the drop-on-misalignment audit applied to a symbolic-concept LM bottleneck appears unpublished," not as a wholesale breakthrough.

### What I could NOT find / could not fully verify
- No peer-reviewed paper that drops concepts purely on a per-concept own-label-beats-rivals criterion. (OPEN.)
- MLP/rule *distillation* from nets (decision-list/rule extraction) exists in vision/tabular; I did **not** find a canonical, verified text-LM version with a named-concept bottleneck to cite beyond the LEN/Tsetlin line — treat as a gap I didn't fully cover rather than a firm "doesn't exist."
- Venues marked *[unverified]* (TBM, C3M, ProtoryNet, 2405.01825, 2603.05629, 2606.30498) were confirmed on arXiv/search but I did not open their conference records.
