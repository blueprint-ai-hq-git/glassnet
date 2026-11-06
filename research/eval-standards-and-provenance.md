# How the field evaluates interpretability claims, audits named concepts, and standardises data provenance and reporting

Track: eval standards and provenance (4-track research batch).
Written: 2026-10-16 (IST). Language: English, plain ASCII.
Status: this is the written deliverable for the interrupted track (previous attempt produced no file).

## Method and honesty statement

- Every citation below was checked this session against a live source: arXiv abstract pages,
  arXiv HTML full text, ar5iv renderings of pre-2023 papers, official project docs, or the
  publisher page. The arXiv API was rate-limited (HTTP 429) for part of the session, so
  verification was done by reading pages in a browser instead of the API.
- Each citation carries the identifier it was verified under. Where a paper's own arXiv page does
  not state a venue (many older pages have an empty "comments" field), the venue is marked
  UNVERIFIED rather than asserted. Papers whose existence I could not confirm are not cited.
- Claims about what a paper does come from that paper's abstract or its own full text, quoted or
  paraphrased within the limits of what I actually read. Where I only verified a title, I say so.
- Labels used throughout: ESTABLISHED (multiple independent verified works agree), DEBATED
  (verified works exist but conflict or the support is thin), OPEN / NOT FOUND (no paper found that
  does the thing), UNVERIFIED (could not confirm this session).
- Coverage limit: this is a focused review of the named works in the task plus directly adjacent
  verified work. It is not an exhaustive crawl of the explainability or data-governance literature.

---

# PART 1 - Metrics, with what each one measures and its known limitation

| # | Metric / paper | What it exactly measures | Known limitation |
|---|---|---|---|
| 1.1 | **TCAV** - Kim, Wattenberg, Gilmer, Cai, Wexler, Viegas, Sayres, "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors", arXiv 1711.11279 (v1 2017-11-30, v5 2018-06-07). ICML 2018 (venue UNVERIFIED on the arXiv page) | Learns a Concept Activation Vector: the normal to a linear boundary separating the layer activations of concept examples from non-concept (random) examples. The TCAV score aggregates directional derivatives of the class logit along that direction over inputs of the class: "quantify the degree to which a user-defined concept is important to a classification result" (e.g. sensitivity of "zebra" to "stripes"). A permutation test against random CAVs gates significance. | The score is built on a **discontinuous indicator function**, which makes it unstable across samples - this is the specific defect identified and analysed by alpha-TCAV (1.7). The CAV is linear, so it assumes the concept is linearly separable in that layer's activation space; and the score is sensitivity of the prediction to a concept *direction*, not evidence that the concept is *present* in the input. The user also supplies the concept set and the non-concept/random example set. I could NOT verify a single canonical paper that isolates "the result depends on which negative examples you chose" as its own contribution. |
| 1.2 | **Concept leakage** - Mahinpei, Clark, Lage, Doshi-Velez, Pan, "Promises and Pitfalls of Black-Box Concept Learning Models", arXiv 2106.13314 (2021-06-24) | Demonstrates that the concept representations learned by concept-learning models "encode information beyond the pre-defined concepts", i.e. the concept layer carries extra, undeclared information into the downstream prediction, and that "natural mitigation strategies do not fully work", rendering the interpretation of the downstream prediction misleading. It also describes the mechanism of the leakage. | It is a demonstration plus mechanism, not a standardised metric with a threshold: there is no per-concept score, no cut-off, and no rule for what to do with a leaking concept. The mitigation it offers is partial by its own statement. So it shows the failure mode exists; it does not give you a gate. |
| 1.3 | **Do CBMs learn as intended?** - Margeloiu, Ashman, Bhatt, Chen, Jamnik, Weller, arXiv 2105.04289 (2021-05-10). The arXiv comment field states: "Accepted at ICLR 2021 Workshop on Responsible AI" | Uses post-hoc interpretability methods to test whether the pre-specified concepts correspond to anything semantically meaningful in **input space**, for models whose stated desiderata are interpretability, predictability and intervenability. Finding: concepts "do not correspond to anything semantically meaningful in input space", questioning the usefulness of CBMs in their current form. | A negative audit result, not a metric and not a remedy: no threshold, no replacement concept set, no gate. It audits the *class* of models, and it leans on post-hoc interpretability tools whose own reliability is contested (see 1.6 and 4). |
| 1.4 | **Concept completeness + ConceptSHAP** - Yeh, Kim, Arik, Li, Pfister, Ravikumar, "On Completeness-aware Concept-Based Explanations in Deep Neural Networks", arXiv 1910.07969 (v1 2019-10-17, v6 2022-02-07). Venue UNVERIFIED on the arXiv page (commonly cited as NeurIPS 2020) | Defines **completeness**: how sufficient a given set of concepts is to explain the model's prediction behaviour, under the stated assumption that complete concept scores are sufficient statistics of the model prediction. Adds a concept *discovery* method that infers a concept set that is additionally encouraged to be interpretable, and **ConceptSHAP**, a game-theoretic (Shapley) importance score per discovered concept, aggregated over sets. Evaluated with proposed metrics and user studies, including on a synthetic dataset. | Completeness is defined **relative to the model and to a predefined concept vocabulary**, not against an external ground-truth label for each concept; the Shapley computation is expensive as the concept set grows; and the whole notion rests on the sufficient-statistic assumption the authors state explicitly. A high-completeness concept set can still consist of concepts the model does not actually use in the way their names suggest (that is exactly what 1.2 and 1.3 show). |
| 1.5 | **ERASER: comprehensiveness and sufficiency** - DeYoung, Jain, Rajani, Lehman, Xiong, Koh, Tan, Wallace, "ERASER: A Benchmark to Evaluate Rationalized NLP Models", arXiv 1911.03429 (2019-11-08). Venue UNVERIFIED on the arXiv page (commonly cited as ACL 2020) | Faithfulness metrics for token rationales, motivated by earlier work. **Comprehensiveness**: "were all features needed to make a prediction selected?" - operationalised by building a contrast example that is the input with the predicted rationale tokens removed, then measuring the resulting drop in the model's predicted probability for the predicted class. **Sufficiency**: "do the extracted rationales contain enough signal to come to a disposition?" - the model should reach a similar disposition using only the rationale tokens. AOPC variants sweep the top 1/5/10/20/50 percent of tokens by importance score, and the paper reports a random-score reference row. Rationale plausibility is measured separately, by human agreement. | The paper itself states "the best way to measure rationale faithfulness is an open question" and that the proposed metrics are "simple". Comprehensive (exhaustive) rationales exist in the test set for only some corpora, so recall/evaluation of comprehensiveness is only noisy elsewhere, by the paper's own account. The metrics are functions of the model under test (they measure the model's reaction to token removal), so they are relative to that model and to the removal protocol. ICE (arXiv 2603.18579, 2026) is a later work whose title claims statistical grounding is needed for intervention-based explanation evaluation - I verified only its title and existence, not the details of its argument. |
| 1.6 | **Plausibility vs faithfulness/correctness** - Jacovi & Goldberg, "Towards Faithfully Interpretable NLP Systems: How should we define and evaluate faithfulness?", arXiv 2004.03685 (2020-04-07). Venue UNVERIFIED on the arXiv page (commonly cited as ACL 2020) | An explicit separation of criteria, quoted from the paper: "'Plausibility' refers to how convincing the interpretation is to humans, while 'faithfulness' refers to how accurately it reflects the true reasoning process of the model." It surveys faithfulness evaluation, groups current approaches around three assumptions, gives guidelines on how evaluations should and should not be done, and argues the binary notion of faithfulness is an unrealistic bar and should be replaced by a graded one. | It is an opinion/position paper: it defines the criterion and the failure mode (satisfying one property without the other) but supplies no metric, no threshold and no protocol. Its practical consequence is a prohibition, not an instrument: human plausibility agreement cannot be reported as evidence of faithfulness. The exact formalisation of the criteria is explicitly left "as a central challenge to the community". |
| 1.7 | **alpha-TCAV** - Schnoor, Said, Tiomoko, Samek, Jung, "alpha-TCAV: A Unified Framework for Testing with Concept Activation Vectors", arXiv 2605.15688 (v1 2026-05-15, v2 2026-10-12) | Analyses the stochastic nature of CAVs and identifies "a fundamental flaw in the standard TCAV score, its instability by using a discontinuous indicator function". Introduces a parameterised unified framework replacing the discontinuous indicator with a smooth function, recovering TCAV as a limiting case. Under stated realistic assumptions it characterises the score distributions and derives principled parameter choices: one matches the mean of Multi-TCAV with no larger variance using a single CAV; the other yields a calibrated probability that the concept positively influences the prediction. | A refinement of the TCAV estimator, so it inherits TCAV's structural dependencies: a user-supplied concept set, a user-supplied negative/random example set, and a linear CAV. It improves the stability and interpretability of the number; it does not add an independent label for the concept, and it does not test whether the concept is present in the input. Its guarantees hold "under realistic assumptions" whose reach I did not verify in full. |

Cross-cutting Part 1 verdict:

- **ESTABLISHED**: concept-based explanations can be systematically unfaithful, and the field has
  both failure-mode demonstrations (leakage, 2106.13314; input-space mismatch, 2105.04289) and
  formal criteria for separating plausibility from faithfulness (2004.03685). It is also
  ESTABLISHED that explanation quality is measured along more than one axis (faithfulness vs
  plausibility vs completeness vs sufficiency).
- **DEBATED**: which single metric is the right one. Completeness (set-level, model-relative),
  sufficiency/comprehensiveness (model-relative, removal-based), TCAV (sensitivity to a direction)
  and identifiability (probe accuracy against labels) each measure a different thing; no verified
  work establishes one as canonical.
- **OPEN**: a metric that certifies a concept is "the" thing its name claims. Every verified metric
  here is relative to a model, a vocabulary, or a removal protocol.

---

# PART 2 - Is there a published concept-honesty gate (audit against an independent label, then DROP failures)?

Names as given in the task, verified precisely:

| Work | Verified identity | What it DOES | What it does NOT do |
|---|---|---|---|
| **Network Dissection** | Bau, Zhou, Khosla, Oliva, Torralba, "Network Dissection: Quantifying Interpretability of Deep Visual Representations", arXiv 1704.05796 (2017-04-19). The arXiv comment states an oral presentation at CVPR 2017 | Quantifies interpretability "by evaluating the alignment between individual hidden units and a set of semantic concepts": scores every unit's alignment (IoU against a broad, independently labelled concept set spanning objects, parts, scenes, textures, materials, colours) and gives a unit a concept label only when alignment is high enough. Units with semantics are labelled; units without are left unlabelled. It also tests the hypothesis that interpretability equals random linear combinations of units, and compares networks across training regimes. | It does not edit the model or drop concepts from a claimed concept set: low-alignment units are simply not given a label (a thresholded post-hoc labelling, not an admission gate for a concept vocabulary). The vocabulary is closed and fixed by the probe dataset, so a real concept outside that vocabulary is invisible. There is no retraining, no claim-time verdict table, and it targets single units in vision CNNs, not a named concept layer. |
| **CBDebug** | Enouen & Galhotra, "Debugging Concept Bottleneck Models through Removal and Retraining", arXiv 2509.21385 (v1 2025-09-23, v2 2026-03-25). Comment: "Accepted to ICLR 2026" | A two-step framework: **Removal** - "experts use concept explanations to identify and remove any undesired concepts"; **Retraining** - CBDebug converts concept-level user feedback into sample-level auxiliary labels, then applies supervised bias mitigation and targeted augmentation so the model relies less on the removed/undesired concepts. Evaluated with real and automated expert feedback, across PIP-Net and post-hoc CBM and benchmarks with known spurious correlations. | The decision to remove is **human/expert judgement**; the "automated expert feedback" is a stand-in for that judgement, not an independent per-concept label metric. It evaluates the *downstream effect* of a removal (accuracy/generalisation under spurious correlation) rather than auditing each concept against an externally measurable ground truth. It is a debugging framework, not an admission gate that reports a concept-honesty pass/fail table. |
| **CT-CBM** | Bhan, Choho, Moreau, Vittaut, Chesneau, Lesot, "Towards Achieving Concept Completeness for Textual Concept Bottleneck Models", arXiv 2502.11100 (v1 2025-02-16, v3 2025-05-28). Note: this is a **textual** CBM, not a vision CBM | Builds concept labels unsupervised using a small language model, then **iteratively adds "important and identifiable" concepts** to the concept bottleneck layer until a completeness criterion holds. Concretely (from the v3 HTML): concept presence is predicted from the CAVs by a linear rule with threshold M_c = the median projection value across the development split; a **linear identifiability score g(c) is the F1 between those predictions and ground-truth labels on the dev split**, and the final concept score is i(c) x g(c), "ensuring that selected concepts are both discriminative for the task and reliably detectable". The bottleneck is initialised with concepts covering at least 99% of the training texts, and completeness is estimated as the smallest concept basis that lets the TCBM approximately replicate the black-box model. | It **adds and ranks** concepts; it does not remove concepts it finds bad - concepts failing the identifiability check are down-weighted or not selected in the first place. The provenance of the "ground truth labels" used in g(c) is not spelled out in the sections I read, while the paper's headline claim is that concept discovery is "fully unsupervised" and that it eliminates predefined human-labelled concepts - so how independent that label is, is an open question I could not resolve (UNVERIFIED). Completeness is measured against the black-box model's performance, not against an external concept label. |
| **Concept-existence metrics** | Aysel, Cai, Prugel-Bennett, "Concept-Based Explainable Artificial Intelligence: Metrics and Benchmarks", arXiv 2501.19271 (2025-01-31) | Proposes three metrics - concept **global importance**, concept **existence**, concept **location** - plus concept activation mapping for visualisation, and benchmarks post-hoc CBMs with them. Key empirical finding: "in many cases, even the most important concepts determined by post-hoc CBMs are not present in input images", and when present, saliency fails to align with the expected region; the paper analyses root causes such as natural concept correlation. | It **measures** existence/location/importance and reports that concepts frequently fail; it does not drop concepts, does not retrain, and gives no admission rule. The check is against the input image (an independent observable, which is the right shape for a gate) but the output is a benchmark verdict about method families, not a per-concept pass/fail gate applied to a specific model's concept set. |
| **Robust concept metrics** | Espinosa Zarlenga, Barbiero, Shams, Kazhdan, Bhatt, Weller, Jamnik, "Towards Robust Metrics for Concept Representation Evaluation", arXiv 2301.10367 (2023-01-25). Comment: "To appear at AAAI 2023" | Shows disentanglement-learning purity metrics are not appropriate for concept learning and proposes novel metrics for evaluating the **purity of concept representations** in both families; demonstrates their utility for evaluating robustness of concept representations and of interventions on them; benchmarks state-of-the-art methods from both families and finds that, "contrary to common assumptions, supervision alone may not be sufficient for pure concept representations". | It is a metric suite for representation purity and robustness: it **detects** impurities, it does not act on them (no removal, no threshold, no gate), and it does not audit a concept against an independently measured label for that concept. I verified the paper, the authors, the venue comment and the abstract; I did **not** verify the individual metric names (a sibling draft mentions a "1-1 score" - treat that term as UNVERIFIED here). Note also the follow-up position paper: Espinosa Zarlenga, "In Defense of Information Leakage in Concept-based Models", arXiv 2606.10669 (2026-06-09), comment says accepted as a position paper at ICML 2026 - i.e. there is a live argument that leakage is not always bad, which any drop-on-failure rule must engage with. |

### Does the full gate exist? Precise answer

Split the question into its two halves and the answer separates cleanly.

1. **Audit each concept against an independently measurable label: ESTABLISHED as an activity.**
   Verified instances: Network Dissection aligns units to an independent concept-label set with an
   IoU threshold; concept-existence/location metrics (2501.19271) check presence in the input;
   purity metrics (2301.10367) check representational purity; CT-CBM (2502.11100) computes an F1
   identifiability score against dev-split labels and multiplies it into concept selection.
   Adjacent verified work: Nanda, Speicher, Dickerson, Gummadi, "Unifying Model Explainability and
   Robustness via Machine-Checkable Concepts", arXiv 2007.00251 (2020-07-01) - concepts as
   predicates that a checker can evaluate over inputs, which is the closest verified precedent for
   "every concept is a mechanically computable predicate".
2. **Drop the concepts that fail, as an enforced admission criterion: NOT FOUND.** Everything close
   does something else:
   - Thresholded post-hoc labelling that withholds a label (Network Dissection) - a labelling
     threshold, not a gate on a claimed concept vocabulary.
   - **Selection by downstream utility**: Yuksekgonul, Wang, Zou, "Post-hoc Concept Bottleneck
     Models", arXiv 2205.15480 (v1 2022-05-31, v2 2023-02-01; comment: ICLR 2023 Spotlight) turns
     any network into a CBM and lets users debug/update and remove concepts; Oikarinen, Das,
     Nguyen, Weng, "Label-Free Concept Bottleneck Models", arXiv 2304.06129 (v1 2023-04-12,
     v2 2023-06-05; comment says published at ICLR 2023) builds CBMs without labelled concept data
     and filters candidate concepts using an independent multimodal signal while keeping accuracy.
     These are the closest verified thing to an automatic drop, but the criterion is **task
     accuracy / probe filter**, not an independent label for the concept's meaning.
   - **Expert-driven removal plus retraining** (CBDebug, 2509.21385) - removal happens, but the
     decision is human judgement and the evidence reported is downstream behaviour.
   - **Debugging experiments** rather than admission criteria.
3. **The union - audit against an independent label AND automatically drop the failures, reported
   as a per-concept pass/fail table before the claim is made: OPEN / NOT FOUND.** I found no paper
   that does this as a named, standardised protocol.
4. **The closest methodological precedent for "a gate that disqualifies things that fail an
   independent check" is not in the concept literature at all**: Adebayo, Gilmer, Muelly, Goodfellow
   et al., "Sanity Checks for Saliency Maps", arXiv 1810.03292 (2018-10-08) - randomises model
   parameters and data and shows some saliency methods are independent of both, i.e. the check
   *invalidates* the method. Verified related example of the same pattern: Oikarinen, Yan, Weng,
   "Evaluating Neuron Explanations: A Unified Framework with Sanity Checks", arXiv 2506.05774
   (2025-06-06; comment says published at ICML 2025), and Ji, Hu, Bi, Liu, "Select, Hypothesize and
   Verify: Towards Verified Neuron Concept Interpretation", arXiv 2603.24953 (2026-03-26; comment
   says accepted in CVPR 2026) - an explicit hypothesise-and-verify loop for neuron-concept claims.
   Verified audit-flavoured concept work: Mollapour, Aslansefat, Dehghani, Mishra, "ConceptSMILE:
   Auditing the Trustworthiness of Concept-Based Explainable AI", arXiv 2607.09649 (2026-07-10);
   Huang, Lai, Yue, "Concept Labels Are Not Enough: Rethinking Concept Bottleneck Models through
   Representation Integrity", arXiv 2510.15770 (v1 2025-10-17, v4 2026-08-11). I verified titles,
   authors, dates and the venue comments only - **not** their metrics or whether they drop concepts.

Conclusion for Part 2, honestly labelled: **the pieces are ESTABLISHED, the packaged discipline is
OPEN.** A concept-honesty gate with drop-on-failure is a defensible practice assembled from verified
components (alignment/existence/identifiability checks, removal-and-retrain validation) rather than
an existing protocol you can cite and follow. That means any claim built on it must say "we composed
known checks into an enforced gate", and must engage the ICML 2026 position that leakage is not
always bad (2606.10669).

---

# PART 3 - Data licensing, provenance, and what a small project must record

## 3.1 The provenance standards

- **Data Provenance Initiative (DPI)** - Longpre, Mahari, Chen, Obeng-Marnu, Sileo, Brannon,
  Muennighoff, Khazam, Kabbara, Perisetla, Wu, Shippole, Bollacker, Wu, Villa, Pentland, Roy,
  Hooker, "The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution
  in AI", arXiv 2310.16787 (v1 2023-10-25, v3 2023-11-04). Journal version: "A large-scale audit of
  dataset licensing and attribution in AI", Nature Machine Intelligence 6(8):975-987 (2024),
  DOI 10.1038/s42256-024-00878-8 (publisher page verified).
  - Scale: 44 widely used fine-tuning data collections comprising **1,858 individual datasets**.
  - Finds **licence omission rates above 70%** on popular hosting sites and **error rates above
    50%**; its stricter re-annotation finds **66% of analysed Hugging Face licences were in a
    different use category** than the original author's, often more permissive.
  - Ships: the **DPExplorer** (dataprovenance.org) for filtering by provenance/characteristics, the
    DPCollection repository, **symbolic attribution** (a structured bibliography for data), and
    auto-generated **Data Provenance Cards**.
- **Data Provenance Cards** - there is no separate paper under this exact title that I could find.
  The term is used by the DPI paper for its **auto-generated human-readable markdown summaries** of
  the datasets used, their languages, tasks and licences; the paper positions them against
  Datasheets and Data Statements: they are searchable/sortable/filterable and are meant to be
  concatenated as collections grow, whereas datasheets are read manually per dataset. Treat "Data
  Provenance Card" as DPI terminology for a generated composition-and-licence summary, not as an
  independent standard. (The Data & Trust Alliance "Data Provenance Standards" and MLCommons
  Croissant come up in this space; I did **not** verify either this session.)
- **Consent in Crisis** - Longpre, Mahari, Lee, Lund, Oderinwale, Brannon, ... Pentland, "Consent in
  Crisis: The Rapid Decline of the AI Data Commons", arXiv 2407.14933 (v1 2024-07-20, v2
  2024-07-24). First large-scale longitudinal audit of consent protocols for the web domains
  underlying training corpora: 14,000 web domains, tracking how codified data-use preferences
  change over time, finding proliferating AI-specific restrictions and inconsistencies. This is the
  problem upstream of the licence problem.
- **Datasheets** - Gebru, Morgenstern, Vecchione, et al., "Datasheets for Datasets", arXiv
  1803.09010 (v8). Venue UNVERIFIED on the arXiv page (commonly cited as CACM 2021).
- **Data Cards** - Pushkarna, Zaldivar, Kjartansson, et al., "Data Cards: Purposeful and Transparent
  Dataset Documentation for Responsible AI", arXiv 2204.01075 (2022-04-03).
- **Data Statements** - Bender & Friedman, "Data Statements for Natural Language Processing: Toward
  Mitigating System Bias and Enabling Better Science", TACL 6:587-604 (2018), ACL Anthology
  Q18-1041, DOI 10.1162/tacl_a_00041 (publisher and anthology pages verified).
- **Corpus-level licence practice example that a small project can copy** - Kocetkov, Li, Ben Allal,
  et al., "The Stack: 3 TB of permissively licensed source code", arXiv 2211.15533 (2022-11-20):
  documents how the full dataset was collected, how a **permissively licensed subset** was
  constructed, a **data governance plan**, limitations, an "Am I in The Stack" search tool, and a
  **removal process**. That is the practical template for "record enough to honour removals".

## 3.2 Hugging Face license tagging norms (verified from the official docs)

Sources verified: huggingface.co/docs/hub/repositories-licenses, huggingface.co/docs/hub/datasets-cards,
the dataset-card metadata spec at github.com/huggingface/hub-docs/blob/main/datasetcard.md, and the
huggingface_hub card reference (docs/huggingface_hub/en/package_reference/cards).

- Licence is declared in the repo's README.md card **YAML metadata** (`license: apache-2.0`), and
  the accepted identifiers are a **curated list on the Hub, not arbitrary SPDX strings** (the list
  includes apache-2.0, mit, bsd/bsd-2-clause/bsd-3-clause, afl-3.0, artistic-2.0, bsl-1.0, the cc
  family, cc0-1.0, c-uda, openrail, bigscience-openrail-m, bigscience-bloom-rail-1.0,
  creativeml-openrail-m, bigcode-openrail-m, and so on).
- Custom licences: `license: other` plus `license_name` and `license_link` (the spec says to add the
  text to a LICENSE file or link it), with `license_details` described as legacy.
- Dataset cards carry more than licence: `source_datasets`, `annotations_creators`,
  `language_creators`, `language_details`, `size_categories`, `task_categories`, `configs`,
  `data_files`, `pretty_name`, `tags`.
- Model cards carry a **`datasets`** field documented as "Dataset or list of datasets that were used
  to train this model", plus `license`, `license_name`, `license_link` and a `model-index` used for
  leaderboards. This is the machine-readable place to record which corpora trained a model.
- Consequence a small project must internalise: **a dataset with no card has no licence tag, and
  the Hub metadata is crowdsourced and frequently wrong.** By the DPI audit's numbers (more than 70%
  unspecified, 66% miscategorised on Hugging Face), a tag alone is not evidence - you must record
  where the licence claim came from.

## 3.3 The record a small project should keep about which corpora trained a model

One row per corpus, stored as versioned text next to the checkpoint, plus a generated summary card:

1. **Identity**: dataset name, Hub/URL identifier, exact revision or commit hash, and the local file
   hash of the copy actually used. Reason: a mutable dataset means an unreproducible run.
2. **Licence as tagged**: the Hub licence identifier (from the curated list) or
   `license_name`/`license_link` for custom licences; the licence text location; and the licence of
   the **upstream** source if the corpus is a repackaging. Reason: DPI found repackaging and
   "licence laundering" to be the common failure.
3. **Licence as verified**: whether the tag is self-reported only, or checked against the original
   author's terms; who checked it and when. Reason: 66% of audited HF licences were in a different
   use category from the author's.
4. **Missing licence = unspecified = not usable** under a strict rule; record that decision
   explicitly rather than defaulting to use.
5. **Lineage**: `source_datasets`, original creator/s, collection method, and the date the copy was
   taken. Reason: consent status is time-dependent (2407.14933).
6. **Consent/access status at collection time**: robots.txt, terms of use, or explicit AI-use
   clause, with the date; note any AI-specific restrictions. Reason: the same paper shows
   restrictions proliferating and becoming inconsistent over time.
7. **Composition summary**: languages, task categories, size, and the fraction of the training mix
   each corpus contributes. Reason: this is exactly what a Data Provenance Card auto-generates, and
   it is the part a reviewer or downstream user actually reads.
8. **Attribution text** to reproduce in the model card, verbatim where the licence demands it.
9. **Redistribution decision**: whether the corpus is redistributed with the model or only used for
   training; and how a removal request would be honoured.
10. **Model link**: the model card's `datasets` field, kept in sync with the above.
11. **Traps to record explicitly**: licence of the *code* (MIT/Apache-2.0 are software licences)
    versus the *data*; and use-restricted licences (RAIL/OpenRAIL family, which are on the Hub's
    list) which are permissive in copyright but carry use-based restrictions.

---

# PART 4 - Reporting standards

| Standard | Verified identity | What it is / what it demands |
|---|---|---|
| **Model Cards** | Mitchell, Wu, Zaldivar, et al., "Model Cards for Model Reporting", arXiv 1810.03993 (v2 2018-10-05). Venue UNVERIFIED on the arXiv page (commonly cited as FAT*/FAccT 2019) | A short model-level document covering intended use, factors, metrics, **evaluation data**, **training data**, quantitative analyses (disaggregated where possible) and ethical considerations. This is where a per-corpus provenance table and the audit verdicts belong. |
| **Datasheets for Datasets** | Gebru, Morgenstern, Vecchione, et al., arXiv 1803.09010 | The dataset-level questionnaire (motivation, composition, collection, preprocessing/cleaning, uses, distribution, maintenance). The dataset-level counterpart of the model card. |
| **Data Statements** | Bender & Friedman, TACL 6:587-604 (2018) | NLP-specific statement of curation rationale, language variety and speaker demographics. |
| **NeurIPS Paper Checklist** | neurips.cc/public/guides/PaperChecklist (page verified 2026-10-16) | The institutional precedent for a **binary gate with a written justification**. Verified wording: the checklist is included in the LaTeX style file; papers not including it "will be desk rejected"; each question is answered yes / no / n/a, and "if you would like to explain your answer (e.g. explain why the answer is 'no') ... consider referencing the section(s) of the paper"; answers are visible to reviewers, area chairs, senior area chairs and ethics reviewers, and reviewers are asked to use the checklist as one factor. First items include **Claims** (main claims must match theoretical and experimental results and stated scope/assumptions/limitations) and **Limitations** (a separate Limitations section is encouraged, with assumptions and the robustness of results to their violation). Answering "no" with justification is acceptable and not grounds for rejection. |
| **Reproducibility programme report** | Pineau, Vincent-Lamarre, Sinha, et al., "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)", arXiv 2003.12206 (v4) | The published basis of the reproducibility-checklist practice: code/data availability, hyperparameters, experimental details, error bars. |
| **REFORMS** | Kapoor, Cantrell, Peng, Pham, et al., "REFORMS: Reporting Standards for Machine Learning Based Science", arXiv 2308.07832 (v2 2023-09-19) | A structured reporting standard for ML-based science (claim-to-evidence traceability, data leakage checks, error bars). I verified title/authors/date only; the specific item count is UNVERIFIED here. |
| **Statistical grounding of explanation verdicts** | Basu & Chakraborty, "ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs", arXiv 2603.18579 (v1 2026-03-19, v2 2026-09-02) | Title verified; argued direction is that intervention-based explanation evaluation (the ERASER family) needs statistical grounding. Details UNVERIFIED - cite the direction, not specifics. |
| **Pre-registration in ML** | (a) Hofman, Chatzimparmpas, Sharma, Watts, Hullman, "Pre-registration for Predictive Modeling", arXiv 2311.18807 (2023-11-30): proposes adapting pre-registration to predictive modelling, introduces a lightweight template, and reports a qualitative study with ML researchers, while acknowledging limitations. (b) "NeurIPS 2020 Workshop on Pre-registration in Machine Learning", PMLR volume 148 (11 December 2020; editors Bertinetto, Henriques, Albanie, Paganini, Varol) - verified on proceedings.mlr.press/v148. (c) "Preregistration for Experiments with AI Agents", arXiv 2606.11217 (2026-05-03), whose own text states it appears in Proceedings of the 43rd ICML, PMLR 306, 2026: proposes an AI-agent-specific preregistration template, recommends registering on OSF Registries before data collection, allows a staged preregistration option borrowed from registered reports, and asks venues to at least require disclosure of whether a study was preregistered. | Pre-registration in ML is **voluntary and marginal** - there is no mandatory registration, no stopping rule enforced by a venue, and no clinical-trial-style registry. Practical conventions worth copying: write the plan before data collection; register publicly (OSF); allow disclosed, justified deviations; label exploratory analyses as exploratory. |

---

# What to adopt for our project (run BEFORE claiming any result)

Ordered as a gate. Items 1-4 and 8-10 are measurements; 5-7 are decisions that must be declared in
advance; 11-16 are documentation. Each carries the reason.

**Measurement (concept side)**

1. **A mechanically computable definition for every named concept** - a predicate or probe a script
   can evaluate on an input, not free text. Reason: verified precedent that concepts can be made
   machine-checkable (2007.00251); without it, "concept present" is an opinion and the audit is
   unfalsifiable (the failure mode in 2105.04289).
2. **Per-concept identifiability score against a held-out labelled probe**: F1/AUROC of a linear
   read-out of the concept from the model's representation, computed on data the concept was not
   trained on. Reason: this is the auditable core of CT-CBM (2502.11100) and the closest thing to
   "the model can actually detect this concept"; it is also the check that catches a named concept
   that is merely a label the model never uses.
3. **Per-concept existence and location check on inputs**, reported as a count, plus what fraction
   of the inputs where the concept is *claimed* actually contain it. Reason: verified finding that
   even the most important post-hoc concepts are often absent from the inputs (2501.19271).
4. **An alignment gate against rival concepts, not just against the null**: for each unit/direction,
   report the best-matching and second-best-matching concept, with the IoU/AUROC margin and a
   pre-declared threshold, over a closed vocabulary. Reason: this is Network Dissection's operation
   (1704.05796) and it is the only verified way to stop a concept from being "confirmed" by a test
   that never considered an alternative; a closed vocabulary is required, and saying so up front
   prevents post-hoc additions.
5. **Declare the drop rule in advance, and report the audited set separately from the full set**:
   concepts failing 2-4 are removed from the claimed concept set, and the model is reported both
   with-and-without (removal + retrain, CBDebug-style, 2509.21385), including the accuracy cost of
   the removal. Reason: no standard mandates dropping, and removal has a price; reporting only the
   surviving set after seeing the results is exactly the practice this gate exists to prevent.
   Also pre-empt the verified counter-position (2606.10669) that leakage is not always harmful.
6. **Leakage check on the concept layer** - test whether concept representations carry information
   beyond their declared concept, and report it per concept. Reason: 2106.13314 shows this happens
   and that natural mitigations do not fully fix it; 2301.10367 adds that supervision alone may not
   give pure concept representations.
7. **Never substitute human plausibility for faithfulness**: report the two axes separately and
   never cite human agreement as evidence of faithfulness; state faithfulness as a graded claim, not
   a binary one. Reason: the verified definitions and the explicit call to drop the binary notion
   (2004.03685).
8. **Sanity-check every explanation method used**: randomise model parameters and, where possible,
   training data, and report whether the explanation changes. Reason: independent check that the
   explanation depends on the model at all (1810.03292); the same pattern is standardised for neuron
   explanations (2506.05774, 2603.24953).
9. **Faithfulness scores with a protocol and a random reference row**: if using rationale removal
   metrics, report comprehensiveness and sufficiency with the top-k% (AOPC-style) sweep used by
   ERASER, alongside the random-score baseline, and state the model-relative nature of the numbers.
   Reason: verified protocol and its own random reference (1911.03429); the paper itself flags
   that faithfulness measurement is an open question and that comprehensive ground-truth rationales
   exist only for some corpora, so the limitation must travel with the number.
10. **If using TCAV, use the smoothed estimator and report the calibrated statistic**: report the
    alpha-TCAV variant with its parameter choice and the calibrated probability that the concept
    positively influences the prediction, and keep the permutation test. Reason: the standard TCAV
    score is verified to be unstable because of its discontinuous indicator (2605.15688).
11. **Statistics beside every number**: seeds (>=3), error bars, and a significance test for each
    headline comparison; report a margin smaller than the pooled noise as "no difference". Reason:
    the reproducibility-checklist practice (2003.12206) plus the statistical-grounding argument in
    the explanation-evaluation literature (2603.18579, direction only) and the pre-registration
    literature's emphasis on avoiding data-dependent decisions (2311.18807).

**Documentation (before the claim goes anywhere)**

12. **Pre-registration file, frozen before the first run**: the clause list with thresholds, the
    predictions (including the ones expected to fail), and the drop rule from item 5. Amendments
    only as dated entries that state no result had been read. Reason: the ML pre-registration
    conventions (2311.18807; PMLR v148; 2606.11217) and the fact that a gate is only credible if its
    bar was fixed before the numbers existed.
13. **A model card** with intended use, factors, metrics, evaluation data, **training data**,
    quantitative analyses and ethical considerations, plus the per-concept audit verdict table.
    Reason: 1810.03993; this is where the audit becomes reviewable.
14. **One dataset record per corpus** (datasheet / data statement / data card content, whichever
    fits) and a **generated Data Provenance Card** summarising the training mix's languages, tasks
    and licence composition. Reason: 1803.09010, Q18-1041, 2204.01075, and the DPI's auto-generated
    provenance cards (2310.16787 / NMI 2024) which are designed exactly for "which corpora trained
    this model".
15. **Licence field per corpus with the Hub identifier from the official list, including
    `license_name`/`license_link` for custom terms, and a flag for whether the tag is self-reported
    or verified** - treating an unspecified licence as unusable for a strict claim. Reason: Hub
    tagging norms (docs verified) plus the verified audit numbers (more than 70% unspecified, 66%
    in a different use category).
16. **A claims/limitations section whose claims match the experiments, with a written justification
    for every "no"**, kept as a checklist artifact even for internal work. Reason: the NeurIPS Paper
    Checklist is the verified institutional precedent for a binary, justified, reviewer-visible
    gate; it also encodes the two items this project most needs - claims must match scope, and
    limitations must be stated.

Practical sequencing: run items 1-11 on the model, fill in 12 first (it is the only item that must
exist before the first run), then assemble 13-16 from the artifacts the gate produced. If an item
cannot be run, say which one and why in the same document - the reporting standards above all
tolerate a justified "no"; none of them tolerate a silently missing check.

---

# Verified citation list

| Short name | Verified citation | Verified how |
|---|---|---|
| TCAV | Kim, Wattenberg, Gilmer, Cai, Wexler, Viegas, Sayres. arXiv 1711.11279 (2017-11-30 / v5 2018-06-07) | arXiv abstract page: title, authors, date, full abstract |
| alpha-TCAV | Schnoor, Said, Tiomoko, Samek, Jung. arXiv 2605.15688 (v1 2026-05-15, v2 2026-10-12), 26 pages | arXiv abstract page (title/authors/dates/abstract) |
| Concept leakage | Mahinpei, Clark, Lage, Doshi-Velez, Pan. arXiv 2106.13314 (2021-06-24) | arXiv abstract page, full abstract |
| CBM as intended? | Margeloiu, Ashman, Bhatt, Chen, Jamnik, Weller. arXiv 2105.04289; comment: ICLR 2021 Workshop on Responsible AI | arXiv abstract page incl. comment field |
| Koh CBM (background) | Koh, Nguyen, Tang, Mussmann, Pierson, Kim, Liang. arXiv 2007.04612; comment: "Edited for clarity from the ICML 2020 version" | arXiv abstract page |
| Completeness / ConceptSHAP | Yeh, Kim, Arik, Li, Pfister, Ravikumar. arXiv 1910.07969 (v1 2019-10-17, v6 2022-02-07) | arXiv abstract page; venue not on page |
| ERASER | DeYoung, Jain, Rajani, Lehman, Xiong, Koh, Tan, Wallace. arXiv 1911.03429 (2019-11-08) | arXiv abstract page + ar5iv full text (comprehensiveness/sufficiency definitions quoted) |
| Plausibility vs faithfulness | Jacovi & Goldberg. arXiv 2004.03685 (2020-04-07) | arXiv abstract page + ar5iv full text (definitions quoted) |
| Sanity checks | Adebayo, Gilmer, Muelly, Goodfellow, et al. arXiv 1810.03292 (2018-10-08) | arXiv abstract page |
| Network Dissection | Bau, Zhou, Khosla, Oliva, Torralba. arXiv 1704.05796; comment: CVPR 2017 oral | arXiv abstract page |
| CBDebug | Enouen & Galhotra. arXiv 2509.21385 (v1 2025-09-23, v2 2026-03-25); comment: ICLR 2026 | arXiv abstract page, full abstract |
| CT-CBM | Bhan, Choho, Moreau, Vittaut, Chesneau, Lesot. arXiv 2502.11100 (v1 2025-02-16, v3 2025-05-28) | arXiv abstract page + arXiv HTML v3 (identifiability score and initialisation quoted) |
| Concept-existence metrics | Aysel, Cai, Prugel-Bennett. arXiv 2501.19271 (2025-01-31) | arXiv abstract page, full abstract |
| Robust concept metrics | Espinosa Zarlenga, Barbiero, Shams, Kazhdan, Bhatt, Weller, Jamnik. arXiv 2301.10367; comment: "To appear at AAAI 2023" | arXiv abstract page, full abstract |
| Post-hoc CBM | Yuksekgonul, Wang, Zou. arXiv 2205.15480 (v1 2022-05-31, v2 2023-02-01); comment: ICLR 2023 Spotlight | arXiv abstract page |
| Label-free CBM | Oikarinen, Das, Nguyen, Weng. arXiv 2304.06129 (v1 2023-04-12, v2 2023-06-05); comment: published at ICLR 2023 | arXiv abstract page |
| Machine-checkable concepts | Nanda, Speicher, Dickerson, Gummadi. arXiv 2007.00251 (v1/v2 2020-07-01/02) | arXiv abstract page |
| Concepts' Information Bottleneck | Galliamov, Kazmi, Khan, Ramirez Rivera. arXiv 2602.14626; comment: to appear ICLR 2026 | arXiv abstract page |
| Leakage position paper | Espinosa Zarlenga. arXiv 2606.10669; comment: ICML 2026 position paper | arXiv abstract page |
| ConceptSMILE | Mollapour, Aslansefat, Dehghani, Mishra. arXiv 2607.09649 (2026-07-10) | arXiv abstract page (title/authors) |
| Representation integrity | Huang, Lai, Yue. arXiv 2510.15770 (v1 2025-10-17, v4 2026-08-11) | arXiv abstract page (title/authors) |
| Neuron-explanation sanity checks | Oikarinen, Yan, Weng. arXiv 2506.05774; comment: ICML 2025 | arXiv abstract page (title/venue) |
| Select, hypothesise, verify | Ji, Hu, Bi, Liu. arXiv 2603.24953; comment: CVPR 2026 | arXiv abstract page (title/venue) |
| ICE | Basu & Chakraborty. arXiv 2603.18579 (v1 2026-03-19, v2 2026-09-02) | arXiv abstract page (title only) |
| DPI | Longpre, Mahari, Chen, et al. arXiv 2310.16787; Nature Machine Intelligence 6(8):975-987 (2024), DOI 10.1038/s42256-024-00878-8 | arXiv listing + Nature article page + publisher PDF/ESM text |
| Consent in Crisis | Longpre, Mahari, Lee, et al. arXiv 2407.14933 (2024-07-20/24) | arXiv abstract page |
| The Stack | Kocetkov, Li, Ben Allal, et al. arXiv 2211.15533 (2022-11-20) | arXiv abstract page |
| Datasheets | Gebru, Morgenstern, Vecchione, et al. arXiv 1803.09010 | arXiv abstract page (title/authors) |
| Data Cards | Pushkarna, Zaldivar, Kjartansson, et al. arXiv 2204.01075 (2022-04-03) | arXiv search listing |
| Data Statements | Bender & Friedman. TACL 6:587-604 (2018), ACL Anthology Q18-1041, DOI 10.1162/tacl_a_00041 | ACL Anthology + MIT Press TACL page |
| Model Cards | Mitchell, Wu, Zaldivar, et al. arXiv 1810.03993 (v2) | arXiv search listing |
| NeurIPS checklist | neurips.cc/public/guides/PaperChecklist | Page fetched and read 2026-10-16 |
| HF licence/dataset-card tagging norms | huggingface.co/docs/hub/repositories-licenses; docs/hub/datasets-cards; github.com/huggingface/hub-docs/blob/main/datasetcard.md; docs/huggingface_hub/en/package_reference/cards | Docs pages fetched and read |
| Reproducibility programme | Pineau, Vincent-Lamarre, Sinha, et al. arXiv 2003.12206 (v4) | arXiv search listing |
| REFORMS | Kapoor, Cantrell, Peng, Pham, et al. arXiv 2308.07832 (v2) | arXiv abstract page (title/authors) |
| Pre-registration for predictive modeling | Hofman, Chatzimparmpas, Sharma, Watts, Hullman. arXiv 2311.18807 (2023-11-30) | arXiv abstract page |
| NeurIPS 2020 Pre-registration Workshop | PMLR volume 148, 11 Dec 2020; editors Bertinetto, Henriques, Albanie, Paganini, Varol | proceedings.mlr.press/v148 page fetched |
| Preregistration for AI-agent experiments | arXiv 2606.11217 (2026-05-03); own text: ICML 2026, PMLR 306 | arXiv HTML/PDF text |

# Things I could NOT verify (do not cite these as established)

1. **"Data Provenance Cards" as a standalone paper or standard**: not found. The term is used inside
   the DPI work (2310.16787 / NMI 2024) for auto-generated markdown provenance summaries.
2. **Data & Trust Alliance "Data Provenance Standards"** and **MLCommons Croissant**: not checked
   this session. Do not cite without verification.
3. **EU AI Act Article 53(1)(d)** training-content summary requirement: not checked this session.
4. **Venues for older papers were not stated on their arXiv pages** and are therefore marked
   UNVERIFIED above: TCAV (ICML 2018), ERASER (ACL 2020), Jacovi & Goldberg (ACL 2020), Yeh et al.
   (NeurIPS 2020), Mitchell et al. (FAT*/FAccT 2019), Gebru et al. (CACM 2021).
5. **A canonical paper isolating TCAV's sensitivity to the choice of negative/random examples**: not
   found. The instability defect documented by alpha-TCAV (2605.15688) is verified; the
   negative-set sensitivity claim is my characterisation and should be presented as such.
6. **CT-CBM's "ground truth labels" for the identifiability score**: the paper's own framing is
   unsupervised concept discovery, but the score is an F1 against "ground truth labels on the dev
   split" whose provenance I could not determine from the sections read. Do not claim that CT-CBM
   audits concepts against an independent human label.
7. **Details (metrics, whether they drop concepts) for ConceptSMILE (2607.09649), representation
   integrity (2510.15770), neuron-explanation sanity checks (2506.05774), select-hypothesise-verify
   (2603.24953) and ICE (2603.18579)**: titles/authors/venues/dates verified only.
8. **The specific metric names in the robust-concept-metrics paper (2301.10367)**, including any
   "1-1 score": not verified.
9. **REFORMS item count (32)**: not verified.
10. **Method caveat**: verification used live page reads (arXiv abstract pages, arXiv HTML, ar5iv,
    publisher pages, official docs) because the arXiv API returned HTTP 429 during part of the
    session. This review covers the works named in the task plus directly adjacent verified work as
    of 2026-10-16; it is not an exhaustive literature crawl.
