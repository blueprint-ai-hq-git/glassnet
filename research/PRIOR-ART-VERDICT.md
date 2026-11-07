# PRIOR-ART VERDICT - 2026-10-15 (research batch completed)

## The headline answer to "is ours better?"
NO on accuracy, and the architecture itself is published prior art. One piece of our method appears
to be unpublished. Publish as method + demo, never as a new architecture.

## Closest published match (verified, real paper)
CB-LLM (Sun et al., ICLR 2025, arXiv:2412.07992) - a language model whose answer layer sees only
named concepts. Also text CBMs: TBM (2310.19660), CT-CBM (Findings EMNLP 2025, 2502.11100),
C3M (2311.05014), SelfExplain (ACL 2021, 2103.12279).
DIFFERENCE THAT SURVIVES: theirs use LEARNED/unsupervised concepts on PRETRAINED backbones. Ours uses
hand-named SYMBOLIC formulas (token regex/support) on a TINY FROM-SCRATCH LM. That is a scope
difference, not a new mechanism.

## The one piece we could NOT find published anywhere
The strict drop-on-misalignment honesty audit: each named concept neuron must track its OWN measurable
label better than every rival label, and concepts that fail are DROPPED and the model re-shipped with
a fully honest wall. Nearest neighbours found:
* Network Dissection (CVPR 2017, 1704.05796) - labels units that align with SOME concept; no
  beats-all-rivals test, no drop.
* CBDebug (ICLR 2026, 2509.21385) - removes concepts from expert/LLM "spuriousness" feedback, not from
  an intrinsic own-label test.
* "Do concept bottleneck models learn as intended?" (Margeloiu 2021, 2105.04289) and concept-leakage
  work (Mahinpei 2021, 2106.13314) - MEASURE the failure, report it, do not gate the ship on it.
* CT-CBM (2502.11100) - optimises completeness by ADDING concepts, never dropping.
MARKED OPEN, not claimed as ours until re-verified tomorrow.

## The rest of our pipeline, mapped to prior art (all components known)
input->named concepts->answer = CBM (Koh et al. 2020, ICML, 2007.04612); concept whitening
(Nature MI 2020, 2002.01650); Concept Embedding Models (NeurIPS 2022, 2209.09056); Label-Free CBM
(ICLR 2023, 2304.06129); Probabilistic CBM (ICML 2023, 2306.01574); Post-hoc CBM (ICLR 2023,
2205.15480); CBM survey (TMLR 2026, OpenReview IF5vnqxBEW). Rule-shaped relatives: LEN
(Artif. Intell. 2022, 2108.05149), SENN (NeurIPS 2018, 1806.07538), ProtoPNet (NeurIPS 2019,
1806.10574), Tsetlin Machine (1804.01508). Metrics: TCAV (ICML 2018, 1711.11279), completeness/
ConceptSHAP (NeurIPS 2020, 1910.07969), ERASER.
Verdict label, per the three-way rule: RECOMBINATION, NOT REPLICATION. Every component is prior art;
the combination (tiny from-scratch dialogue LM + ~40 symbolic named concepts + <=3-concept sparse
rules + strict per-concept audit that drops misalignments) is not matched in what was verified.

## Market (verified track)
Money is in audit tooling for other people's black boxes (Arize/Fiddler/Arthur; SHAP 25,814 stars vs
imodels 1,627), not in selling tiny readable models. Regulation does NOT mandate auditable-by-construction
(EU AI Act Art.13 is a documentation + output-interpretability duty; a SHAP report satisfies it).
Best-fit positioning: teaching artefact (B) + publish the gate methodology (E). Revenue-shaped paths
(C glass-box for one regulated decision, D edge/logic models) both require closing the accuracy gap
first. Papers with Code shut down July 2025 -> HF papers + GitHub are the discovery channels.
Note: our 56-64% next-token on our own split is NOT comparable to published small-LM numbers
(SmolLM2-135M MMLU 31.5, Gemma 2 2B MMLU 51.3). Do not print a comparison table unless same
benchmark and same domain.

## Community-training precedent (verified track)
Closest to "auditable + community-trained": Gensyn OPEN-1B (1.61B params, every step hashed,
volunteers certify individual steps) and EleutherAI Pythia (interpretability-native, 154 checkpoints,
Apache-2.0). Working templates for open community training: Prime Intellect INTELLECT-1/2, Ai2 OLMo.
2026 individual bar: GPT-2 capability from scratch is a ~$48 weekend (nanochat); a real 1-3B model is
hundreds-thousands of dollars and days. Participation paths without budget: run a node, run audits,
submit to BabyLM, replicate TinyStories/llm.c.

## Not finished (re-run tomorrow, ~4 min)
Track 2 (interpretability evaluation + licensing/provenance standards) was interrupted mid-write on
the maintainer's stop order. Its raw log is research/task-2.log. It had already verified: TCAV, alpha-TCAV
(2026), ERASER, concept leakage, completeness. Re-run it in the morning to get the written deliverable.
