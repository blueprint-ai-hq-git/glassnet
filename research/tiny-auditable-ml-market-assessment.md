# Tiny, auditable, interpretable ML — real demand or not?

Legend: **[F]** = sourced fact (link given) · **[I]** = my inference.

## Bottom line
**[I]** The money is in *audit/observability tooling for other people's black-box models*, not in selling
small readable models. There is a real but tiny market for interpretable-by-design models, and it is
funded mostly for **energy/logic efficiency (Tsetlin)** and **regulatory explainability in credit/health**.
Nobody is paying for a *tiny* auditable general model. Details below.

---

## 1. Interpretability / observability vendors — audit tooling, not interpretable models

| Vendor | What they actually sell | Money |
|---|---|---|
| Fiddler AI | AI observability + guardrails + "control plane for AI"; started as explainability (SHAP-class) | $32M Series B (2021), Series B Prime (Dec 2024), **$30M Series C Jan 2026** — fiddler.ai/press-releases/fiddler-raises-30m-series-c |
| Arthur AI | Model monitoring, drift, fairness, LIME/SHAP explainability, governance | ~$63M total **[F, weak source: appsecsanta.com/arthur-ai]** |
| WhyLabs | Observability; LangKit OSS text metrics for LLMs | Current funding/ownership **unverified** — docs.whylabs.ai |
| Arize AI | Evaluation + observability + tracing for LLM/agent apps | $19M A, $38M B (Sep 2022), **$70M C** — arize.com/blog/arize-ai-raises-70m-series-c |
| Robust Intelligence | Algorithmic red-teaming + "AI Firewall" | **Acquired by Cisco, ~$400M, Aug 2024** → Cisco AI Defense — newsroom.cisco.com/.../Cisco-Acquires-Robust-Intelligence.pdf |
| Lakera | Runtime guardrails / adversarial testing | **Acquired by Check Point, ~$201.8M, closed 22 Oct 2025** — checkpoint.com/press-releases/check-point-acquires-lakera... |

**[F] All six sell monitoring/evaluation/guardrails wrapped around *someone else's* model.** None ships a
readable model. The whole category is "make a black box auditable after the fact."
**[F]** Popularity signal: SHAP 25,814★ and LIME 12,167★ (post-hoc explainability) vs imodels 1,627★ and
skope-rules 663★ (interpretable-by-design) — the audit-tooling side is ~15x larger than the
build-readable-models side by the only rough proxy available.
**[I]** These companies are your *competitors for the same purchasing language* and would beat you on
distribution, not on model architecture.

## 2. Interpretable-by-design vendors & research

- **Literal Labs** (Newcastle University spinout, UK) — Tsetlin machines, led by ex-Arm VP Noel Hurley,
  **£4.6M pre-seed, June 2025**, Northern Gritstone's first investment; Ole-Christoffer Granmo chairs its
  technical steering committee. [F] businesswire.com/home/20250602939541 · ncl.ac.uk/press/articles/archive/2025/06/literallabs/
- **Anzyz Technologies** (Grimstad, Norway) — Granmo co-founder; ships Tsetlin-based text analytics. [F] anzyz.com/products/tsetlin
- **⚠ Naming check:** "Cairnstack" is **not** a Tsetlin Machine company — Tracxn lists it as asset-tracking
  SaaS. The real Tsetlin commercial players are Literal Labs + Anzyz. [F]
- **Interpretable AI LLC** (MIT, Bertsimas) — Optimal Classification Trees sold as commercial software. [F] interpretable.ai/products/optimal-trees/
- **Zest AI** — credit underwriting where *explainability is the product* (adverse-action reasons). [F] zest.ai/redefining-explainable-ai
- **RAAPID** — "black box to glass box" neuro-symbolic risk adjustment for healthcare payers. [F] blogs.microsoft.com/bayarea/2026/07/22/from-black-box-to-glass-box-how-raapid-is-bringing-explainable-ai-to-healthcare-risk-adjustment
- **Symbolica** — category-theory symbolic reasoning, **$31M** (Reuters, Apr 2024). [F] reuters.com/technology/symbolica-raises-31-mln...
- **Logical Intelligence** — energy-based reasoning + formal verification, LeCun-linked; funding reports
  conflict ($7M seed vs much larger later claims) → **low confidence**. [F/uncertain]
- **Tooling:** imodels (Berkeley, 1.6k★), skope-rules (663★), Google TF-Lattice (reimplemented in PyTorch, HN 13 pts). [F]
- **Prior art that commoditises the core idea:** **Concept Bottleneck Models** (Koh, Nguyen, Tang,
  Mussmann, Pierson, Kim, Liang — ICML 2020, arXiv:2007.04612). features→concepts→answer is **6 years old
  and well cited**. [F] **[I]** The *pipeline shape* is not your differentiator; the multi-modal coverage +
  binary gates are.

## 3. Tiny / on-device LM ladder — where 8M params actually sits

| Model | Params | Benchmark | Source |
|---|---|---|---|
| SmolLM2-135M | 135M | MMLU(cloze) 31.5, HellaSwag 42.1 | huggingface.co/HuggingFaceTB/SmolLM2-135M |
| TinyLlama-1.1B | 1.1B | MMLU 25.36 | github.com/jzhang38/TinyLlama/blob/main/EVAL.md |
| SmolLM2-1.7B | 1.7B | HellaSwag 68.7, ARC 60.5 | HF card |
| Gemma 2 2B (PT) | 2B | **MMLU 51.3** | HF card / ai.google.dev/gemma/docs/core/model_card_2 |
| Phi-3-mini-4k-instruct | 3.8B | **MMLU 70.9**, mean 67.6 | arXiv:2404.14219 |

- **[F] The strongest published precedent for your scale:** TinyStories (Eldan & Li, arXiv:2305.07759) —
  models **below 10M parameters** "still produce fluent and consistent stories… almost perfect grammar,
  and demonstrate reasoning capabilities." **Caveat in the same abstract:** the data is synthetic,
  restricted to words a 3–4-year-old knows, generated by GPT-3.5/4.
- **[F]** Community proof it gets attention: "MacroStories — 19,969 parameters, 81 KB FP32"
  (r/LocalLLaMA, huggingface.co/raincandy-u/MacroStories); TinyTTS ~9M params.
- **[I] Blunt:** your ~56–64% next-token accuracy is on *your own* held-out split, so it is not comparable
  to any MMLU/HellaSwag number above. Against general capability, 8M params is ~3 orders of magnitude below
  what the market calls "small" (2–4B). Do **not** place it in a benchmark table next to SmolLM2 unless you
  evaluate on the *same* benchmark and domain.

## 4. Regulation / standards drivers

- **EU AI Act** — transparency is real, but **later and narrower than commonly pitched**. The **AI Omnibus**
  (part of the Digital Omnibus proposed 19 Nov 2025) **entered into force 27 July 2026** and pushed
  high-risk rules out: **Annex III high-risk applies 2 Dec 2027; Annex I 2 Aug 2028**. [F]
  digital-strategy.ec.europa.eu/en/news/ai-omnibus-enters-force · artificialintelligenceact.eu/implementation-timeline/
- **Art. 13** requires that high-risk systems be "designed and developed in such a way as to ensure that
  their operation is **sufficiently transparent to enable deployers to interpret a system's output**" +
  instructions for use. [F] artificialintelligenceact.eu/article/13/ — **[I] this is a documentation +
  output-interpretability obligation, NOT a mandate for glass-box architectures.** A post-hoc SHAP report
  can satisfy it as well as your model can. **Art. 86** gives individuals a right to explanation. [F]
- **FDA** — Predetermined Change Control Plan final guidance (**Federal Register, 4 Dec 2024**) and
  AI-enabled device software function guidance (**18 Aug 2025**). Transparency = *documentation of
  description / modification protocol / impact assessment*, not architecture. [F]
  fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/predetermined-change-control-plans...
- **NIST AI RMF 1.0** lists "explainable and interpretable" as a trustworthiness characteristic under
  Measure/Manage — guidance, not a mandate. [F] nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- **Defence** — DoD RAI Strategy (Traceable / Governable / Reliable / Equitable, 2024); UK **JSP 936**:
  "Understanding is driven by a combination of transparency and explainability"; OMB **M-24-18** AI
  acquisition memo. **Actual money exists:** AFWERX SBIR Phase I to ResilienX for *explainable* AI
  model-health monitoring (Aug 2026); USAF ≈**$14.54M** HAARDCORE "explainable AI and autonomy" contract. [F]
- **[I] "Auditable by construction" is not a purchasing criterion in any regulation.** The phrase appears in
  vendor marketing and academic governance frameworks (e.g. arXiv 2605.07728, "SARC"), not in the AI Act,
  FDA guidance, NIST AI RMF, or DoD procurement text. Do not build a pitch on that exact phrase.
- **[I]** Regulation creates demand for **evidence artefacts and workflow** (documentation, logs, drift
  evidence, impact assessments) — which is the Fiddler/Arize/Arthur shape, not a model.

## 5. Community distribution — what actually gets attention

- **[F] Appetite for tiny models is real, but for *utility*:** "Ask HN: Is anyone doing anything cool with
  tiny language models?" — **684 points / 346 comments**
  (news.ycombinator.com/item?id=42784365). The top comment is a working Excel add-in using **Gemma 2 2B** to
  screen 7,000 paper titles — not an interpretable model.
- **[F] Interpretability-on-its-own Show HNs flop:** "Show HN: Coelanox – auditable inference runtime in
  Rust" 3★; "Ollie – Glass-box AI code editor" 3★; "Open-Sourcing Google's Lattice Models" 13★;
  "Interpretable AutoResearch" 4★; "Concept-Vector – human-interpretable word embeddings" 3★.
- **[F] What *does* front-page in interpretability is big-model research or utility:** "Llama 3.2
  Interpretability with Sparse Autoencoders" **579★**; IBM/Euler patent for interpretability 172★;
  mechanistic-interpretability + causality 117★.
- **[I]** An 8M glass-box model will not reach the front page on "interpretability" alone. Frame it as a
  *result or a tool* (e.g. "N concepts, max-3-concept rules, binary gates, 4 modalities at 8M params —
  here's a model you can read end-to-end"), not as a model release.
- **[F] Hugging Face has no auditable-model hub:** tag `interpretability` ≥1,000 models (API cap),
  `explainable-ai` 155, but search "auditable" = **2 models**, "tsetlin" = **0**. HF `filter=tsetlin` = 0.
- **[F] Papers with Code is dead** — shut down July 2025, paperswithcode.com redirects to Hugging Face;
  trending papers moved to huggingface.co/papers. GitHub is now the main discovery channel for
  interpretable-ML code (SHAP 25.8k★, imodels 1.6k★).

---

## Positioning options (evidence → audience → is the differentiator real?)

| # | Position | # | Overlaps with | Audience | Differentiator real? |
|---|---|---|---|---|---|
| A | Audit-evidence / observability layer for other models | | Fiddler, Arize, Arthur, WhyLabs, Audicta, Pratvi AI | Compliance/risk engineering | **Commoditised.** Funded incumbents, no distribution edge for you. |
| B | Readable-model **teaching artefact** (4 modalities, concept+rule chains, gates) | | TinyStories, nanoGPT 63.8k★, HF smol-course 6.8k★, imodels | Students, teachers, self-learners, CS courses | **Real.** Nobody ships a multi-modal end-to-end-readable stack at this size. Monetisation is weak (course/book/kit), attention is high. |
| C | Glass-box models for **one narrow regulated decision** (credit / claims / triage) | | Zest AI, Interpretable AI, RAAPID | Model-risk-management + compliance at lenders/payers | **Half-real.** Auditability-by-design is a genuine wedge, but 66% PathMNIST / 37.4% speech / 56–64% text is far below the accuracy bar those buyers enforce. You lose on accuracy, not transparency. |
| D | **Edge/logic models** (deterministic, low-energy, FPGA/CPU) | | Literal Labs (£4.6M), Anzyz, Tsetlin FPGA work (arXiv:2507.14874) | Embedded/hardware, defence autonomy (AFWERX-style SBIR) | **Real but contested** — a funded spinout's entire thesis is already this. |
| E | **Research artefact:** publish the explanation-chain + binary-gate methodology; model as the demo | | Mechanistic-interp literature (the 117–579★ HN class), CBMs | Researchers, arXiv/HN, reviewers | **Most real per unit of effort.** The *gates* over explanation chains are the least commoditised piece — CBM prior art (2020) covers features→concepts→answer. Not revenue. |
| F | Sell the **rule layer + gates** as verification for high-assurance agents | | Symbolica ($31M), Logical Intelligence, formal-verification startups | High-assurance / safety engineering | Real in principle, crowded with well-funded specialists. |

**[I] Recommended honest framing:** B + E as the beachhead (attention, credibility, zero accuracy risk);
C and D are the only revenue-shaped paths, and both require closing an accuracy gap first — a 66%-accurate
medical classifier will not be bought, however readable it is.

## Weaknesses to state plainly
- **Accuracy:** 56–64% text / 86% Fashion-MNIST / 37.4% 8-word speech / 66% PathMNIST. Free open models
  beat all of these by wide margins at 100–500x the parameters. TinyStories-style coherence exists at <10M
  params, but **only inside a narrow synthetic domain**.
- **Scale:** 8M "small" until you notice "small" now means 2–4B (Gemma 2 2B, Phi-3-mini).
- **Capacity ceiling:** ~40 concepts and max-3-concept rules bound expressivity by design. Good for audit,
  disqualifying for almost any real task.
- **Novelty:** CBM (2020) already did features→concepts→answer. The multi-modal coverage + pass/fail gates
  is a modest, defensible delta — not a paradigm.
- **Distribution:** no users, no benchmark standing, no community, no hub on HF. That is the binding
  constraint, not the model.

*Unverified / low confidence:* WhyLabs current ownership and funding; Arthur AI's $63M (secondary source
only); Logical Intelligence's funding level; all market-size figures for "explainable AI" / "neuro-symbolic
AI" (vendor-commissioned reports, e.g. $1.62B→$2.13B 2026 for neuro-symbolic, $26.5B by 2035 for
XAI/transparency — directionally useful, numerically unreliable).
