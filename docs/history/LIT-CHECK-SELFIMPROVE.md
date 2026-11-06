# LIT-CHECK — SELF-IMPROVEMENT BY DISTILLATION

**Date:** 2026-10-05. **Method:** arXiv API lookups (`export.arxiv.org`) for every anchor paper,
plus keyword searches for the specific angle. **Nothing here is cited from memory or from a
search-result snippet** — each anchor was confirmed by direct `id_list` lookup, with title and
authors printed back.
**Scope, stated honestly:** this covers the arXiv-verified record as of 2026-10-05, plus two
web searches. It is **not** an exhaustive crawl — one of the two searches failed with a
backend 403 and is reported as a gap, not silently dropped.

**Owner's goal being checked:** *"take a model and then we can make data and put other model and
then it can learn and get much smarter than the model it was from."*

---

## 1. The anchors, verified by direct lookup

| paper | arXiv | authors | date | verified title |
|---|---|---|---|---|
| Born Again Neural Networks | `1805.04770` | Furlanello, Lipton, Tschannen et al. | 2018-05-12 | "Born Again Neural Networks" |
| Be Your Own Teacher (self-distillation) | `1905.08094` | Zhang, Song, Gao et al. | 2019-05-17 | "Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation" |
| STaR | `2203.14465` | Zelikman, Wu, Mu et al. | 2022-03-28 | "STaR: Bootstrapping Reasoning With Reasoning" |
| The Curse of Recursion (model collapse) | `2305.17493` | Shumailov, Shumaylov, Zhao et al. | 2023-05-27 | "The Curse of Recursion: Training on Generated Data Makes Models Forget" |
| On the Impact of Knowledge Distillation for Model Interpretability | `2305.15734` | Han, Kim, Choi, Yoon | 2023-05-25 | ICML 2023 |

Search-surfaced (IDs as returned, **not** individually verified — treat as pointers, not
citations): OPSD `2601.18734`, DualOPSD `2608.26019`, "What Should a Self-Teacher See?"
`2609.25623`, "Student Capacity Moderates Knowledge Distillation Effectiveness" `2605.31191`.

## 2. What is already known — labeled honestly

**ESTABLISHED — a model can generate its own training data and improve.**
STaR (`2203.14465`) and self-distillation (`1905.08094`) both do exactly this. *Our proposed
mechanism is not new.*

**ESTABLISHED — a student can end up better than the model it came from.**
This is the owner's exact headline, and it is a **2018 result**: Born Again Networks
(`1805.04770`). *The claim "student beats teacher" is not new.*

**ESTABLISHED — training on self-generated data can make a model WORSE.**
The Curse of Recursion (`2305.17493`) is the counterweight: recursively training on generated
data degrades the model. So self-improvement is **not** guaranteed by construction — it is the
thing being tested, and the literature says it can go either way.

**ESTABLISHED — distillation and interpretability interact.**
Han et al., ICML 2023 (`2305.15734`) measured **concept detectors via network dissection** and
found distillation **improves** interpretability, attributing it to class-similarity information
transferred through logits. This is the closest published work to our angle.

**DEBATED — whether self-improvement compounds or collapses.** STaR-side results say it
compounds with filtering; collapse-side results say recursion degrades. The difference is
generally data selection and verification, which is exactly why the 2026 OPSD line
(`2601.18734` and follow-ups) concentrates on *which* self-generated trajectories to learn from.

**OPEN — the specific angle we actually want.**
No verified paper was found that does self-improvement **through a pre-registered, audited
concept layer** — where the student's concepts are human-named, must pass a fidelity audit, and
where the *share of the model's reasoning carried by named concepts is measured*. Han et al.
go the other direction: they distil first and detect concepts afterwards by dissection. They do
not design, name, pre-register or audit the concepts, and they do not measure reasoning
coverage.

## 3. Novelty verdict for our specific claim

Narrowest form of what we could claim: *"self-improvement where the student reasons entirely
through named concepts that are audited, and where readability coverage is a measured
quantity."*

**Verdict: MINOR EXTENSION at best — not yet "genuinely new."**

Reasoning, against the anti-oversell checklist:
* The **mechanism** (self-generated data → train a second model → it beats the source) is a
  **replication** of `1805.04770` / `2203.14465`, not a discovery. We must say so.
* The **readable-concept framing** is adjacent to `2305.15734`, which already links distillation
  and concept-level interpretability. Our difference is design-and-audit rather than
  post-hoc dissection, plus a coverage metric — **a real difference, but a modest one.**
* **We have not verified a gap** in the way the skill requires for "genuinely new": I can say
  "no verified paper does exactly this", but a keyword search is not proof of absence, and one
  search backend returned 403.
* We have no result yet. Novelty cannot be claimed before there is something to compare.

## 4. What this changes about the plan

1. **We must not present the self-improvement method as new.** It is a replication of a known
   family, and the honest framing is "known mechanism, our own from-scratch implementation,
   through a readable layer."
2. **The interesting, defensible claim is the readable-layer one** — and it only becomes a real
   claim if we can show something the literature has not: for example that a student trained
   through audited concepts is **more** readable than one distilled normally, at equal accuracy.
   That is a measurable, falsifiable claim and it is not in the papers found.
3. **The collapse literature is a warning we should design against, not ignore.** Data
   selection matters (`2305.17493` vs the STaR line). Our design should pre-register a selection
   rule for self-generated data **before** the first cycle, exactly as we pre-register training
   runs.
4. **Our own evidence base already bears on this.** Target C/D/E measured exactly what the
   literature does not: readable coverage of 6.40% (C) vs 100.0000% (D, E), and the accuracy
   cost of forcing it (BEHIND / PARITY). That measurement is a more solid contribution than the
   distillation mechanism, and it is already recorded.

## 5. Open questions this check leaves

* Is there published work on **audited** concept layers (pre-registered names + fidelity
  audit + lie-probe) rather than post-hoc dissection? Not found; not proven absent.
* Does self-distillation **preserve or erode** readability across cycles? Unaddressed by
  anything found, and it is directly testable here — and it is a real risk: nothing guarantees
  the concepts survive a second generation.
* Does a readable student avoid the collapse failure mode, or is it more exposed to it?
  Untested.

**Next step, per the owner's rules: pre-register the design and its predictions before any run.**
