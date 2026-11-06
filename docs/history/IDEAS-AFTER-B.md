# IDEAS AFTER TARGET B — five pre-specified changes, one at a time

**Written:** 2026-10-05, BEFORE any of them runs.
**Authority:** owner mission — "IF TARGET B IS NOT MET: You may try up to 5 more improvement
ideas, ONE change at a time. Write the idea and the prediction BEFORE each run. Each one
must keep the audit clauses passing. After 5, or after 15 iterations with no real
improvement, stop and write STOPPED.md."

**Status: none started. Each entry below is the prediction that will be on record before its
run.** Each idea changes exactly ONE thing, keeps the same task, the same matched-parameter
normal, the same budget, the same frozen selection rule, validation only, Meter-03 unchanged,
test set untouched.

**Why these five, and why in this order.** The STEP 0 readability sheet found that **6 of the
10 taught A3 concepts are single-word detectors** and only **2** are real multi-word ideas.
The concept layer sits in the middle of the forward pass, so if it is carrying words rather
than ideas it is plausibly a bottleneck — which is a concrete, testable explanation for
readable being marginally *behind* on both metrics. Ideas 1–3 and 5 therefore attack **what
the concepts are**; idea 4 attacks **how many** there are.

---

## Idea 1 — MULTI-WORD CONSTRUCTION TARGETS

**One change:** the targets become multi-word constructions instead of single-word clusters.
Derivation: TRAIN only, frequent 2–4 word n-grams with TRAIN support ≥ 2.5% (the threshold
A3 established), deduplicated, frozen in a file before use.

**Motivation, from the data:** the only two concepts that read like real ideas in the sheet —
`for such a one as` and `gave him` — are exactly the multi-word ones.

**Prediction (before running):**
* The readability sheet's "meaningful ideas" count rises **well above 2 of 10** — I expect
  **6 or more of 10** to be genuine phrases.
* **The audit pass fraction may FALL.** A real construction is harder to detect perfectly
  than the token `The`, which reaches AUROC 1.0000 precisely because it is trivial. I predict
  **50–80%** of taught concepts still pass, i.e. A3's 80% is not safe.
* **The margin is the thing that matters.** I predict readable's perplexity gap **narrows**
  versus A3's −1.50, but I do **not** predict it turns into a win.
* **Falsifier:** if the concepts become nameable and the margin does not move at all, then
  the concept layer's contents are not the bottleneck, and the "words not ideas" explanation
  is wrong.

## Idea 2 — SPEAKER-NAME TARGETS

**One change:** concepts become **who is speaking**. The corpus carries speaker labels
(`TRANIO:`, `PETRUCHIO:`, `GRUMIO:` …), so "this window is a TRANIO line" is a genuine,
human-nameable idea that a person would recognise instantly.

**Prediction (before running):**
* Nameability is essentially guaranteed — the name IS the concept. Readability sheet count
  should reach **10 of 10**.
* Detection should be **strong**: speaker identity is highly distinctive, so I predict
  **AUROC ≥ 0.9** for the major speakers and a pass fraction of **≥ 70%**.
* Firing rates should be healthy for major speakers, so the 2% floor should not bite — but
  **minor speakers will be thin** and will fail it, exactly as in A2. I predict the pass
  fraction is limited by **speaker frequency**, not by detection.
* **On the margin:** this is the idea I consider most likely to help, because knowing the
  speaker is genuinely predictive of the next words. I still do **not** predict a clean win.

## Idea 3 — CLEAN PRONOUN TARGETS

**One change:** a single, clean pronoun concept (or a small set: subject pronouns, object
pronouns) instead of A3's cluster `[4]`, which mixes `they/us` with the conditionals `If/let`.

**Prediction (before running):**
* The name becomes honest — this directly addresses the mixed-cluster problem the sheet
  flagged in concepts `[0]`, `[3]` and `[4]`.
* AUROC should be high (pronouns are frequent and syntactically marked), firing healthy, so
  I predict a **pass** — but this idea is about **name honesty**, not about the margin.
* **I predict NO measurable margin improvement.** A pronoun concept is easy but weakly
  predictive, so if the margin moves it would be surprising and I would look for a confound.

## Idea 4 — MORE MEANINGFUL TARGETS AT HIGHER K

**One change:** raise the number of concept columns (K = 16 → 24 or 32) **while** keeping
meaningful targets. Everything else fixed.

**Prediction (before running):**
* More concepts means more of them are thin, so the **per-concept pass fraction should FALL**
  — consistent with the earlier K sweep, where smaller K was worse on the audit but larger K
  did not fix it either.
* The margin may improve slightly from added capacity; I do **not** predict it clears the 2σ
  win bar.
* **Honest risk:** this is the idea most likely to produce a *worse* audit with no margin
  gain, i.e. the most likely to be a wasted run. It is last-but-one for that reason.

## Idea 5 — QUESTION TARGETS

**One change:** a concept for "this window asks something", using wh-words
(`what/why/how/when/where/who`). **Must first be checked:** the tokeniser may have stripped
`?`, in which case the concept is really "contains a wh-word", and it must be **named that
way honestly** rather than called "question".

**Prediction (before running):**
* Nameable and easy to explain, but **likely diluted**: wh-words are common in non-questions
  too, so I predict a **moderate** AUROC (~0.7) and a pass.
* **I predict NO margin improvement.**

---

## Rules for all five (pre-recorded)

* **ONE change at a time.** Everything else stays exactly as it was.
* **The prediction is written before the run** — the text above is that prediction; it will
  be copied into the run's own pre-registration file so it cannot drift.
* **Every idea must keep the audit clauses passing** (≥50% of taught concepts, lie probe
  caught live, random control <5%). An idea that breaks the audit is a failure even if the
  margin improves.
* **A win must still be a win by the original definition**: readable beating the normal on
  **both** perplexity and accuracy by more than 2× pooled std. Nothing here softens that.
* **Stop after 5 ideas, or after 15 iterations with no real improvement**, and write
  `STOPPED.md` with an honest conclusion and the best next idea.
* **Targets F, D and E are not started.**
* Failed runs are kept. No bar changes. No "passed in spirit". Parity is never a win.

## The honest expectation

**I predict most of these will fail to produce a Target B win.** The A3 evidence says readable
is at parity or marginally behind, and the concept layer's contents are one plausible cause
among several. Ideas 1 and 2 are the ones with a real mechanism behind them; 3 and 5 are
about name honesty rather than performance; 4 is the least promising. If all five fail, the
honest conclusion is that **readability's cost on this task is roughly one perplexity point
and a fraction of an accuracy point**, and that is a result worth reporting rather than
hiding.
