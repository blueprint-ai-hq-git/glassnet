# QUEUE AFTER TARGET B — ideas only, NOT STARTED

**Written:** 2026-10-05 13:30
**Status: IDEAS ONLY. Neither item has been run, pre-registered, or coded.**
**Both require explicit owner approval before starting.** Nothing here changes Target B,
which is running exactly as written.

These two items were added on the owner's instruction while Target B's grid is in flight.

---

## Item 1 — NAME PRECISION CHECK (a new meter)

**The problem it addresses, concretely.** Concept `[5]` of the A3 model is named
`KING_What_you,_RICHARD`, but its top firings are all driven by the single token `you,`.
`KING`, `RICHARD` and `What` never appear in its five highest-activation windows. **It still
passes Meter-03**, because `you,` *is* one of its pre-registered target words — so the audit
is satisfied while the name remains partly false.

**The proposed rule.** A concept's **name must not contain words the neuron does not
actually use.** Name precision = the fraction of the words in the name that the column
actually uses; a name with unused words is imprecise.

**Design questions that MUST be settled in the pre-registration, before any scoring:**

1. **What does "actually uses" mean?** At least three defensible operationalisations, and
   they are not equivalent:
   * **ablation** — remove that word's embedding contribution to the column and measure the
     drop in the column's activation (closest to "uses", most expensive);
   * **contribution** — the word's share of the column's existing contribution sum (cheap,
     reuses the current meter's machinery);
   * **detection** — the column's AUROC/firing on windows containing that word versus
     windows without it (cheap, but a column can *use* a word it does not *detect*).
2. **What is the pass threshold per word?** Any number is arbitrary; it must be fixed before
   scoring and defended, not chosen after seeing the result.
3. **Is the name rejected wholesale, or are unused words pruned from the name?** Pruning
   changes the name (and therefore the target), which would break comparability with every
   earlier result unless reported both ways.
4. **Interaction with the existing bars.** Meter-03's three clauses stay exactly as they
   are. Name precision would be an **additional** clause, or a separate reported metric —
   that choice must be made in advance.

**What must be done when it is run:**

* **Pre-register the whole rule in its own file before scoring anything.**
* **Re-score EXISTING checkpoints** — no retraining. Report **old vs new, side by side**, for
  A3, A2, and the earlier rungs (Fashion-MNIST, PathMNIST, Banking77, the T2 runs).
* **Flag every earlier claim the new meter changes.** The expected direction is that pass
  rates **fall**, since a stricter rule can only remove passes. If the pass rate *rises*,
  something is wrong with the implementation and it must be investigated, not celebrated.
* Keep the lie probe and the random-column control, and confirm both still behave.

**Explicit warning, recorded now.** A stricter meter applied after A3 was declared MET must
**not** be used to retroactively re-judge A3. A3 was MET under the rule that was frozen
before it ran. A new meter creates a **new record going forward**, and the old-vs-new table
is a *report*, not a revision of a past verdict. Anything else is post-hoc bar-moving, which
is banned.

**Effort estimate:** moderate. Re-scoring is CPU-only and cheap; the work is in the
pre-registration and in the "uses" definition.

---

## Item 2 — MEANINGFUL TARGETS (real ideas instead of single words)

**The problem it addresses.** The STEP 0 readability sheet (`READABILITY-SHEET.md`) found
that **6 of the 10 taught A3 concepts are single-word detectors** (`And`, `The`, `you,`,
`them`, `My/know`, `they/If`), and only **2** are genuine multi-word constructions
(`for such a one as`, `gave him`). A concept layer carrying mostly single words is a
plausible reason readable is marginally behind the normal on both metrics.

**The proposed change.** Replace the cluster targets with **targets that are real ideas**,
for example:

* **speaker names** — the text has speaker labels (`TRANIO:`, `PETRUCHIO:`, `GRUMIO:` …), so
  "this window is a TRANIO line" is a genuine, human-nameable idea;
* **pronouns** — a clean pronoun target rather than a cluster that mixes pronouns with
  conditionals;
* **questions** — windows that ask something (wh-words: `what/why/how/when/where/who`; note
  the tokeniser may have stripped `?`, which must be checked before designing this one).

**A prediction must be written BEFORE the run.** The shape it will take, recorded here so it
cannot be presented later as an after-the-fact insight:

* **Expected:** the concept layer's concepts become genuinely nameable, and the readability
  sheet's "meaningful ideas" count rises well above 2 of 10.
* **Expected, and this is the honest risk:** the **audit pass fraction may FALL**, because a
  real idea is harder to detect perfectly than a single frequent token — `The` reaches AUROC
  1.0000 precisely because it is trivial.
* **The real question is whether readable's perplexity/accuracy margin improves** when the
  concepts carry ideas instead of words. That is the measurement that matters, and it is the
  one Target B already sets up.

**Constraints that carry over:** same task, same budget, matched-parameter normal, validation
only, Meter-03 unchanged (or with the Item 1 clause only if that has been separately
approved), test set untouched, one change at a time, failed runs kept.

**Effort estimate:** moderate-to-large. It needs new target derivation (TRAIN only, frozen
before use), a fresh pre-registration, and a fresh grid.

---

## Ordering note

**Item 2 is the more likely of the two to move the actual result**, because it changes what
the concepts *are*, whereas Item 1 changes how strictly names are *judged*. Item 1 is
nevertheless valuable on its own: it closes a real hole (a name can be partly false and still
pass), and it is cheap because it needs no training.

**Neither item is started. Neither is pre-registered. No result exists for either.**

---

## Added 2026-10-05 19:05 — three items from the owner's R4 instructions (IDEAS ONLY)

**Status: IDEAS ONLY. Not started, not pre-registered for use. Each requires the owner's
approval, and each must be pre-registered in its own file before it is ever run.**
Recorded here by the owner's instruction, alongside the earlier NAME PRECISION CHECK and
MEANINGFUL TARGETS items.

### (a) HARDER READER TEST
Decoys that almost match — for example windows that contain a subject pronoun but do **not**
fire. Same bar: **8/10**.

*Why:* caveat 3 of the R4 result — the decoys used so far are easy, because most of them lack
the key words entirely, so 10/10 on concepts 2-5 is expected rather than informative. A near-miss
decoy set is the only version of R4 that can actually separate a real name from a lucky one.
*Open question to settle first:* whether the near-miss decoys are drawn from the same support
region (e.g. windows containing the target words with low activation) or from a neighbouring
concept's firing windows. These test different things and are not interchangeable.
*Prediction shape, recorded now:* concept 0 (`pronoun_subject`) should drop sharply, because
caveat 2 already showed its name overclaims; concepts 2-5 should hold.

### (b) HUMAN SCORING
Cards for a **human** reader, **2 cards minimum**, scored by the existing key.

*Why:* caveat 1 — the 60/60 came from an **informed AI reviewer who was not blind to the
project**. The human score is **PENDING** and is the only one that answers the original question
("can a human understand this?"). The AI score is not a substitute for it.
*Constraint:* the cards and the key must not be edited for this (owner's instruction). A human
reader uses the existing cards, or new ones generated under a new pre-registered file.

### (c) NAME PRECISION FOR `pronoun_subject`
**The name must match what the neuron actually does.**

*Why:* caveat 2 — the neuron fires on windows **ending in "I"**, not on all subject pronouns.
The name `pronoun_subject` therefore **overclaims**.
*Constraint:* this may **not** be handled by quietly renaming the concept after the fact. Under
`METER-05`, a name is SPECIFIC or CATEGORICAL and the type is declared before the run; a name
that overclaims fails R5/R2 rather than being re-labelled. Any new rule for "name must match
behaviour" must be **pre-registered before use**, and it may **not** re-score Target C, D, A3,
A2 or A.
