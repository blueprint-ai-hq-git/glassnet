"""record_r4.py — record the R4 reader-test result EXACTLY as the owner wrote it.

Appends to LOG.md and QUEUE-AFTER-B.md. Does NOT touch the cards, the answer key, the rules,
or anything belonging to Target B.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-05 19:05 — R4 READER TEST (Target C cards) — recorded exactly as the owner wrote it

> READER TEST (R4), Target C cards. Scored by an AI reviewer (an external reviewer, NOT a human,
> and NOT the agent): 60/60, with all six concepts at 10/10. R4 = 6 of 6 passed by this reader.

**CAVEATS (recorded with it, not dropped):**

> 1. The reviewer was not blind to the project. Count this as an informed AI reader's score, not
>    a fresh human one. A human score is still PENDING.
> 2. Concept 0 (pronoun_subject): every window contained a subject pronoun, so the name alone
>    could not separate firing from non-firing. The reviewer tie-broke 50/50 by "ends in I vs
>    ends in thou" and was right. The neuron fires on windows ending in "I", not on all subject
>    pronouns, so the NAME OVERCLAIMS. Concept 1 was also partly luck.
> 3. The decoy windows are easy (most lack the key words). 10/10 on concepts 2 to 5 is expected.
> 4. R4 only tests whether a reader can predict firing from the name. It does not test whether
>    the concept matters to the model's answers.
> 5. The final readability count needs R1 to R3 as well. Do NOT report a final readability number
>    from R4 alone.

**Standing instructions from the owner, in force from here:**

* **Do not edit** the cards, the answer key, or the rules.
* **Do not change Target B.**
* **Do not claim "readable" anywhere without the R1-to-R4 count AND the caveats above.**
* The three follow-up ideas below go into `QUEUE-AFTER-B.md` as **IDEAS ONLY**, pre-registered
  before any use.

**Caveat 2 is a finding in its own right, and it agrees with the earlier measurement.** The
owner's reading — the neuron fires on windows *ending in "I"*, not on all subject pronouns —
is the same class of defect that `METER-04`'s R2 was built to catch: **a name that claims more
than the neuron does.** Target D later confirmed the concept is fragile (`pronoun_subject` AUROC
0.9982 in C -> 0.5119 in D). It is recorded here as an open defect, not a passed check.

**Where Target C's readability stands, as a count (never a vibe):**
`R1 6/6 · R2 0/6 · R3 6/6 · R4 6/6 by an informed AI reader (human PENDING)` — plus the five
caveats above. **No final readability number is reported, and none may be inferred from R4.**
"""

QUEUE = """
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
"""

for path, text in ((os.path.join(PROJ, "LOG.md"), LOG),
                   (os.path.join(PROJ, "QUEUE-AFTER-B.md"), QUEUE)):
    with open(path, "a") as f:
        f.write(text)
    print(f"appended {len(text)} chars to {os.path.basename(path)}")
