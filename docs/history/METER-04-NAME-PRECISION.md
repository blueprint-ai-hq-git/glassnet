# METER-04 — NAME PRECISION (R2) + THE READER TEST (R4)

**PRE-REGISTERED: 2026-10-05, BEFORE any scoring. May not be edited after seeing results.**
**Authority:** owner goal `GOAL.md` — "set the goal to make it 100% fully understand it".
**What this adds:** the two readability requirements the current audit is missing. It
**changes no existing bar** — Meter-03's three clauses (AUROC 0.55, contribution 0.25, firing
floor 2%) stay exactly as they are. R2 and R4 are **additional** requirements.

---

## 0. Why this exists

Today's "80% readable" is **8 of 10 concepts passing Meter-03**. But Meter-03 only asks *"does
this neuron detect the thing its name claims?"* — it never asks whether the **name** is honest
or whether a **human** could understand it. That gap is not hypothetical:

```
concept [5] name: KING_What_you,_RICHARD
its five highest-activation windows, all driven by one token:
  '<PAD> <PAD> <PAD> And, Will you, <PAD> you,'
  '<PAD> God save you, sir! TRANIO: And you,'
  'you, <PAD> you, I will marry you. Now,'
  KING / RICHARD / What appear in NONE of them.
Meter-03 verdict: PASS (AUROC 1.0000) -- because 'you,' IS in its target set.
```

**A name can be two-thirds false and still pass.** R2 closes that. R4 closes the deeper
question: whether the name tells a human anything.

---

## 1. R2 — NAME PRECISION

**Rule.** A concept's name must not contain words the neuron does not actually use.

**Definition of "uses", pre-registered — ABLATION, as the primary measure.**

For concept column `c` with pre-registered name words `W = {w1 … wk}`:

1. Identify the column's **firing windows**: activation > 0.5 (the same rule Meter-03 uses).
2. Let `A0` = the column's **mean activation over its own firing windows** (this is the
   column's own scale, so the test is scale-free).
3. For each word `w` in the name, **ablate it**: zero that word's embedding rows for the whole
   window and recompute the column's activation. Let `A_w` = the same mean over the same
   windows.
4. **Relative drop** `d_w = (A0 - A_w) / A0`.
5. **Word `w` is USED iff `d_w ≥ 0.10`** (removing it costs the column at least 10% of its own
   firing activation).

**Name precision** = the fraction of name words that are USED.
**R2 passes** iff **name precision = 1.0** (every word in the name is used).

**Secondary measures, reported but NOT part of the pass rule** (so the primary cannot be
quietly swapped for a convenient one):
* the **contribution share** of each word (cheap proxy, reuses Meter-03 machinery);
* the **detection** measure (AUROC on windows containing that word) — reported specifically to
  show where "uses" and "detects" disagree, since a column can use a word it does not detect.

**Threshold justification (written now, not after):** 10% is chosen because a word whose
removal costs less than a tenth of the column's own firing activation is, in plain terms, not
what the column is about. It is deliberately **not** a high bar — this rule is meant to catch
*absent* words like `KING`/`RICHARD` in concept `[5]`, not to demand that every word be
equally important. A stricter bar (25%, 50%) would reject legitimate multi-word concepts like
`for such a one as`, where a function word carries little of the signal.

**Applied to existing checkpoints. No retraining.** Old vs new reported side by side for A3
(and, where checkpoints exist, A2 and the earlier rungs). Expected direction: **pass counts
FALL**, because a stricter rule can only remove passes. **If the count rises, the
implementation is wrong and that must be investigated, not celebrated.**

**Explicit, and recorded before running:** R2 is a **new record going forward**. It must
**not** be used to retroactively re-judge A3. A3 was MET under the rule frozen before it ran.
Anything else is post-hoc bar-moving, which is banned.

## 2. R4 — THE READER TEST

**Rule.** Given **only the name**, an independent reader must be able to tell which windows the
neuron fires on.

**Procedure, fixed here before any reader sees anything:**

1. For each concept, build one **blind card**: the concept's name, plus **10 validation
   windows in random order — 5 the column fires on (activation > 0.5) and 5 it does not.**
   The card shows **no activations, no ordering hints, no labels.**
2. The card is shuffled with a **fixed seed (0)**, so the same card can be re-scored.
3. The reader marks, for each window, whether they believe the neuron fires on it.
4. **Score = accuracy over the 10 windows. Chance = 0.50.** Reported per concept and pooled.
5. **R4 passes for a concept iff accuracy > chance** — hmm, corrected: **iff accuracy ≥ 0.80**
   (8 of 10). Chance is 0.50; 0.80 is the point at which the name is genuinely informative
   rather than lucky. Written now, before any reader runs.

**Who reads — stated plainly because this is the one human-judged metric in the project:**

* **The official reader is the owner or the external reviewer.** Not the agent.
* **The agent must not be the official reader for its own model**, because it wrote the names
  and knows the answers; that would be circular.
* The agent MAY fill a card as a **sanity check only**, clearly labelled `NOT OFFICIAL`, to
  confirm the cards are well-formed (that the 5 firing windows are genuinely firing, etc.).
* Cards are written to disk as a self-contained artifact, so they can be filled in by a human
  or by the reviewer without any access to the model.

**Deliverable:** `READER-TEST-CARDS.md` — one card per concept, plus an answer key kept in a
**separate file** (`runs/t3/reader_test_key.json`) that the reader does not open.

## 3. What "100% readable" then means, as a number

For every concept column, all four must hold:

| # | requirement | measured by |
|---|---|---|
| R1 | passes the audit | Meter-03, unchanged |
| R2 | honest name | name precision = 1.0 (this file, §1) |
| R3 | a human recognises it as an idea | the readability sheet's honest label — not a word list |
| R4 | the name predicts the firing | reader accuracy ≥ 0.80 (this file, §2) |

**Readability score = the count of columns satisfying all four, out of the total.** Today it
is **not 8/10** — under R1–R4 it will be **lower**, and that lower number is the honest
starting point for the goal. The number must be reported as a count, never as a vibe.

## 4. What does not change

* Meter-03's three clauses, unchanged. No bar moved.
* Validation only. **Test set untouched.**
* The lie probe and the random-column control stay live.
* From-scratch rules, safe licenses only, no license chosen for GlassNet.
* Failed runs kept. Every number traces to an artifact.
* A3 stays MET; A2 and A stay NOT MET. R2/R4 do not revise them.

## 5. Outputs

`runs/t3/name_precision.json`, `READER-TEST-CARDS.md`, `runs/t3/reader_test_key.json`, and a
short old-vs-new report.
