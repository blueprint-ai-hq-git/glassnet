# METER-05 — NAME TYPES (pre-registered before it is used)

**PRE-REGISTERED: 2026-10-05, after Target C's results, BEFORE any run that uses it.**
**Does not change Meter-03, and does not change Target C's recorded R2 = 0/6.** C's result
stands exactly as measured. This rule applies to runs that declare their name types up front.

---

## 0. Why a second rule is needed

Meter-04's R2 asks one question of every name: *"is each word individually load-bearing?"*
That question is correct for one kind of name and wrong for the other:

```
KING_What_you,_RICHARD   -> KING drop 0.0000, RICHARD drop 0.0000
                            Those tokens are in NONE of the firing windows.
                            The name makes a FALSE claim. R2 catches it. CORRECT.

is_a_negation            -> never / nor / none / nothing each drop < 0.10
                            but removing the whole set destroys 99.65% of the neuron.
                            The name makes a TRUE claim. R2 fails it. WRONG INSTRUMENT.
```

`is_a_dialogue_line` makes this unanswerable as a per-word question: it names a 101-member
category, and removing any single one of 101 members **cannot** cost 10% by construction.

## 1. The rule

Every concept name is declared, **in the pre-registration, before the run**, as one of two
types:

| type | what the name asserts | test |
|---|---|---|
| **SPECIFIC** | particular tokens ("these words") | **per-word ablation** — every word must drop ≥ 0.10 |
| **CATEGORICAL** | an idea ("this kind of thing"); the members are examples | **category ablation** — remove the whole named set |

### 1a. SPECIFIC names
Unchanged from Meter-04 §1: the name passes iff **every** word in it, ablated alone, costs the
column ≥ 10% of its own mean firing activation.

### 1b. CATEGORICAL names
Remove **the entire named set** at once and measure the relative drop `d_all` in the column's
mean firing activation. Compare against **20 control draws**, each removing the same *number*
of unrelated words. Let `z = (d_all − mean(control)) / sd(control)`.

**A categorical name passes iff `d_all ≥ 0.25` AND `z ≥ 3.0`.**

**Threshold justification, written now:** 0.25 because a category that supplies less than a
quarter of its own neuron's firing activation is not what that neuron is about; 3.0 because the
control removes the same number of words, so it captures size effects, and 3σ is the point at
which the named set is doing something the same-size random set is not. Both are deliberately
modest — this rule is meant to reject a name whose idea is **absent**, not to rank concepts by
strength.

### 1c. The false-word check applies to BOTH types
**No name may contain a word that appears in none of the column's firing windows.** This is
kind-independent and it is the check that actually caught `KING` and `RICHARD`. It is reported
for every concept under either type.

## 2. What this rule may NOT be used for

* **It may not re-score Target C.** C's R2 = 0/6 is the recorded result under the rule frozen
  before it ran. Re-scoring an existing result under a new rule is post-hoc bar-moving.
* **It may not re-score A3, A2, or A.** Same reason.
* It applies only to runs that declare their name types **before** running.
* A concept's type may not be changed after seeing whether it passed.

## 3. Effect on the readability score

The score stays a **count**, and it now reads:

`R1 (audit) · R2/R5 (honest name, by type) · R3 (a human recognises the idea) · R4 (a reader
can predict the firing from the name)`

For Target C, reported as: **R1 6/6 · R2 0/6 · R3 6/6 · R4 pending** — with the category
diagnosis reported *alongside*, clearly labelled a diagnosis, not a re-score.
