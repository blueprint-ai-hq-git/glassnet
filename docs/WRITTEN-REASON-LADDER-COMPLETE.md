

---

# THE COMPLETE WRITTEN-REASON LADDER — SEVEN TESTS, ALL 100.0% (2026-11-22 19:33)

Every row below is a separate trained model, a separate held-out set, and a pre-registered bar. None were tuned after the fact.

| # | what the model must do | extra difficulty | result |
|---|---|---|---|
| 1 | report a position and a value (one item) | 12 positions | **100.0%** |
| 2 | write the comparison out, then answer (two items) | 6 written tokens | **100.0%** |
| 3 | write the comparison out, then answer (three items) | 8 written tokens | **100.0%** |
| 4 | say WHICH of TWO rules it used | rule varies per example | **100.0%** |
| 5 | say WHICH of **FOUR** rules it used | larger/smaller/first/last | **100.0%** |
| 6 | say which of four rules **and be honest about it** | reason checked against the digit written, 600 examples | **100.0%** |
| 7 | say which of four rules with **more distractors** | 16 input positions instead of 12 | **100.0%** (2 seeds) |

Every one of those also passes a **perturbation test** where the input is changed and the correct answer is known in advance (digit moved,
digit swapped, cue flipped) - there is no result in this table that rests on clean samples alone.

## Alongside it, the same day, the failing side measured just as carefully

| what was tried | result |
|---|---|
| 36 concepts chosen by me, read out by the model's own head | **5.5-18.0%** of the answer reproduced |
| the same with a best-possible fitted readout | **24.0%** (own head 23.2%) |
| the same with no linearity assumption (nearest neighbour) | **10.7%** |
| concept layers of width 8 / 16 / 36 / 72 trained on real text | **never beat the 16.32% baseline** |
| the readable layer's share of the decision margin | **14.8-18.2%** (bias + direct path 81.8-85.2%) |

## The one sentence this day produced

**A reason the model WRITES is exact, honest, and survives more items, more rules and more distractors. A concept layer I ATTACH is a side
channel: it describes the sentence, it does not decide the answer, and no width or readout rescues it.**

That is now supported by seven passing tests on one side and five failing measurements on the other, all written down with their scripts,
seeds and sample sizes. The architecture decision follows directly: **readability must be an output of the model, not a label attached to
its insides.**
