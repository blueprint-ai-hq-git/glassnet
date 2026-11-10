# The readable half explains 7.3% of the model's decisions (2026-11-10)

The project's core promise is that a person can read *why* the model answered. This measures whether the readable
path actually decides anything, by splitting the logits the model already computes.

```
logits = concept path (36 readable numbers -> vocabulary)  +  copy path (echo of the window, tied to the embedding)
```

PREDICTION, recorded before measuring: the 36 concept rows of the answer layer would be strongly correlated, so the
answer layer reads a few shared directions instead of 36 separable features, which would explain why no single
neuron means anything nameable.

## Prediction falsified, cleanly
```
answer.W shape (36, 12002)   row norms 25.545 / 25.833 / 26.039  (min / median / max)
pairwise cosine between rows  mean -0.002   mean|.| 0.007   max +0.025
singular values              top-1 holds 3.0% of the energy, top-5 hold 15.0%
participation ratio          35.97 independent directions of 36
```
The concept rows are essentially orthogonal. Redundancy is not the explanation, and that hypothesis is closed.

## What the second measurement found instead
```
readable (concept) path vs copy path, 233 validation windows:
  logit variance from the concept path : 0.7
  logit variance from the copy path    : 2.3
  concept path ALONE picks the same token as the full model : 7.3% of the time
```
Per-concept energy on 369 validation windows: the top three concepts explain **13.0%** together, the bottom ten
explain 16.06%, so the 36 concepts each contribute a small, even slice of a path that is itself a third the size of
the copy path. Activation spread is mean 0.592, sd 0.169, so the neurons are not simply saturated dead.

**Conclusion: the readable path is nearly decorative.** When this model answers, 92.7% of the time the answer is
not what its 36 named concepts would have chosen. `explain()` can report which concepts fired, honestly, and that
report would not be the reason for the answer.

## Why this matters more than an accuracy point
Every honesty commitment in this project rests on the concept layer being the decision. It tracks its labels
(35/36 AUROC pass) and it does not drive the answer. Both things are true at once, and only the first was being
reported.

## What follows
1. **`explain()` must state the split.** Any explanation that omits "the decision came from the copy path" is
   misleading, even when every individual number in it is correct.
2. **The architecture has a choice to make, and it is a design decision for Yahya, not a tuning knob:** either the
   concept path is given the decision (and accuracy is paid for it), or it stays an accurate monitor of a decision
   it does not make (and the project stops calling itself readable at the answer level).
3. Nothing here changes the accuracy work. It changes what the words "readable" and "explain" are allowed to mean
   on the public page.
