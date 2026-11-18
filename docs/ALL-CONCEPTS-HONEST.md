# Every concept is honest on every seed, and the accuracy gap is not resolvable (2026-11-18)

Two results from one 3-seed arm. The first is a milestone. The second corrects a claim this project has been repeating for days.

## The arm

`centre_heavier` had been the one dishonest concept: its formula fired on **96.8%** of images (threshold 0.05 on the centre-versus-border mass ratio, which almost any Fashion-MNIST image
satisfies), so the neuron - firing 39.2% of the time - was firing *where the claim was false*, scoring **AUROC 0.3196, below chance**. The fix was to move the threshold to **0.40**, which
drops the concept's support from 96.9% to 22.9% and makes the claim mean something. Re-run at the matched parity configuration, 3 seeds:

```
seed   READABLE   PLAIN    delta        concepts    centre_heavier AUROC   fire vs truth
  0    15,977     15,910   -0.14 pts     21/21            0.9676            28.0% vs 22.9%
  1    15,977     15,910   -1.75 pts     21/21            0.9412            34.0% vs 22.9%
  2    15,977     15,910   +2.26 pts     21/21            0.9327            37.9% vs 22.9%
```

Parameter counts differ by **0.42%**, so this is a matched comparison.

## Result 1: 21 of 21 concepts are label-honest on every seed

**0 concepts failing, on all three seeds** - the first time this project has measured that. `centre_heavier` went from **AUROC 0.3196 to a mean of 0.9472**, and its firing rate now tracks its
true support (28-38% against 22.9%) instead of contradicting it (39.2% against 96.8%).

The lesson is the one the doctrine already had: **a concept whose formula fires on almost everything cannot be detected by a neuron.** The defect was never in the model - it was in the
definition of the concept. Moving one threshold fixed a below-chance AUROC into a 0.95.

## Result 2: the readable-vs-plain accuracy gap is NOT resolvable

```
gaps (readable minus plain): -0.14, -1.75, +2.26
mean +0.12 points | sample sd 2.02 | SE of mean 1.17 | 95% interval [-3.11, +3.36]
```

**The interval includes zero**, so a 3-seed run at this scale cannot distinguish the readable model from a matched plain control. That is not the same as proving they are equal - with
sd 2.02 the run cannot resolve anything smaller than ~2.3 points - but it does mean the direction is not established.

**And that puts the earlier claims in context.** This project published single-seed gaps of **-1.47** (width21), **-1.46** (parity49) and **-1.53** (the original matched pair), and drew
conclusions about *why* the readable model loses. **All three of those numbers sit inside this arm's seed spread (sd 2.02).** A single-seed gap of -1.5 is an ordinary draw from a
distribution that averages +0.12. The earlier 8-seed image summary had already shown gap sd 1.22 with a mean of -0.30; this arm says the same thing more plainly.

So the honest status of the image readability line is: **the readable architecture is not demonstrably worse, it is not demonstrably better, and the mechanism questions asked about the
"gap" (width, per-unit cost, the constraint itself) were asked about a quantity that never had a resolvable sign.** Those experiments were not wasted - they were well-controlled and they
correctly failed to move a number - but their premise needs restating.

## What would settle it

The same 3-seed arm at **8-12 seeds**, where the SE of the mean falls to 0.6-0.7 points and a 1.5-point difference becomes resolvable. That is the only way to change this from "no
resolvable difference" into either answer.
