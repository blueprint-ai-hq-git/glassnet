# Shipped-model verification: the explanation is exact, and the concept names are keyword detectors (2026-11-16)

Predictions in `lens/runs/SHIPPED-VERIFY-PREDICTIONS.txt`, written before the run. All three shipped checkpoints (4M windows, 12 epochs,
53 concepts, lam 1.0), measured through `lens.load()` and the library's own `explain()` - never a hand-built forward.

## Explanation exactness: PASSED, decisively

```
seed   max |residual|   mean |residual|   concept share (median / min)
 0       2.027e-06         5.490e-07          1.000000 / 1.000000
 1       2.354e-06         6.375e-07          1.000000 / 1.000000
 2       2.921e-06         6.107e-07          1.000000 / 1.000000
```

- **V1 PASSED**: worst residual across 300 windows is **2.9e-06**, three orders of magnitude inside the 1e-5 bar.
- **V2 PASSED**: concept share is **1.000000 on every window of every seed**. With no direct path, the named concepts account for the
  model's prediction to within rounding.
- **FALSIFIER NOT TRIGGERED**: nothing came near the 1e-3 that would have withdrawn the exactness claim.

**So `explain()` is not a story.** The arithmetic that it prints is the model's arithmetic, to six significant figures.

## Concept-name honesty: the hardest number in the project

```
seed   trained-label AUROC   held-out synonym AUROC   concepts below 0.60 (of 22)   trainer's own audit
 0          0.9990                 0.6398                    9                        52/53
 1          0.9992                 0.6184                   10                        50/53
 2          0.9994                 0.5930                   14                        52/53
```

- **V3 PASSED** (the band 0.50-0.75 was met: 0.6398 / 0.6184 / 0.5930).
- **V4 PASSED**: 50-52 of 53 concepts pass the trainer's own audit on all three seeds.

**But look at the gap: trained-label fidelity is now 0.999, while transfer to unseen synonyms is 0.59-0.64.** That gap is
**0.36 to 0.41** - the largest measured in this project, and it grew as the model got better data. The concepts have become
**near-perfect detectors of the specific words they were trained on, and only modest detectors of the idea those words stand for.**

## The two claims that must never be conflated again

1. **The explanation is exact** - "the model answered because of these 53 named numbers, and here is each one's contribution" is
   arithmetically true to six figures. **Verified.**
2. **The names mean what they say** - "this neuron detects questions" rather than "this neuron detects the words `?`, `what`, `how`" -
   is **true for about 0.59-0.64 of its own meaning, not 0.999**. **Not verified, and now measured.**

A release note may claim (1) without qualification. It may claim (2) only as "the concepts are honest about their trained vocabulary and
about 60% honest about the underlying idea". Anyone reading `explain()` output must be told that distinction, or the transparency promise
becomes a different kind of opacity.

## What the verification changes about shipping

The shipped model is now documented end to end, on the model that actually ships: 24.58% +/- 0.06 validation, perplexity 181.04,
coverage 0.343 (greedy) rising to a T=0.7 knee of 0.292 coverage with 0.027 repetition, explanation exact to 2.9e-06, concept share
1.000000, and concept naming 0.59-0.64 honest.

The next lever remains data: accuracy and corpus fidelity both improved monotonically from 600k to 2M to 4M windows and neither has
flattened. `lens/runs/DATA8M-PREDICTIONS.txt`.
