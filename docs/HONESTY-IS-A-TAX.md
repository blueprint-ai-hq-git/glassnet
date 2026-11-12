# Honesty is a tax paid every step, not a state the model holds (2026-11-12)

The 92.0% honesty configuration costs 3.18 points of accuracy against the same concepts at 1x. The obvious next idea was
to pay the tax only early: hold the concept loss at 5x for the first 800 steps so the concepts become honest, then release
it to 1x and let the token objective win the accuracy back.

**It does not escape the trade-off. It interpolates it.**

```
same 12 concepts, same 3000 steps        accuracy   honesty   ablation   readability price
concept loss 1x                            16.65%     71.8%     17.40%         0.75 pp
5x for 800 steps, then 1x                  15.28%     84.0%     17.75%         2.47 pp
concept loss 5x                            13.47%     92.0%     17.18%         3.71 pp
trivial baseline                            5.42%       -         -              -
```

**Prediction on record for the schedule: "honesty at or above 85%, accuracy at or above 15.5%". Both missed narrowly -
84.0% and 15.28% - and the miss is the finding: the schedule lands on the line between the two endpoints rather than
above it.**

## What that means
Roughly **1.2 to 1.4 points of accuracy per 6 points of honesty**, at every point on the curve. Honesty is not something
the model can be taught once and then keep for free; it is a constraint that costs accuracy in every step it applies,
because the token objective pushes the shared embedding away from whatever state made the concepts honest. Releasing the
weight lets the concepts drift back immediately.

This is the same mechanism, measured a third way on this project: earlier, freezing a pointer pathway protected nothing
because it read the shared embedding table that the corpus kept moving.

## So the honest answer to "can we have both"
At this scale and this architecture, no: the frontier is real and the schedule slides along it. The three defensible
configurations are

- **fastest**: 1x, 16.65% accuracy, 71.8% honesty, readability costs 0.75 points
- **balanced**: warmup, 15.28% accuracy, 84.0% honesty, readability costs 2.47 points
- **most honest**: 5x, 13.47% accuracy, 92.0% honesty, readability costs 3.71 points

and **92.0% is the only one that clears the brief's 85% honesty bar.**

## What would break the frontier rather than move along it
Two candidates, neither tried yet:
1. **Give the concepts their own embedding table**, so the token objective cannot drag the inputs the concepts read. The
   earlier attempt at this on a different pathway cost language accuracy, but the diagnosis there was the shared table.
2. **Make the concept layer a fixed computation instead of a learned one** (compute the label functions directly), so
   honesty is 100% by construction and only the rules are learned. That changes what the middle layer is, so it is a
   design decision rather than a tuning one.
