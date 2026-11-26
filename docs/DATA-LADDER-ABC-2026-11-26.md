# 2026-11-26 - the data ladder on the real model: loss gains flatten at rung C while ACCURACY gains grow

Source: lens/runs/OLD-MODEL-REPLAY_poolA/_poolB/_poolC.json, journals of units glassnet-datalever3 and glassnet-rungc. Model
direct1024_d512 (47.5M parameters). Every rung scored on the CANONICAL held-out set (lens/runs/OLD-HELDOUT-CANONICAL.npz, 4,000
windows, seed 20261126), so all four rows below are directly comparable - baseline 6.9218 nats / 13.33% for every arm.

    rung   distinct windows   steps     after           gain vs before        seeds
    A        200,000           3,000    6.7375 nats      +0.1843 nats   +0.05 pts   2
             13.38%
    B      2,000,000          30,000    6.1032 nats      +0.8187 nats   +0.29 pts   2
             13.62%
    C      8,000,000         120,000    5.9772 nats      +0.9446 nats   +1.65 pts   1
             14.97%

    step to step:  A->B  -0.6343 nats and +0.24 points
                   B->C  -0.1260 nats and +1.35 points

## My prediction, and it was wrong in an instructive way

I wrote: "rung C lands between 0.2 and 0.4 nats better than rung B, i.e. the gain shrinks from 0.63 to roughly 0.3. FALSIFIER: if C
is within 0.1 nats of B, the lever has flattened."

C came in **0.126 nats** better than B - below my predicted range, and just outside the 0.1 falsifier bar. So on LOSS the lever is
close to flattening, faster than I predicted.

But on ACCURACY the opposite happened: A->B bought 0.24 points, B->C bought **1.35 points**, and C sits 1.65 points above the
starting model. **The loss is nearly flat while the accuracy is still climbing steeply.** A 0.126-nat loss gain buying 1.35
accuracy points means the model was re-ordering its predictions rather than sharpening them - the probability mass it needed to
spend to move the argmax was already available.

## Which number should be trusted

Accuracy is the metric that matters for a chat model, and it is the one still moving. But C is **one seed**, and today has already
shown that a difference under 0.2 points is not distinguishable without seeds. The B->C accuracy gain of 1.35 points is large
enough to survive that, but it must be replicated before it is quoted as a law.

## Practical conclusion

Ten times the data still buys something at rung C, and what it buys has moved from loss to accuracy. The next rung should be
judged on accuracy, not loss, and it should be run with at least two seeds. Rung D (or a second seed of C) is the next
measurement, not a rung E.
