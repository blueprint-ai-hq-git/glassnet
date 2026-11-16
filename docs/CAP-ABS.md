
## Second confirmation: the absolute cap flips the share to 0.1830 and meaning stays at 0.0018 (2026-11-16)

Predictions in `lens/runs/CAP-ABS-PREDICTIONS.txt`, written before the run. Numbers from `lens/runs/CAP-ABS.txt`. This is an independent
mechanism from the hard cap - a cap expressed in ABSOLUTE Frobenius units rather than as a multiple of the structural norm - and it lands on
the same answer.

```
final VAL accuracy        15.35%     (unpriced structural arm: 24.33%, shipped 4M: 24.58%)
final VAL perplexity     361.24
concepts passing          47/53
cap decisions logged      11,250
median correction share   0.1830     (A1 predicted 0.30-0.90)   -> below the band, i.e. the cap bit harder than predicted
median ||structural||     5.6159
median ||A_c B||          2.5393
rows with correction >50% : 4 of 53   (unpriced: 53 of 53)
median own-word mass      0.0018     (A3 predicted below 0.01)  -> PASSED
explain() max |residual| 1.245e-06   | concept share 1.000000    -> PASSED
```

## Two independent caps, one answer

```
cap mechanism                    corr share   rows corr>50%   own-word mass   VAL
none (unpriced structural)         0.9999        53 of 53        0.0008       24.33%
HARD cap at 0.5 x structural       0.2087        10 of 53        0.0016       14.48%
ABSOLUTE cap at 8.0 Frobenius      0.1830         4 of 53        0.0018       15.35%
```

**Both caps succeeded on their own terms** - shares of 0.2087 and 0.1830, rows dominated by the correction cut from 53 to 10 and 4 respectively -
and **both left own-word mass at 0.0016-0.0018**, three orders of magnitude below the 0.10 bar, for 9-10 accuracy points each.

**A3 was the prediction I wrote in advance**: *"own-word mass stays below 0.01 - I expect this, because five mechanisms have already failed and a
cap in different units should not change what the row contains."* It passed. The falsifier (mass above 0.05 with the cap holding) did not fire.

## What two independent confirmations establish

The correction's dominance of a concept's answer row is **not** what suppresses the row's meaning. Two different caps, expressed in different
units, both made the structural term the dominant carrier, and neither produced a row that favours its own words. Combined with the earlier
arms spanning correction shares from 0.0000 to 0.9999 and leaving mass inside 0.0008-0.0019 throughout, the conclusion holds:

**In this architecture at 16M parameters, what a concept's answer row CONTAINS is not controlled by what the row does, how large it is, what
supports it, or what dominates it.** The row is a prediction device; its name is a handle attached to it afterwards.

The one thing still standing is the supervision change: predicating a concept on its own word being the next token reached **25.7%** validation
at epoch 8 - the only arm to beat the shipped 24.58% - with its own mass creeping 0.0006, 0.0007, 0.0008. Its endpoint is the last open question
of the day.
