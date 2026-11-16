
## Scale 10 on the structural term: 23.58% accuracy and the correction still carries 0.9877 (2026-11-16)

Predictions in `lens/runs/SCALE-2-PREDICTIONS.txt`, written before the run (re-aimed from 70 to 10 after a post-hoc pre-test showed 70 destroys
accuracy). Numbers from `lens/runs/SCALE.txt`.

```
final VAL accuracy        23.58%     (shipped 4M: 24.58%, so -1.00)
final VAL perplexity     160.87      (shipped 181.04, better)
concepts passing          52/53
median correction share   0.9877     (S1 predicted below 0.10)   -> FAILED
median ||structural||     2.1871     (unscaled arm: 1.5466)
median ||A_c B||         19.3273
median own-word mass      0.0012     (S2 predicted above 0.10)   -> FAILED
explain() max |residual| 1.260e-06   | concept share 1.000000     -> PASSED
```

## What scaling bought

**A fixed multiplier of 10 on the structural rows did what it was asked to do geometrically** - structural norm 1.5466 -> 2.1871 - and it moved
the correction's share by **0.012** (0.9999 -> 0.9877) while costing **0.75 accuracy points**. Own-word mass went from 0.0013 to 0.0012, i.e.
nowhere.

**So the fourth and last row-level lever is also measured and also fails**: a stronger structural term does not become the carrier of the
row, because the correction grows to match whatever the structural term does. The share is a ratio, and both sides of the ratio are under the
optimiser's control.

## The complete row-level table

```
arm                                   struct norm   corr norm   corr share   own-word mass   VAL
unpriced structural                      1.5466      87.42       0.9999        0.0008       24.33%
priced at 11% of the CE gradient         1.5466      27.16       0.9968        0.0013       23.66%
priced + structural scale 10             2.1871      19.33       0.9877        0.0012       23.58%
pure structural, no correction           -           -            0.0000        0.0019       13.74%
content term at 13% of the CE gradient   (running)                                       0.0006 at ep3
```

Four interventions, each moving its own quantity, none moving meaning above 0.002. **The pattern is consistent and it is the finding**: in this
architecture, a concept's answer row is whatever predicts best, and any pressure aimed at the row is absorbed by the other side of the ratio.
