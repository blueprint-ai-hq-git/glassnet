
## The supervision change buys accuracy and does not buy meaning: next-token concepts reach 25.97% VAL, the best in the project, with own-word mass back at 0.0008 (2026-11-16)

Predictions in `lens/runs/NEXT-TARGET-PREDICTIONS.txt`, written before the code changed. Numbers from `lens/runs/NEXT-TARGET.txt` and
`logs/exposure/nodirect1024_nexttarget_data4m_e12_s0.log`. This was the last mechanism standing, because it acts on what a concept is asked to
PREDICT rather than on its answer row.

```
final VAL accuracy        25.97%     (ep 11: 26.01% - best in the project)
final VAL perplexity     169.67      (shipped 4M: 181.04)
train accuracy           29.81%
own-word mass            0.0008      (N1 bar 0.10; free-matrix baseline 0.0008)
concepts at or above 0.05 0 of 53     (highest: is_a_request 0.003)
explain() max |residual| 2.384e-06   | concept share 1.000000   -> N4 PASSED
```

## Both halves of the prediction

**N2 was exceeded on the good side.** I predicted 15.0-22.0% because the label becomes a rarer, sharper event; the model reached **25.97%**,
**1.39 points above the shipped 24.58%** and better perplexity. Asking a concept whether its own word is the token being predicted is a *better*
training signal than asking whether its words appear anywhere in the window.

**N1 failed and the falsifier fired.** Median own-word mass is **0.0008** - exactly the free-matrix baseline. The mid-run creep (0.0006 at
epoch 2, 0.0007 at epoch 5, 0.0008 at epoch 8) **plateaued at the baseline and went no further** in epochs 9-12. So even supervision that
couples a concept's firing to its own word being imminent does not make its row favour that word.

## The concept-content question, closed

Every mechanism this project can bring to bear has now been tested:

```
mechanism                                  own-word mass   VAL
free answer matrix (shipped 4M)               0.0008       24.58%
structural rows + unpriced correction         0.0008       24.33%
structural rows + priced correction           0.0013       23.66%
structural rows + structural scale 10         0.0012       23.58%
structural rows + HARD CAP (share 0.2087)     0.0016       14.48%
structural rows + ABSOLUTE cap (share 0.1830) 0.0018       15.35%
pure structural, no correction                0.0019       13.74%
row-content supervision at 13% of CE          0.0008       23.73%
next-token concept supervision                0.0008       25.97%
```

Nine arms. Correction shares from 0.0000 to 0.9999. Supervision targets changed from "present in the window" to "is the next token". Row
structure changed from free to structurally derived to capped in two independent units. **Own-word mass never leaves 0.0008-0.0019.**

**The conclusion, stated as narrowly as nine arms allow**: in this architecture at 16M parameters, a concept's answer row is a prediction device.
It becomes meaningful only if training rewards it for being meaningful, and next-token prediction does not - because at any position the model
wants mass on whatever token is likely, which is not the concept's own word. Making the row structurally derived, dominant, small, large, or
supervised on a sharper label changes what the row does; none of them changes what the row contains.

**What this does not say**: that readability failed. `explain()` is exact to 2.4e-06 with concept share 1.000000 in every arm - the explanation
faithfully reports which concepts drove each decision. What is not yet true is that a concept's NAME predicts the words it promotes.

## The accuracy result stands on one seed and is not yet claimable

**25.97% is one seed.** The shipped 24.58% carries a 3-seed bar with sd 0.055, so seeds 1 and 2 are running now with the prediction that all
three land above the shipped upper edge of 24.64%, and the falsifier that any seed at or below 23.6% means the 1.39-point gain was seed luck.
No accuracy claim will be made public before those land.
