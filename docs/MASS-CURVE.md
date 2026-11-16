
## A 15x price change moves own-word mass by nothing: the row-content penalty is inert under training, at every weight tested (2026-11-16)

Predictions `lens/runs/MASS-CURVE-PREDICTIONS.txt` and decision rule `lens/runs/MASS-CURVE-DECISION-RULE.txt` were both written before the third
weight finished. Numbers from `lens/runs/MASS-CURVE.txt`.

```
weight   VAL accuracy   VAL ppl    own-word mass   concepts >= 0.05
 2.0       23.16%       222.07        0.0007            0 of 53
 8.0       23.12%       229.00        0.0007            0 of 53
30.0       23.07%       236.79        0.0007            0 of 53
unweighted baseline                    0.0008
```

## Every pre-registered prediction failed except the one bounding it

- **C1** (every weight at or above the 0.0008 baseline): **FAILED at all three.** Mass is 0.0007 at every weight - *below* the baseline, not
  equal to it.
- **C2** (mass rises with weight): **FAILED.** Fifteen-fold increase in price, zero change in mass.
- **C3** (accuracy falls with weight): technically yes, 23.16 -> 23.12 -> 23.07, which is **0.09 points across a 15x price change** - flat, so the
  price is not biting on anything.
- **C4** (mass stays below 0.05): **PASSED**, trivially and uninformatively.
- **FALSIFIER** ("if mass does not exceed baseline at ANY weight, the anti-correlation is mechanical rather than a price effect and this line
  closes"): **FIRED IN FULL.**

## The conclusion, which is sharper than "the term is too weak"

**On a frozen model this exact penalty raises own-word mass.** One gradient step moved it **0.1207 → 0.1237**, measured through the term's gradient,
verified against finite differences at **3.7e-10**. The term works on the model it is applied to.

**But training under it for six epochs returns the model to baseline** - 0.0007 at weight 2.0, and still 0.0007 at weight 30.0.

So the objective is not *resisting* the penalty; **the training process erases its effect**. A term can measurably move the quantity on a fixed
model and still produce no such model when you train with it. That distinction is the real result here, and it also explains why the earlier
frozen-model cost curve - which priced this penalty against held-out accuracy on fixed checkpoints - pointed the wrong way. Pricing a term on a
frozen model does not tell you what the term will do when trained with.

## What this closes, and what it does not

**Closed**: the row-content penalty as a route to concept-row meaning. It is inert under training at 2.0, 8.0 and 30.0, and fifteen-fold more weight
buys nothing.

**Still open, unchanged**: `explain()` remains exact (residual 2.4e-06, concept share 1.000000), so the system truthfully reports which concepts
drove each decision. The unresolved claim remains the stronger one - that a concept's NAME predicts the words it promotes. Ten mechanisms,
correction shares from 0.0000 to 0.9999, two supervision targets, and a 15x price sweep all leave own-word mass between 0.0007 and 0.0019.
