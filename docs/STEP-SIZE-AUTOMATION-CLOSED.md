

---

# STEP-SIZE AUTOMATION IS ABANDONED: IT TRAINS, BUT GIVES UP 3.6-6.6 POINTS TO A HAND-SET CONSTANT (2026-11-20 18:44)

The gain rule was the mechanism the evidence demanded, and it works - it is the only variant besides plain momentum that trains at
all. It is also clearly worse than a constant someone tuned. Both facts are true and both are reported.

## The complete optimiser ordering, all measured at emb_rank 32, seed 0, 600 steps

```
optimiser                  setting      held-out acc
momentum (hand-set)        lr 0.05        12.51%     <- shipping choice
gain (loss response)       start 0.50      8.90%     beat the constant, 3.61 below reference
gain (loss response)       start 0.005     5.89%     beat the constant, 6.62 below reference
adaptive per-element       lr 0.50         0.26%
adaptive per-tensor        lr 0.50         0.03%
adaptive per-element       lr 0.05         0.00%
momentum (hand-set)        lr 0.50       DIVERGED
```

- **P1 FAILED:** gain from 0.50 landed **3.61 points** below the 12.51% reference.
- **P2 FAILED:** gain from 0.005 landed **6.62 points** below.
- **FALSIFIER FIRED** on both arms, so by the rule written before the run: **step-size automation is abandoned for this model**, the
  hand-set learning rate is documented as the shipping choice, and the effort moves to capacity allocation.

## Why the gain rule loses, and it is a real lesson

It decides by asking one question per trial: *did the loss on THIS batch fall?* That is a **greedy, myopic** test. A step that lowers
this batch's loss can raise the loss on the data as a whole - the usual overfitting-to-the-batch failure, applied to the step size
instead of the weights. Its adaptive size climbed to **1.11-1.58** and stayed there, which is not a scale a hand-tuned run would pick.

So the ordering is informative rather than a wash:

**measuring the effect of a step on the current batch is more useful than rescaling the gradient** (it trains; the normalisation family
cannot train at all) **but less useful than a constant chosen against the real objective**. Automation bought robustness at the cost of
accuracy, and the accuracy cost is too large to ship.

## What this closes and what it does not touch

**Closed:** gradient-magnitude normalisation (both granularities, broken at every lr) and loss-response step control (trains, 3.6-6.6
points short). Step size stays hand-set at 0.05, recorded as a known limitation rather than hidden as an automation.

**Untouched, and next:** capacity allocation, and the memorise-vs-generalise switch. Neither has been attempted yet, and neither is
affected by any of this - they are different mechanisms with different jobs.
