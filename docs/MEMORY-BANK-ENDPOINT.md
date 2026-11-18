# Memory bank, endpoint verdict: the symmetry fix is real, durable, small, and not free (2026-11-18)

The 8-slot bank was proven to be **one slot with eight times the write rate** (all decays bit-identical). Per-slot decay init (`MEM_SYM=1`) breaks that. This is the **endpoint** answer.

```
                        symmetric bank     symmetry-broken bank
                          (step 3500)          (step 4500)
k=16 memory                 1.605%               2.214%        +0.609 pts, +38% relative
k=32 memory                 0.093%               0.112%
held-out teacher-forced     21.03%               20.37%        -0.66 pts
loss                       6.3199               6.2993
slot decay spread          0.000000             0.068527       (8 of 8 distinct)
```

## Predictions, scored at the endpoint

- **P1** "max-min decay spread > 0.005" - **MET**: **0.068527**, 8 of 8 values distinct, and the spread kept *growing* across checkpoints (0.0500 -> 0.0646 -> 0.0685), so the slots never collapse back.
- **P2** "k=16 memory > 1.605%" - **MET**: **2.214%**, +0.609 points, **+38% relative**.
- **P3** "no accuracy cost" - **NOT met in direction**: held-out **20.37% vs 21.03%**, a **-0.66-point** change. Against the 0.60-point same-step spread measured on this hardware it is about 1 SE, i.e. not resolvable, but it is on the wrong side and must be stated as such.
- **FALSIFIER** "decays still identical, or memory <= 1.605%" - **did not fire**.

## The trend across four checkpoints, which is the real finding

```
step 3250   1.512%   <- the broken bank was WORSE here
step 3750   2.254%
step 4000   2.164%
step 4500   2.214%   endpoint
```

The gain is **delayed and then flat**. Between 3750 and 4500 the bank moved 2.254 -> 2.164 -> 2.214: a plateau. A single early checkpoint at 3250 would have concluded the fix did nothing, which is why this arm was worth running to its endpoint.

## What the line now knows

Three fixes have been made to this mechanism and every one paid, in descending size:

```
radius-only recurrent state        1.312%
+ symmetric 8-slot bank            1.605%   (+0.293)
+ broken symmetry                  2.214%   (+0.609)
gate                              10.000%   still 4.5x away
```

**The gate is not reachable by fixing the bank.** Each defect was real, each fix paid between +0.3 and +0.6 points, and the required move is +7.8. That is the pre-registered falsifier's branch, now measured rather than predicted: **the limiter is the training signal, not the memory mechanism.** A 32-token window never asks the model to retain anything for 16 steps, so the slots have almost nothing to learn *for*, and a better bank mostly stores better noise.

The next attempt on this line should therefore change what the model is **asked** to do (an objective with genuine cross-step demand) before changing the bank again.
