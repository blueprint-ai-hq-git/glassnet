# Radius 1.0 at 3000 steps: accuracy recovered, and the free-running gain is RESOLVABLE (2026-11-18)

The 1500-step radius ladder showed a monotone memory/accuracy trade. This arm doubles the length at radius 1.0, and two of the three headline numbers move.

```
                              radius 1.0 @1500   radius 1.0 @3000   radius 0.9 @1500 (reference)
held-out teacher-forced            13.23%           16.80%               16.27%
free-running                        5.81%            8.44%                5.00%
repetition                          0.908           0.923                0.915
k=16 memory                         4.730%           4.434%               0.878%
Wh spectral radius                  1.0000           1.0000               0.9000
paired free-running vs reference   NOT RESOLVABLE   RESOLVABLE          -
  mean +0.325 tokens / CI [-1.38,+3.00]   mean +1.375 tokens / CI [+1.75,+5.44]
```

## What changed, and what it means

**1. The accuracy cost of unclamping mostly disappears with training.** 13.23% -> **16.80%** at 3000 steps, which is level with the 1500-step radius-0.9 reference (16.27%). The
3.04-point penalty measured at 1500 steps was largely **under-training**, not a permanent price of radius 1.0. This was prediction P2 of the 3000-step experiment ("the gap narrows
below 1.5 points"), and it holds on this comparison.

**2. The free-running gain becomes RESOLVABLE for the first time in this line.** 8.44% against 5.00%, paired difference mean **+1.375 tokens per 40**, bootstrap 95% CI
**[+1.75, +5.44] points** - the interval excludes zero. Every earlier free-running comparison in this project had an interval spanning zero. This is the first one that does not.

**3. Memory does not improve with training length:** 4.434% at 3000 steps against 4.730% at 1500 - slightly *lower*. The k=16 gate still fails, as the ladder predicted it would at
this radius. Longer training buys accuracy and generation, not memory.

## The caveat that must travel with number 2

**This comparison is not matched.** The 3000-step arm is being compared against a **1500-step** reference, so the resolvable free-running gain could be a product of **twice the
training** rather than of the radius. The matched control - radius 0.9 at 3000 steps - is at step 2700 of 3000 and will answer that within minutes. Until it does, the honest claim is
precisely this: **a 3000-step radius-1.0 model beats the 1500-step reference on free-running by a resolvable margin, and the radius's own contribution is not yet separated from the
training length.**

The same applies to the accuracy-recovery claim: "16.80% vs 16.27%" compares 3000 steps of training against 1500, so it establishes that the deficit is not permanent, not that radius
1.0 is equal to radius 0.9 at equal training. Only `scan11_r090_ep3000.npz` settles either question.
