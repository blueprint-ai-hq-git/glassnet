# The radius dial trades memory for accuracy monotonically - and the 10% memory gate sits past the collapse (2026-11-18)

Four arms, identical recipe, same seed, same data, 1500 steps each, **one number changed**: `TARGET_RADIUS`. Every row is a finished checkpoint scored by
`tools/eval_radius_arm.py` against the same reference.

```
 radius   k=16 memory   held-out acc   free-running   repetition   gnorm sd
   0.90      0.878%        16.27%         5.00%         0.915       0.81
   1.00      4.730%        13.23%         5.81%         0.908       1.25
   1.05      6.695%        11.90%         5.50%         0.893       1.38
   1.10      8.511%        10.05%         5.06%         0.881       1.43
```

## Predictions, scored

- **P1** "k=16 memory keeps rising: 1.05 >= 6%, 1.10 >= 8%" - **HOLDS** (6.695% and 8.511%).
- **P2** "held-out accuracy keeps falling: 1.05 <= 12%, 1.10 <= 11%" - **HOLDS** (11.90% and 10.05%).
- **P3** "at 1.10 the loss curve is less stable than at 1.05" - **HOLDS**, and it holds as a trend rather than a single point: gradient-norm sd rises **0.81 -> 1.25 -> 1.38 -> 1.43**
  as the radius climbs.
- **FALSIFIER** "memory at 1.10 is NOT higher than at 1.05" - **did not fire**; memory kept rising.

## What the ladder establishes

**Memory and accuracy are in direct, monotone opposition on this dial.** Every +0.05 of radius buys about **+1.8 points of k=16 memory** and costs about **1.3 points of held-out
accuracy**. Extrapolating that local slope, the 10% memory gate would need a radius of about **1.14**, where held-out accuracy would be around **9%** - against **16.27%** at radius
0.9. Even that extrapolation is generous, because the gnorm trend shows the optimisation itself getting rougher as the radius climbs.

So the honest conclusion is: **the memory gate is not reachable by tuning the clamp at any accuracy worth having.** Reaching 10% memory means giving up roughly 7 points of accuracy,
which is 44% of the shipped 16.27%.

**And free-running never moves.** 5.00 / 5.81 / 5.50 / 5.06% - every paired comparison against the reference has a bootstrap interval spanning zero. Four radii and the model still
does not generate. Memory and generation are not the same problem, and the memory dial does not touch the second one.

## Why this is the end of the clamp line, not another iteration

The clamp was worth testing because it was a confirmed cause: unclamping from 0.9 to 1.0 multiplied long-range memory **5.4x**. That is real and stands. But a dial with a monotone
trade has no setting that solves the problem - it only chooses which failure to have. The structural answer, keeping memory and accuracy separable, is a **separate linear state
channel**, and the orthogonal-recurrence arm (`lens/scripts/run_ortho.sh`) is now running as the first attempt at getting both from the same number of parameters.
