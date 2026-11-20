

---

# THE GENERALISATION HALF FAILED: 4x TRAINING BOUGHT +11.7 POINTS AND THE RULE IS STILL NOT LEARNED (2026-11-20 21:07)

The pre-registered run that was supposed to show the model CAN apply a rule did not show it.

## Measured, 3 seeds, 6000 steps on the same trap dataset

```
seed   held-out rule acc   train acc   gap      fact acc (with switch)   collisions
 0          35.00%           52.50%    17.5          100.00%              0.00%
 1          34.33%           53.44%    19.1          100.00%              0.00%
 2          40.83%           55.94%    15.1          100.00%              0.00%
mean        36.72%           53.96%    17.2
```

Comparison with 1500 steps (mean **25.00%**): four times the training bought **+11.72 points** of held-out rule accuracy.

- **P1 FAILED:** the bar was **above 60% on all three seeds**; the actual values are **34.33-40.83%**.
- **Falsifier:** the written condition was "held-out rule accuracy stays below 40% even at 6000 steps". Two of three seeds are below
  40% and the **mean (36.72%) is below it**, so its substance is met - though not its strictest reading (seed 2 reached 40.83%). Both
  readings are recorded rather than picking the one that is convenient.
- **The switch still behaves everywhere:** facts **10.00-12.75% -> 100.00%**, collisions **0 of 600** on every seed, rule accuracy
  changed by exactly **+0.00**.

## What this means, without softening it

**The memorize half of the ask is demonstrated. The generalise half is not.** The model reaches **~37%** on held-out two-digit addition
while fitting its own 600 training pairs to **~54%**. A 17-point train/held-out gap is the signature of fitting the sample, not of
learning a rule.

**The diagnosis, which sets the next single change:** the task has **10,000 possible pairs** and the model is given **600**. It is
data-starved, not obviously capacity-starved - and the only way to tell those apart is to raise the data and change nothing else. That is
the next pre-registration, with one change.

## What this does NOT license

Widening or deepening the model to rescue the number. The pre-registration said no rescue by widening without a new registration, and
"add capacity until the number moves" is exactly the move that makes this project's earlier results untrustworthy.
