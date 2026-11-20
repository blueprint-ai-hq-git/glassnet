

---

# CAPACITY ALLOCATION: A 30%-TABLE CONFIG BEATS THE 85%-TABLE DEFAULT WITH 29% FEWER PARAMETERS (2026-11-20 19:12)

The capacity question has been the honest hole in this programme: every earlier result left the output table holding 74-92% of the
parameters, and I recorded each time that this did not move. **This is the first measurement that moves it.**

## All four configs, 300 steps, seed 0, momentum lr 0.05, same data

```
config                     params     table share   held-out acc   vs default
A  default  rank32 d64  L2   454,336      85.0%         10.29%          -
B           rank8  d96  L3   322,384      30.0%         10.74%       +0.45
C           rank16 d96  L2   345,056      56.1%         10.87%       +0.58
D           rank8  d128 L2   364,560      26.6%          5.47%       -4.82
```

- **P1 MET:** the best config (C) beats the hand-chosen default by **+0.58** points, against a +0.5 bar.
- **P2 MET:** config **B** is **29% smaller** than the default (322,384 vs 454,336) and **still beats it** (+0.45) - and its table is
  **30.0%** of the model instead of 85.0%.
- **Falsifier not triggered** (something did beat the default).

## Why B is the interesting row, not C

C is the accuracy winner, but its table is still 56.1% of the model. **B moved capacity out of the table into the blocks and depth** -
table share **85.0% -> 30.0%** - while using **fewer total parameters** and scoring **higher**. That is the first evidence in this
project that the layers, not the lookup table, are where the capacity should sit. It is exactly the direction asked for.

D shows the limit: at 300 steps, width 128 with a rank-8 table collapses to 5.47%. Widening the blocks is not free - it needs more
training to pay off, or it is simply too wide for this budget.

## What may not be claimed yet

**+0.58 and +0.45 are ONE seed at 300 steps.** This project has measured paired-delta sd of **0.35** at 600 steps, so both figures are
inside a single standard deviation. They are a **direction**, not a result. The claim that survives publication needs 3 seeds, paired,
which is the next run - and if that fails, the honest sentence becomes "allocation made no difference at this budget".

## What this is and is not

**Is:** measurement-driven selection of where a fixed parameter budget sits - the split decided by evidence instead of by hand.
**Is not:** growth or pruning *during* training, which remains unbuilt and is the harder version of the same idea.
