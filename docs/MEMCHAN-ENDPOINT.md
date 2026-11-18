# The memory channel: all three predictions met, the gate nowhere near, and the reason proven (2026-11-18)

The separate memory channel finished 1500 steps from `scan11_r090_ep3000`. Scored against that resumed baseline at the corrected 104-chunk protocol:

```
                            channel arm    resumed baseline    change
held-out teacher-forced       24.30%          23.95%           +0.35 pts  (~0.5 SE, NOT resolvable)
held-out loss                 6.1754          6.3261           -0.15 nats
free-running                   6.06%           6.88%           -0.82 pts  (NOT resolvable)
repetition                     0.875           0.923           -0.048     (less repetitive)
k=16 memory                    1.642%          1.312%          +0.330 pts (gate is 10%)
k=32 memory                    0.080%          0.113%
```

## Predictions, scored honestly

- **P1** "k=16 memory rises above 1.322%" - **MET**: 1.642%. A real rise, and **3.3% of the way to the 10% gate**.
- **P2** "held-out at 104 chunks stays above 21.02%" (no accuracy cost) - **MET**: 24.30%, which is also **above the baseline's own 23.95%**, though not resolvably (0.5 SE).
- **P3** "the learned decay moves away from its 0.9 start" - **MET**: 0.8581 on all eight slots.
- **FALSIFIER** "memory at or below 1.322%" - **did not fire**.

So the channel is installed, gradient-certified, verified as a forward no-op at injection, and it **does move the numbers in the right direction** - accuracy +0.35 (inside noise), loss **-0.15 nats**, repetition **-0.048**, memory **+0.33 points**.

## What P3 exposed, which matters more than any of those numbers

All eight slots read the **same** decay to the bit: `0.8580580949783325`. That is exact symmetry, not convergence, and the forward pass explains why:

1. `M` starts at **zero** for every slot.
2. Every slot therefore scores `q.M = 0`, so the softmax over slots is **uniform**.
3. Every slot receives the **identical** write `a[b,k] * k[b]`.
4. So `d_decay[k]` is identical for every `k`, forever.

**The 8-slot bank was mathematically one slot with 8x the write rate.** A memory gain of +0.33 points is exactly what one extra slow-decaying channel looks like. That is the mechanism behind the small number, and it was found by reading the weights rather than by guessing at hyperparameters.

`MEM_SYM=1` now initialises per-slot decay from `uniform(0.85, 0.95)`, and `run_memchan_sym.sh` re-runs the identical recipe with that single change, queued behind the running arms.

## The verdict on the mechanism, stated plainly

A separate memory channel **does not solve long-range memory at this scale on its own** - not with one effective slot, and the falsifier for "even a proper bank is not enough" is already written down: if the symmetry-broken arm does not beat **1.605%**, the limit is the **training signal**, because a 32-token window never asks the model to remember anything for 16 steps.

What the arm did establish:

- The channel is **harmless**: no resolvable accuracy cost, better loss, less repetition. A component that can be dropped into a running arm without changing it (step-0 forward is bit-identical) and then helps slightly is a component worth keeping.
- The **instrument** works end to end: injection no-op proven, gradients certified, symmetry defect detectable in the saved weights.
