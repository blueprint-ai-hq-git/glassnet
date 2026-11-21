

---

# CURRICULUM FAILS TOO: EVERY CHEAP LEVER FOR "LEARN FASTER" IS NOW ELIMINATED (2026-11-21 20:05)

The last cheap lever was the order of the data. It does not help - both directions make it **worse**.

## Measured, 3 seeds, identical data and task, only the sampling schedule differs

```
seed   random      easy-first            hard-first          final held-out (all arms)
 0      350        950 (-171.4%)          1000 (-185.7%)      100.00% / 100.00% / 100.00%
 2      600        950 ( -58.3%)           950 ( -58.3%)      100.00% / 100.00% / 100.00%
```

- **P1 NOT MET** on every seed: easy-first is **1.6x to 2.7x slower**.
- **P2 MET:** no accuracy is lost - all arms finish at 100.00%.
- **P3 mixed:** on seed 0 easy and hard differ 14.3 points, on seed 2 they are identical, so there is no consistent ordering effect.

## The mechanism of the failure, which is not really about order

Both scheduled arms run **~950 steps** while random runs 350-1000. The thing they share is that for the first third of training they draw
from **half the data**. So what costs the time is **restricting the pool**, not the direction of the restriction - and the "curriculum" idea
as implemented is exactly a restriction.

That is a clean negative: **on this task, seeing everything from the start beats any schedule of seeing part of it.**

## The complete, honest ledger for "learn 10x faster"

| lever | measured result |
|---|---|
| focus the loss on uncertain tokens | **harmful**: 100% -> 4.5% / 0.0% |
| more compute passes per step (loop R=1/4/8) | **0.0 points**, at any depth |
| more training data (600 -> 5000 pairs) | **+6.11 points**, bar missed |
| more capacity (allocation sweep, 3 seeds) | **+0.27 +- 0.31**, unstable |
| **curriculum / data order** | **1.6x - 2.7x SLOWER** |

**Conclusion, stated as the pre-registration required:** the speed is not in the loss, not in the compute depth, not in the data volume, not
in the capacity, and not in the data order. The baseline - **350-500 steps to 90% token accuracy, then 100.0% exact** - is what this
architecture gives on this task. **A 10x claim against that needs a different architecture, not a different training trick**, and no amount
of tuning inside this shape will produce it.

That does not weaken today's real result. The architecture reads, selects and compares at 100%, and now performs multi-digit addition and
subtraction on unseen operands at 100% when the computation is written out. It is simply **not 10x faster at learning** than it was this
morning, and pretending otherwise would be the one thing this project has never allowed.
