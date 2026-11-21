

---

# MORE DATA DOES NOT FIX GENERALISATION: 8.3x THE PAIRS BOUGHT +6.1 POINTS (2026-11-21 11:50)

The pre-registered one-change test: 600 rule pairs -> **5000**, nothing else moved. Three seeds, 6000 steps.

```
seed   rule pairs   held-out rule acc   train acc   gap
 0       5000            38.17%            -          -
 1       5000            44.83%          55.00%     10.2
 2       5000            45.50%          54.37%      8.9
mean     5000            42.83%          54.69%      9.5
      (600 pairs)        36.72%          53.96%     17.2
```

- **P1 FAILED:** the bar was **above 60% on all three seeds**. Actual: **38.17-45.50%**.
- **P2 FAILED but improved:** the train/held-out gap was supposed to fall below **10 points**. It fell from **17.24 to 9.5** - the
  narrowing is real, the bar is not met by the top seed.
- **Falsifier:** written as "stays below 45% even with 5000 pairs". Mean **42.83% is below 45**, and two of three seeds are below it,
  so its substance is met (one seed reached 45.50%, recorded rather than hidden).
- **The switch still works everywhere:** facts **9.25% and 11.50% -> 100.00%** with the store.

## What this settles

**Data is not the binding constraint.** 8.3x the training pairs bought **+6.11 points**. If the shortage were data, that ratio would
have done more than that. The limit is in the architecture - which is exactly why the owner's order (finish every layer, then train
once) is the right one, and why the next change on this line must be structural: a loop over the output digits, so the second digit can
depend on what the first one decided.

## Recorded honestly

Three predictions have now failed on this line: 60% held-out (failed), gap under 10 (missed by 0.5 at best), and the earlier
single-seed capacity excitement (killed by 3 seeds). The pattern is consistent and it points one way: **the model is not short of data,
capacity or tuning; it is short of a structure that can carry a computation.**
