
## Reconciled: the top-1 discrepancy was one chunk leaking out of a loop (2026-11-17)

Last entry flagged that two of my scripts disagreed on top-1 confidence at identical accuracy: **0.1951** (`state_ablation.py`) against **0.1256** (`calibration_final.py`). The cause is found and fixed.

**The bug:** in the ablation script, `pr` was computed inside the per-chunk loop but read *after* it, so `pr.max(1).mean()` averaged **one chunk of 128 tokens** instead of all 5,120. Accuracy and entropy were accumulated correctly inside the loop and therefore agreed exactly; only the top-1 column was a leaked single chunk.

After accumulating top-1 across all chunks:

```
arm            acc%     entropy    sum(pr)       top1
full         16.70%      6.6505   1.000000     0.1256   <- now matches calibration_final.py exactly
zero          0.02%      9.3894   1.000000     0.0001
shuffled      5.66%      6.7721   1.000000     0.1056

cost of zeroing the state   : 16.68 points
cost of destroying the order: 11.04 points
```

**The two scripts now agree to four decimals (0.1256), and the load-bearing conclusions are unchanged**: zeroing the state costs 16.68 points, destroying token order costs 11.04 points. Only the shuffled arm's confidence moves, 0.1317 -> 0.1056.

## Why this keeps happening, stated as a rule rather than a regret

Five instrument defects in this thread, four of them mine, and they are all one family:

1. **A number computed on a subset while presented as the whole** - top-1 from one chunk (this one), and earlier the 40-window reading quoted as the arm's result.
2. **A missing guard on the quantity being measured** - the missing `<unk>` mask, the broken entropy expression.
3. **A proxy standing in for the thing itself** - F4's KL across six prompts measuring variation instead of competence.

The rule that would have caught every one: **every figure prints what it was computed over** - the token count, the chunk count, the forward path - beside the value. Not just the sum of probabilities, which I added earlier; the *denominator* too.

`sum(pr) = 1.000000` was true and passed in all three arms while the top-1 column was still wrong, because a single chunk is a perfectly valid distribution. **The guard has to name its sample.**
