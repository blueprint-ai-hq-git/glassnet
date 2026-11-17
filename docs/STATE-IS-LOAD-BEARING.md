
## CORRECTION: the recurrence IS load-bearing. Ablation kills my "context-free" reading (2026-11-17)

The previous entry concluded from a flat accuracy across context lengths that the model was "effectively context-free at prediction time". **The ablation test written in that same entry
refutes it.** `tools/state_ablation.py`, same 5,120 held-out tokens, same path, same UNK masking, `sum(probs)` = 1.000000 in all three arms:

```
arm            acc%     entropy    sum(pr)       top1
full         16.70%      6.6505   1.000000     0.1951
zero          0.02%      9.3894   1.000000     0.0001
shuffled      5.66%      6.7721   1.000000     0.1317

cost of zeroing the state   : 16.68 points
cost of destroying the order: 11.04 points
```

## What the bar says

The bar fixed before the run: *if zeroing the state costs less than 1 point of accuracy, the recurrence is decorative.* It costs **16.68 points**, and accuracy collapses to **0.02%** - chance is 0.008%, so a state-free model predicts essentially nothing.

**And destroying the order of the same tokens costs 11.04 points** (16.70% -> 5.66%). So the state is not reading a bag: order matters to it, by a wide margin.

**The recurrence is load-bearing, order-sensitive, and doing most of the work.** My "context-free" reading was wrong, and it was wrong in the specific way this project keeps catching:
I measured *additional* context past 16 tokens, found no gain, and generalised to "uses no context". Those are different claims, and the ablation separates them.

## The corrected statement

- Context beyond ~**16 tokens** adds **0.00 points** (15.62% flat from 16 to 128).
- Removing the carried state costs **16.68 points**.
- Destroying token order costs **11.04 points**.

Read together: the model has a **short but real effective horizon** - it extracts a great deal from the first handful of tokens and then saturates. That is a different defect from "the recurrence is decorative", and it has a different fix: the state needs a reason to keep integrating information, not to start doing so.

It also means the loops and the failed rollout arm are **not** explained by an inert state. The state is active; what it apparently does not do is keep improving with more input.

## One inconsistency I am flagging rather than quoting past

Top-1 confidence is **0.1951** here but **0.1256** in `tools/calibration_final.py` at the *same* 16.70% accuracy on the same checkpoint. Two of my scripts disagree on that quantity by 0.07 while agreeing exactly on accuracy. The ablation's internal comparisons are unaffected, because all three arms share one code path - but **absolute top-1 confidence is not quotable until those two scripts are reconciled**, and that reconciliation comes before it appears on any page again.
