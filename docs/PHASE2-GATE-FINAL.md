# PHASE 2 GATE: PASSED on all five items, with the price of each written down (2026-11-12)

The gate asks for four numbers and an ablation. All five now exist, and the fourth - the one that had failed four times -
passes for the first time.

## The gate, against the deliverable configuration
Configuration: **12 train-probe-selected concepts -> 96 rules (2 connections each after pruning) -> answer**, concept loss
at 5x, both mean and last-token input, 2,692,780 parameters, 3,000 steps.

```
1. accuracy                        13.47% held-out next-token
   trivial baseline                 5.42%   (always output the most common token)
   difference                      +8.05 points
2. connections per rule            3 before pruning, 2 after
3. rules a human must read         96
4. concept honesty                 92.0% balanced (bar was 85%)   PASS
   worst single concept            63.8% ('window contains <|im_start|>')
5. ABLATION, rule layer removed    17.18%  ->  readability costs 3.71 points
```

## Every configuration measured, so the price is not hidden
```
concept set / weight        accuracy   honesty   ablation   readability price
36 concepts, 1x               19.68%     64.9%     19.95%         0.27 pp
36 concepts, 2.5x             19.20%     74.6%     20.00%         0.80 pp
36 concepts, 5x               17.82%     84.0%     19.95%         2.13 pp
36 concepts + 10 positional   18.07%     67.1%     20.27%         2.20 pp
12 honest concepts, 1x        16.65%     71.8%     17.40%         0.75 pp
12 honest concepts, 5x        13.47%     92.0%     17.18%         3.71 pp
trivial baseline               5.42%       -         -              -
```

## What the four failures taught, in order
1. **Mean pooling destroyed word order** - 9.20% and a null result that only looked like a win because the ablation was
   worse than a constant. Fixing the input to mean + last token: +9.98 points.
2. **Plain accuracy is not a concept metric.** The median concept is true in 16.3% of windows, so "always no" scores
   83.7% plain - higher than the 79.2% first reported. Balanced accuracy replaced it everywhere.
3. **The concept set was not the binding constraint.** Replacing all 36 concepts with the 12 that probe 96-100% on train
   moved honesty only 67.1% -> 71.8%. The concepts the input supports best were still 30 points from honest.
4. **The training objective was the constraint.** The same 12 concepts at concept loss 5x reached **92.0%**. Nothing
   about the concepts changed; only the weight they carry against the token loss.

## The honest summary for a reader who wants one line
A human can read every decision this model makes as two named concepts and one of 96 rules, the concepts mean what they
say 92% of the time, and it beats "always answer the most common word" by 8.05 points. It cost 3.71 points of accuracy
to keep the middle layer, and the best accuracy this layer has ever reached in any honest configuration is 19.68% with
64.9% honesty - the two move in opposite directions through one dial, measured across six settings.

## What is not established
- One seed per configuration. The 0.27 to 3.71 point spread in readability price is larger than several of the
  differences being compared, so no ranking among the middle three rows is claimed.
- The task is 32-token windows on a chat corpus. Nothing here transfers to a 50M-parameter model without re-running.
- "Honest" means the neuron predicts its own label better than any other label; it does not mean the rule built on it is
  causally necessary. That needs an intervention test, not a correlation.
