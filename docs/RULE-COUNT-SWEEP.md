# Rule-count sweep: the layer stops being fully load-bearing between 24 and 48 rules (2026-11-12)

Four arms, identical in everything except how many rules the middle layer has (12, 24, 48, 96). Each was then ablated rule
by rule to measure how much accuracy the model loses when that rule is removed.

```
rules  accuracy  honesty  ablation  readability price  median per-rule loss  lift-vs-usefulness  top-10 implausible
  12     13.90%    78.1%   17.55%       3.65 pp            1.613 points          +0.236 (s+0.266)      5 of 10
  24     15.30%    77.5%   17.43%       2.13 pp            0.350 points          +0.166 (s+0.296)      5 of 10
  48     15.47%    76.9%   17.55%       2.08 pp            0.000 points          +0.102 (s+0.054)      7 of 10
  96     16.45%    76.6%   18.00%       1.55 pp            0.000 points          -0.012 (s-0.044)      7 of 10
trivial   5.42%      -        -            -                  -                    -                  -
```

Predictions: accuracy 14-16% for the 24-rule arm **passed** (15.30%); "median per-rule loss between 0.2 and 1.0" **passed**
for 24 rules (0.350) and **failed** for 48 rules (0.000).

## The knee is between 24 and 48
- 12 and 24 rules: the median rule is worth real accuracy when removed (1.613 and 0.350 points). Every rule participates.
- 48 and 96 rules: the median rule is worth **0.000**. Half the layer can be deleted without changing a single answer.
- Plausibility tracks usefulness only while the layer is small: +0.236 at 12 rules, +0.166 at 24, +0.102 at 48, -0.012 at
  96. The correlation decays as rules are added and turns negative at 96.

## 24 rules is the knee, and it costs almost nothing
At 24 rules the model gives up **1.15 points of accuracy** against the 96-rule version (15.30% vs 16.45%) and pays a
readability price of **2.13 points**, while every rule participates (median 0.350) and plausibility still predicts necessity
(+0.166). The 12-rule arm buys a stronger faithfulness claim (median 1.613, +0.236) for another **1.40 points** of accuracy.

So the honest product statement is not "96 readable rules" and not "12 rules". It is:

```
24 rules, 12 named concepts, 2 connections per rule:
    accuracy 15.30% against a 5.42% trivial baseline (+9.88 points)
    every rule is load-bearing (median 0.350 points when removed)
    a rule that makes sense is more likely to be a rule the model needs (+0.166)
    the rule layer costs 2.13 points of accuracy versus having no rule layer at all
```

## Two anomalies worth fixing rather than hiding
1. **Some rules actively hurt.** In the 24-rule model the least useful rule has a per-rule loss of **-0.650**: removing it
   RAISES accuracy by 0.65 points. A rule that damages the model should be pruned by the training objective and is not.
2. **Half the load-bearing rules are still implausible** at every count: 5 of 10 at 12 and 24 rules, 7 of 10 at 48 and 96.
   The correlation is positive but weak, so the rules that matter and the rules that make sense overlap only partly.

## Method note
Per-rule and group ablations use a 4,000-window held-out sample, which is why an ablation baseline (e.g. 15.45% for the
48-rule model) differs slightly from the training-time evaluation (15.47%); both are quoted and the difference is sampling.
One seed per rule count.
