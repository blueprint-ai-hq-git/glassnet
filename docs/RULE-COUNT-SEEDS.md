# The rule-count trade, with three seeds on both sides (2026-11-12)

The claim that the middle layer's size trades accuracy for faithfulness was measured on one seed per point. Both sides have
now been run three times on an identical recipe - 11 facet concepts after prevalence selection, cross-facet wiring,
answer-row L1 0.02, 3000 steps - with only the rule count changed.

```
                      accuracy (3 seeds)        honesty        readability price
24 rules      15.22 / 15.45 / 14.57            78.0 +/- 1.7     1.57 +/- 0.51 pp
              mean 15.08 +/- 0.46
96 rules      16.40 / 16.68 / 16.38            77.8 +/- 1.4     0.29 +/- 0.31 pp
              mean 16.49 +/- 0.17

accuracy gap (96 - 24) = +1.41 points, combined spread 0.49, ratio 2.9x  -> REAL
```

**The rule-count cost is real**: 1.41 points, 2.9 times the combined seed spread, so it is not a lucky seed on either side.

## The trade, in the direction the seeds actually show
The usual framing - "a readable layer costs accuracy" - is only half right, and the half it gets wrong is the interesting one:

```
96 rules:  accuracy 16.49%   layer costs 0.29 points   but 86 of 96 rules are inert (median per-rule loss 0.000)
24 rules:  accuracy 15.08%   layer costs 1.57 points   and every rule is load-bearing (median 0.35-0.50)
```

- **At 96 rules the readable layer is nearly free** (0.29 +/- 0.31 points) - because most of it is doing nothing. A cheap
  layer that does not participate is cheap for the wrong reason.
- **At 24 rules the layer costs 1.57 +/- 0.51 points**, and that cost is the price of every rule mattering: remove any
  single rule and the model loses about half an accuracy point.
- **Concept honesty is unaffected by rule count** (78.0 vs 77.8), so the rule count does not buy or cost honesty at all. It
  trades accuracy against faithfulness.

## What the product statement can now honestly say
```
24 rules / 11 named concepts / 2 connections per rule
    accuracy 15.08% +/- 0.46 over 3 seeds (5.42% trivial baseline, +9.66)
    concept honesty 78.0% +/- 1.7 balanced, worst concept above 60%
    every rule is load-bearing (median 0.35-0.50 accuracy points when removed)
    a rule that makes sense is more likely to be one the model needs (+0.184 pearson / +0.433 spearman, one seed)
    the readable middle layer costs 1.57 +/- 0.51 accuracy points
```
and the alternative, for a user who wants the last 1.4 points and does not intend to read the rules:
```
96 rules, same recipe: accuracy 16.49% +/- 0.17, layer costs 0.29 +/- 0.31,
    but the median rule is worth 0.000 - a human would have to read 96 rules to find the ~10 that matter
```

## Still single-seed, and labelled as such
- The per-rule ablation medians (0.000 at 96 rules, 0.35-0.50 at 24 rules) come from one seed per rule count.
- The plausibility-vs-necessity correlation (+0.184 / +0.433 at 24 rules, -0.012 at 96) is one seed each.
- Cross-facet wiring removing all contradictory rules is true by construction, not a measurement, so it needs no seed.

Those are the next things to replicate; they are the remaining support for the faithfulness claim, and faithfulness is the
property that makes this layer worth having at all.
