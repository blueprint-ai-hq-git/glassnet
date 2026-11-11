# Concept honesty vs accuracy: one dial, measured at three settings (2026-11-11)

The Phase 2 gate wants concepts that mean what their names say AND a model that predicts. Both are measurable on the same
recipe, so the weight on the concept-label loss was turned from 1x to 2.5x to 5x and nothing else changed.

```
concept loss weight    token accuracy   concept honesty (balanced)   cost per point of honesty
  1x                      19.68%              64.9%                      -
  2.5x                    19.20%              74.6%                     0.05 pp accuracy per pp honesty
  5x                      17.82%              84.0%                     0.10 pp accuracy per pp honesty
  trivial baseline         5.42%                -                         -
  ablation (no rules)     19.95-20.00%          -                         -
```

**Prediction on record for the 5x run: "honesty 75% or better, token accuracy within 1 point of 19.68%". Honesty PASSED
at 84.0%. Accuracy FAILED - 17.82%, which is 1.86 points down.**

## What this establishes
- **The concepts were not dead, they were under-weighted.** 64.9% -> 84.0% balanced honesty from the loss weight alone,
  with the same architecture, same data, same optimiser. Five dead neurons at ~50% became five weak-but-alive ones at
  52.8-62.7%.
- **Honesty and accuracy trade through that one dial.** Paying 5x costs 1.86 points of prediction. That is the first
  measured price for label-honest concepts on this layer, and it is cheap compared with the 14.26 points the whole model
  gains over the trivial baseline.
- **The readability price is not fixed either.** Against its own ablation at the 5x setting the rule layer costs 2.13
  points (17.82 vs 19.95), where at 1x it cost 0.75 - so as the concepts get more honest, the middle layer also gets
  more expensive to keep.

## Honest caveats
- 84.0% balanced honesty still misses the 85% bar, and one concept remains effectively dead at 52.8% ('as').
- The concepts are still "this window contains X". A bag-of-words concept can reach 84% honesty without being a useful
  *reason* for a next-token decision, and nothing in this run shows they are causally useful - only that they are
  predictable from the input and that the answer layer can read them.
- Every number here is one seed. The 1x and 5x runs differ by 1.86 points on a single seed each, so the dose-response
  curve needs a second seed before the slope is trusted.

## The dose-response, completed
The 2.5x dose landed at **19.20% accuracy with 74.6% balanced honesty**. Prediction was "honesty 75-80%, accuracy within 1
point of 19.68%": honesty **missed by 0.4 points**, accuracy **passed** at -0.48. So the curve is:

```
2.5x costs 0.05 points of accuracy per point of honesty  (0.48 pp for 9.7 pp)
5x   costs 0.10 points of accuracy per point of honesty  (1.86 pp for 19.1 pp)
```

**2.5x is the cheap end of this curve**, roughly half the price per point of honesty compared with 5x. Honesty still
misses the 85% bar at every setting, which says the limit is the CONCEPT SET, not the weight: every concept is a
bag-of-words property, and the cheapest way to buy honesty from those is to over-weight them until prediction suffers.

## Next: richer concepts, at the default weight
Half the concept set is being replaced with positional and structural properties that a next-token rule can actually use
as a reason - "the last token is a question mark", "the last token is a digit", "the window has 3+ punctuation tokens",
"the last two tokens are identical", "the window contains <|im_start|>". Prediction: **balanced honesty at or above 85% at
the default 1x weight**, with token accuracy within 1 point of 19.68%. If positional concepts reach 85% at 1x, the honest
concept set costs nothing at all.
