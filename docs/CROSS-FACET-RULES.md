# Cross-facet wiring: no impossible rules, plausible heavy rules, accuracy held (2026-11-12)

The facet experiment showed the model picks impossible rules whenever they are on the menu, and a contradiction is
attractive to the objective because it is rare and sharp: it fires almost never and is nearly always right when it does.
So the last arm stops offering them: a rule may only join two concepts from **different facets**. On the facet set that
leaves 52 legal pairs out of 66, so the layer is constrained without being starved (the statistics-based mask earlier had
only 16 pairs and degraded everything).

```
                                     accuracy   honesty   ablation   price     median lift   independent   impossible
facet concepts, random wiring          16.38%     77.1%     18.30%    1.92 pp      0.82        19 of 96         yes
facet concepts, CROSS-FACET wiring     16.45%     76.6%     18.00%    1.55 pp      0.89        15 of 96         none
single-token concepts (for reference)  15.28%     84.0%     17.75%    2.47 pp      0.98        23 of 96         yes
trivial baseline                        5.42%       -          -         -            -             -             -
```

**Predictions: "zero rules joining logically impossible pairs" passed (by construction); "accuracy within 1 point of
16.38%" passed (16.45%); "at most 15 independent-concept rules" passed exactly (15); "median pair lift >= 1.2" failed
(0.89).**

## The three things that changed
1. **No impossible rules exist any more.** The heaviest rules now read as sentences a person could use:
   `shape: last token is 6+ chars AND position: chat tag in last 8` (lift 1.79),
   `shape: last token is a digit AND content: window has '{'` (lift 2.06),
   `content: window has <unk> AND earlier: token at -3 is punctuation` (lift 1.02).
2. **The influence-vs-lift correlation flipped positive for the first time**: -0.104, -0.100, -0.119 across the three
   earlier concept sets, now **+0.030**. It is barely positive, but the model has stopped putting its heaviest weights on
   pairs that never co-occur.
3. **The readability price is the lowest measured: 1.55 points** (18.00% ablation vs 16.45% with rules), while accuracy is
   the highest of any concept set so far.

## What is still not right
- **Median pair lift is 0.89**, i.e. the typical rule still joins two concepts that are slightly *less* likely to be true
  together than chance. Removing contradictions made the top of the layer sensible, not the whole of it.
- Concept honesty fell to 76.6% (from 84.0% in the single-token configuration) - the two readability properties still pull
  against each other, and this arm chose rule plausibility over concept honesty.
- 15 of 96 rules still join concepts in the independence band, so a sixth of the layer remains unreadable in content.
- One seed per configuration; the differences between 16.38% and 16.45% are far inside the run-to-run spread.

## Where this leaves the middle layer
Two of the three semantic goals are now met simultaneously with the best accuracy measured:
**every rule is readable (12 named concepts, 2 connections each), no rule is a contradiction, and the heaviest rules join
concepts that actually co-occur.** The open item is the long tail: 15 rules are still arbitrary, and the median pair is
still weakly anti-correlated.

The next question is no longer "can the layer be made honest" but "does the honest layer change what the model does" -
which needs the causal intervention test re-run on this cross-facet model, plus a check of whether the plausible rules are
the ones carrying the accuracy.
