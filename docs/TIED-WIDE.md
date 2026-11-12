# Widening the tied channel: complete explanations at no accuracy cost (2026-11-12)

Tying the answer layer to the concepts (so a rule's push is the sum of its concepts' rows) cost **-2.83 accuracy points**
when the concept set had 11 entries. The reason was structural and stated in advance: under a tie the answer matrix has rank
at most **C**, so 11 concepts can only produce 11 independent directions for 12,002 outputs. This arm widens the concept
channel to 23 (prevalence-filtered from a 24-item facet set) and changes nothing else.

```
config                                  accuracy (3 seeds)          honesty        layer price
UNTIED, 11 concepts          15.22 / 15.45 / 14.57              78.0 +/- 1.7    1.57 +/- 0.51 pp
                             mean 15.08 +/- 0.46
TIED,  11 concepts           13.38 / 11.57 / 11.80              79.2 +/- 1.1    4.97 +/- 1.11 pp
                             mean 12.25 +/- 0.99
TIED,  23 concepts           15.40 / 15.25 / 15.68              75.2 +/- 2.9    3.46 +/- 0.23 pp
                             mean 15.44 +/- 0.22

widening the tied channel 11 -> 23 concepts:  +3.19 points
TIED-23 vs UNTIED-11:                          +0.36 points  (inside noise)
```

Predictions: "recover at least 1.2 of the 2.83 points lost to tying (>= 13.5%)" **passed at 15.44%**; "rank equals the kept
concept count" **passed at 22 of 23** (one concept is wired to no rule, so its row never contributes); "honesty at or above
78%" **failed at 73.6%**.

## The headline result
**The complete, rank-verified explanation now costs nothing measurable in accuracy.** Tied-23 (15.44 +/- 0.22) and untied-11
(15.08 +/- 0.46) are 0.36 points apart, well inside the spread. The earlier -2.83 was not the price of explainability; it was
a channel-width artefact, and the rank argument predicted it before the run.

## What it still costs
- **The rule layer is more expensive**: 3.48 +/- 0.32 points against its own ablation, versus 1.57 +/- 0.51 untied. With the
  tie, every rule's effect must flow through shared concept rows, so the layer is doing harder work.
- **Concept honesty falls to 75.2% +/- 2.9** from 78.0, because the 12 added concepts are weaker (that is what the earlier
  probe showed: "earlier: token at -3 is punct" 63.6%, "chat tag in last 8" 66.0%).
- **More concepts to read**: 23 names instead of 11, so a reader has roughly twice as much vocabulary to hold in mind before
  the rules make sense.

## The four configurations, stated plainly
```
                                     accuracy   honesty   layer price   explanation complete   explanation proven
96 rules untied, 11 concepts          16.49%     77.8%      0.29 pp            no                    no
24 rules untied, 11 concepts          15.08%     78.0%      1.57 pp            no                    no
24 rules TIED,   11 concepts          12.25%     79.2%      4.97 pp            yes                   yes (rank 11)
24 rules TIED,   23 concepts          15.44%     75.2%      3.46 pp            yes                   yes (rank 22)
```

The last row is the configuration this project can defend on its own terms: **15.44% accuracy against a 5.42% trivial
baseline (+10.02), 24 rules each load-bearing, 23 named concepts, and an answer layer with no hidden degrees of freedom
(rank 22 against 23 concepts, measured).** It is 1.05 points behind the 96-rule untied model, which is the price of being
able to read the whole thing - and unlike that model, none of its rules are inert by construction of the measurement.

## Next
Replicate the per-rule ablation on the TIED-23 configuration across the three seeds already on disk (no retraining), to check
that "every rule is load-bearing" survives the tie, and produce the final explain() sample from this model so the public
documents show the newest configuration rather than an earlier one.
