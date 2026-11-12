# Answer-row sparsity: four of five metrics improve, the harmful rule survives (2026-11-12)

The rule-count sweep left two defects: at 48 and 96 rules the median rule is worth 0.000 accuracy points, and at 24 rules one
rule was worth **-0.650** (removing it RAISED accuracy). The cause is visible in the code: the existing L1 pressure is
applied only to the concept->rule wires, so a rule can keep a full answer row while contributing nothing.

This arm adds pressure to the **rule->answer rows** (`AR_L1=0.02`), everything else identical to the 24-rule arm.

```
metric                                    24 rules   24 rules + answer-row L1
accuracy                                    15.30%          15.78%     (+0.48)
readability price (ablation - with rules)   2.13 pp         1.65 pp    (cheaper)
median per-rule ablation loss               0.350 pts       0.500 pts  (up)
plausibility-vs-necessity (pearson)        +0.166          +0.263      (stronger)
implausible rules among the 10 heaviest     5 of 10         4 of 10
least useful rule (removing it changes acc) -0.650 pts     -0.550 pts  (still harmful)
```

Predictions: accuracy within 0.5 points of 15.30% **passed** (15.78%, higher); median per-rule loss at or above 0.350
**passed** (0.500); correlation strengthened as a side effect; "the harmful rule's per-rule loss moves to >= 0" **failed**
(-0.550).

## What improved
- **Accuracy rose** rather than falling: 15.30% -> 15.78%. Pressure on the answer rows acts as a regulariser.
- **Every rule matters more**: median per-rule loss 0.350 -> 0.500 points.
- **Plausibility now predicts necessity more strongly than in any configuration so far**: pearson +0.263, spearman +0.402
  (the 12-rule arm was +0.236, the 96-rule arm -0.012).
- **The readability price fell to 1.65 points** (from 2.13), which is now close to the 96-rule model's 1.55 while keeping
  every rule load-bearing.
- The implausible share of the heaviest rules dropped from 5 of 10 to 4 of 10.

## What did not
**One rule still damages the model** (per-rule loss -0.550: deleting it gains half a point of accuracy), and the L1 pressure
did not remove it. So a rule can be net-negative and still survive a term designed to shrink unused rules. Two candidate
reasons, both testable: the pressure is proportional to the row's magnitude, so a row that is large *and* harmful pays a
cost smaller than the harm it does; and a harmful rule can be compensating for another rule's error, which makes its marginal
value negative while its removal is not a net improvement on its own.

## Where this leaves the middle layer
The best configuration measured today, and the one now on the public front page:

```
24 rules, 12 named facet concepts, 2 connections per rule, answer-row L1 0.02
    accuracy 15.78%  against a 5.42% trivial baseline (+10.36 points)
    every rule load-bearing: median 0.500 accuracy points when removed
    a rule that makes sense is more likely to be one the model needs: +0.263 (spearman +0.402)
    the readable middle layer costs 1.65 accuracy points
    one rule is still net-harmful, recorded as an open defect
```

## Next
Prune the net-harmful rule explicitly (measured, not by magnitude) and retrain the survivors, to see whether "no harmful
rules" is achievable at this size. After that, the remaining open item on the gate list is the causal re-run on this model and
the plain-English explanation output.
