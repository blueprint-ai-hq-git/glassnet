# The 1.5 points are inside the concept layer: width, unit cost and the linear answer layer are all ruled out (2026-11-18)

Three matched comparisons and one frozen-layer probe now bound where the readable model's deficit lives. Every one of them was a real test with its falsifier written first.

## The replication

```
13 concepts, d=64                    -1.53   lens/runs/MATCHED-GAP-3SEED.txt (3 seeds)
21 concepts, d=64                    -1.47   lens/runs/WIDTH-MATCH-21.txt
21 concepts, d=49 (unit-cost parity) -1.46   lens/runs/UNIT-COST-PARITY.txt   FALSIFIER FIRED
```

Raising the concept count from 13 to 21 changed the gap by 0.06 points. Equalising the per-unit parameter cost (d=49 makes 16x49 = 784 features, the same as a plain hidden unit's
784 pixels) moved it by 0.01. **Neither width nor per-unit cost is the explanation.**

## The answer layer is not the explanation either

`tools/rule_answer_probe.py` freezes the trained concept layer and gives the same frozen concepts three answer heads, so the comparison is exactly about the head:

```
answer head (frozen concepts)          test acc   params
linear (what ships)                     83.23%      140
rule-pair (40 chosen pairs)             83.49%      540
rule-full (ALL 78 pairs)                84.18%      920
PLAIN CONTROL, seed 0:                  86.42%
```

Pair rules are "concept A **and** concept B" - still a sentence a person can read, so this was the fix that would have cost no transparency. It gained **+0.26** points with 40 rules and
**+0.95** with all 78, and still sits 2.2 points under the plain control. Predictions P2 (>= 0.5 point gain from pair rules) and P3 (reach 86.42%) both failed. Freezing the
concept layer also costs ~0.9 points by itself (83.23 frozen vs 84.15 co-adapted), so the head is not where the deficit lives.

## What this leaves, stated plainly

The constraint that costs is the **concept representation**: a layer whose units must each match one named formula is a weaker encoder than a free hidden layer of the same size, and
no answer head repairs it afterwards. That is a real, measured property of the readable architecture - and it is the number the project should report: **readability costs about 1.5
points at matched parameters, and buys 100% of the middle layer being human-checkable** (21 of 21 concepts honest in the parity arm; zeroing the concept layer drops accuracy from
84.15% to 10.17% and shuffling it drops it to 9.55%, so the layer is load-bearing and genuinely read per-sample).

## The last arm this line gets before the number is accepted

Every matched arm so far gave the readable model **fewer** units than its control and then attributed the difference to readability. `lens/scripts/run_capacity_advantage.sh` removes
that: patch 14 x d 56 makes a concept cost 234 params instead of 1034, so at matched parameters the readable model can afford roughly **three times** the control's units (~34 concepts
against ~24 hidden).

- **P1** readable fields >= 30 concept units against the control's ~24, confirmed from the printed shapes
- **P2** the gap moves to within +-0.5 points
- **P3** honesty >= 85%
- **FALSIFIER** with a 25%+ unit advantage the gap is still at or below -1.0. Then capacity is not the explanation either, the constraint itself is the price, and ~1.5 points at
  matched capacity becomes the reported number.
