# Tied answers: a complete explanation, proven by rank, at a measured price (2026-11-12)

Every explanation produced so far had a hole in it. A rule's *firing* was checkable - it is wired to two named concepts - but
its *effect* was not: the rule owned a free 12,002-dimension answer vector, so "rule #17 pushes 'you' by +1.85" was an
arbitrary learned direction with no relation to the rule's two concepts. A reader could verify WHEN the rule fires and never
WHY that implies a word.

Fixing that is an architectural change, not a measurement: make the rule->answer matrix a function of the concepts.

```
untied:  Ar is (R, V)          each rule owns its own answer direction     R*V parameters
TIED:    Ao is (C, V)          a rule's answer row = SUM of its wired concepts' rows
         Ar_eff = mask^T @ Ao  so C*V parameters, and rank(Ar_eff) <= C by construction
```

## The tie is real, and that is checkable without trusting the training code
`tools/tied_check.py` measures the numerical rank of the saved answer matrix:

```
model      answer matrix     rank    concepts    rules
UNTIED     24 x 12002         24        11         24
TIED       24 x 12002         11        11         24
```

**Rank 11 against 11 concepts, where the untied model is full rank 24.** Every rule's answer row lies inside the span of the
concept rows, so "this rule pushes X because concept A pushes X" is now a fact about the weights, not a story.

## What it costs, three seeds
```
config                     accuracy (3 seeds)          honesty        readability price
UNTIED 24 rules   15.22 / 15.45 / 14.57             78.0 +/- 1.7     1.57 +/- 0.51 pp
                  mean 15.08 +/- 0.46
TIED 24 rules     13.38 / 11.57 / 11.80             79.2 +/- 1.1     4.97 +/- 1.11 pp
                  mean 12.25 +/- 0.99

cost of the tie:  -2.83 accuracy points (5.2x the combined spread, REAL)
readability price: 1.57 -> 4.97 points, i.e. the readable layer costs three times as much
```

Prediction on record: "accuracy within 0.9 points of 15.08%, and the decomposition checkable". **The decomposition passed;
the accuracy prediction failed by 1.9 points.** Tying costs far more than expected.

## Why it costs so much
With the tie, a concept's output row has to serve every rule that wires to it. A rule cannot learn "punctuation AND a long
token implies a comma here" - it can only add its concepts' existing rows. The layer loses the ability to say *rule-specific*
things, which is exactly where a next-token predictor's accuracy lives at this scale. Concept honesty is unaffected
(79.2 vs 78.0), so the tie buys explanation and sells prediction, cleanly.

## The honest state of the middle layer, three configurations
```
                            accuracy   honesty   layer price   explanation complete   explanation verified
96 rules, untied             16.49%     77.8%     0.29 pp       no (arbitrary push)    no
24 rules, untied             15.08%     78.0%     1.57 pp       no (arbitrary push)    no
24 rules, TIED               12.25%     79.2%     4.97 pp       YES                    YES (rank = C)
trivial baseline              5.42%       -          -            -                      -
```

This is the first configuration where "nothing is hidden" is literally true: a reader can follow a decision from the last
two tokens, through a concept's truth value, into a rule, into the exact words that concept pushes, and the rank check proves
no hidden degrees of freedom remain in the answer layer. It costs 2.83 accuracy points and makes the layer three times as
expensive.

## What would reduce the price
- **More concepts.** Rank is capped at C, so 11 concepts is a very narrow channel for 12,002 outputs. A larger facet set
  (say 24-32 concepts) would widen the rank without allowing rule-specific hidden directions.
- **Partial tying.** Tie the answer row to the concepts but allow one shared low-rank correction per rule, and measure how
  much accuracy returns per unit of hidden freedom reintroduced - which makes "how much opacity is worth how much accuracy"
  a measurable dial instead of a choice.
- **Wider d512** (this harness is the small rule-layer testbed, not the 15.5M flagship).

## Tooling gap recorded
The saved checkpoint stores the *learned* Wr, not the structural mask, so the "rules sharing an identical concept pair have
identical rows" check could not be evaluated exactly (it reported 2.1e+01 max difference on TIED, which is a property of
using learned weights rather than the mask). The rank test is unaffected and is the proof used above. Future runs will save
the mask alongside the weights.
