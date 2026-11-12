# Tied answers: structurally complete, causally NOT faithful (2026-11-12)

The tied configuration closes the explanation hole - a rule's push is exactly the sum of its concepts' rows, proven by rank
(22 of 23 concepts). The obvious next question is whether the parts it names are the parts that matter. Two ablations, run on
the three seeds already on disk, say no.

## Rule-level necessity disappears under the tie
```
config                    median per-rule ablation loss (3 seeds)     what it means
24 rules, UNTIED          0.358 +/- 0.154   (0.487 / 0.188 / 0.400)     every rule is load-bearing
24 rules, TIED-23         0.000 +/- 0.000   (0.000 / 0.000 / 0.000)     every rule is individually dispensable
```

**Under the tie, deleting any single rule changes nothing on all three seeds.** That is not a measurement accident, it follows
from the architecture: every rule's answer row is built from the same shared concept rows, so most rules are alternative
routes to the same output. The layer can still produce the right answer, and the explanation is still structurally exact, but
the specific rule named in an explanation is not the reason for the answer.

The plausibility-vs-necessity correlation is also absent, as it must be when necessity is uniformly zero: -0.053 / +0.158 /
-0.116 (pearson), i.e. no relationship on any seed.

## Concept-level influence exists but does not track honesty
```
mean concept honesty                       73.7% balanced
mean answer flip when a concept is forced ON   29.7%
most causal    content: window has '{'      flip 71.8%   honesty 79.7%
least causal   content: window has <unk>    flip  0.0%   honesty 63.6%
correlation honesty vs causal influence    pearson -0.142
```

Forcing one concept changes the answer on **29.7%** of held-out windows, so concepts are causally live - less than the
untied model's 36.2%, but real. Yet `content: window has <unk>` has **zero** causal influence (it appears in 67% of windows,
so forcing it on changes almost no decisions) while scoring 63.6% honesty, and honesty overall is *negatively* related to
influence (-0.142).

## What this means, stated without hedging
Tying buys **structural completeness**: no hidden degrees of freedom in the answer layer, which is a real and verifiable
property. It does not buy **causal faithfulness**: the rule an explanation names is not necessary for the answer, and a
concept's honesty does not predict whether it moves the answer.

```
property                                   UNTIED-24        TIED-23
accuracy                                   15.08% +/- 0.46  15.44% +/- 0.22
concept honesty                            78.0% +/- 1.7    75.2% +/- 2.9
readability price (vs ablation)            1.57 +/- 0.51    3.46 +/- 0.23  pp
answer layer has no hidden freedom         no               YES (rank 22 of 23)
every rule is necessary                    YES (median 0.36) no (median 0.000)
concept honesty predicts influence         no               no (pearson -0.142)
```

So the two most defensible configurations each hold exactly one of the two properties the brief asks for:

- **UNTIED-24**: every rule matters, but each rule's effect is an arbitrary direction - the explanation can be wrong about
  *why*.
- **TIED-23**: the explanation cannot be wrong about *why* (the push is literally the concepts' rows), but the rule it names
  is not necessary - the explanation can be wrong about *what caused this answer*.

Neither is a fully faithful explanation, and the honest thing to report is that the architecture trades one kind of
faithfulness for the other rather than delivering both.

## What would deliver both
1. **Tie the answers AND break rule redundancy**: penalise similarity between rules' answer rows so the layer cannot build
   alternative routes to the same output. Redundancy is what makes rules dispensable; the tied structure makes redundancy
   cheap.
2. **Explain at the concept level, not the rule level, for the tied model.** Since concepts are the shared trunk, the truthful
   explanation of a tied model is "these concepts were true, and they push these words", with the rule shown as the wiring
   that selected them - not as the cause.
3. **Measure necessity per concept under the tie** (the analogous ablation exists - forcing a concept on flips 29.7% of
   answers) and report a concept-level necessity table beside the honesty table, so a reader sees which named parts are both
   honest and load-bearing.
