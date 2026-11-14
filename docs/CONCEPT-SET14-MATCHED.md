# Every concept neuron now means its name: 14 de-duplicated concepts, 0 mislabeled in all three seeds, +1.67 accuracy at equal parameters

Date: 2026-11-14. Tools: `tools/select_concept_set.py --max-concepts 14` (selection), `gates/phase2_train.py --concept-names`
(training), `tools/phase2_reaudit.py` with `GLASSNET_CONCEPT_NAMES` (honest audit). Predictions written before the run in
`lens/runs/CONCEPT-SET14-MATCHED-PREDICTIONS.txt`. Architecture unchanged from the shipped arm: R=32, keep=3, alpha=3.0,
lam=0.005, 10 epochs, batch 128, lr 0.1, 3 seeds, Fashion-MNIST.

## Why this run existed: to kill my own excuse

`docs/history/CONCEPT-SET18.md` reported that 18 de-duplicated concepts scored 79.69% against the shipped 14's 78.46%, and stated
openly that the 18-set has ~28% more parameters (**15,068** vs **11,800**), so the accuracy gain might be capacity rather than
label quality. This run removes that confound exactly: the **same** greedy de-duplication rule truncated to 14 features gives a
model with **precisely the shipped parameter count, 11,800**, and only the labels differ.

```
14 de-duplicated concepts: pairwise |corr| median 0.071, max 0.587
shipped 14 for comparison:  8 pairs above 0.80, max 0.916
```

## Result: label quality, not capacity - my caveat was wrong

```
                              shipped 14           14 de-duplicated        change
parameters                     11,800                  11,800             identical
accuracy (3 seeds)         78.46% +/- 4.63         80.13% +/- 0.81        +1.67
mean |rank corr|, own label     0.732           0.846 / 0.847 / 0.848     +0.114
concepts below 0.50             2 of 14               0 of 14             none left
concepts tracking ANOTHER       2 of 14               0 of 14             none left
dead rules                      0 of 32            3 / 0 / 0 of 32        ~unchanged
concept interaction              0.0%                 0.0%                unchanged
```

**In all three seeds, every one of the 14 concept neurons tracks its own name better than any other feature, and none sits below
0.50 own-label correlation.** That is the first configuration in this project for which the claim "each middle neuron means what
its label says" holds without exception - and it holds at identical parameter count, so it cannot be a capacity artefact.

**The 14-set also beats the 18-set** (80.13% vs 79.69%) with 3,268 fewer parameters. Fewer, cleaner concepts beat more concepts
once the redundancy is gone.

## Predictions, scored against myself

- **P1 the honesty gain is not a parameter effect - HOLDS**: 0.846+ mean own-label correlation, **0 of 14** tracking another
  feature (I allowed up to 1).
- **P2 the accuracy gain is mostly capacity, landing within +/- 1.0 of 78.46 - FAILED** at **80.13**, which is +1.67.
- **P3 dead rules stay at 0 of 32 - HOLDS in 2 of 3 seeds**, with 3 dead rules in seed 0 (a small regression worth watching).
- **FALSIFIER TRIGGERED:** I wrote that if the set scored 79.96% or better, "the accuracy gain was label quality, not capacity, my
  caveat was wrong". It scored **80.13%**. **My capacity caveat was wrong**, and it is withdrawn here.

## Where the readable image model now stands against a matched plain net

```
matched plain control (10,345 params)                84.37% +/- 1.04
readable, 14 clean concepts, rules (11,800 params)   80.13% +/- 0.81     -4.24
```

So the full readable stack - 14 named concepts, 32 rules reading 3 concepts each, trained with L1 so the sparse wiring survives
pruning - costs **4.24 points** against a plain network of the same size, and in exchange every concept is honest, the concept
layer is purely additive (interaction 0.0%), and a rule can be read by a human as "3 concepts, in, one weight out".

## What is still open

- **Rule plausibility**: median rule-pair lift was 0.92 on the 18-set and 19 of 32 rules pair concepts that rarely co-occur. Not
  yet re-measured on this 14-set, and it is the remaining honesty problem: honest concepts do not guarantee sane rules.
- **The alpha setting** was chosen when the labels were redundant. With clean labels, a lower alpha may now hold honesty and
  return some of the 4.24 points. That is the next sweep, and it is cheap.
- **Per-concept honesty at the label level**: 0 of 14 mislabeled is measured against **these** labels. If a label's own definition
  is a poor description of the pixels, every downstream number inherits that. The definitions are documented and each is a simple
  pixel statistic, but "honest about my label" is not the same as "my label is the most useful idea".
