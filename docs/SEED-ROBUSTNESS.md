# Seed robustness: which of today's differences are real (2026-11-12)

Every result reported today was **one seed**, because the initialisation, ablation-model and batch RNGs were hardcoded
(7 / 11 / 3). Seed control has now been added (`SEED`) and the best configuration was run three times.

## The configuration, three seeds
11 facet concepts (one dropped for zero prevalence), 24 rules, answer-row L1, 3000 steps, identical settings.

```
seed    accuracy   honesty   ablation   readability price
0        15.22%     79.5%     17.12%        1.90 pp
1        15.45%     76.1%     16.43%        0.98 pp
2        14.57%     78.5%     16.40%        1.83 pp

accuracy : mean 15.08%   std 0.46   range 0.88
honesty  : mean 78.0%    std 1.7    range 3.4
price    : mean 1.57 pp  std 0.51   range 0.92
```

**The seed spread on accuracy is 0.88 points.** Almost every fine-grained comparison made today was smaller than that.

## Re-grading today's claims against the measured noise
```
comparison                                    measured    vs 0.88 range   verdict
mean pooling -> last token (word order)        +6.18 pp    6.9x            REAL
last token -> mean+last (both inputs)          +4.30 pp    4.9x            REAL
96 rules -> 12 rules (layer size)              -2.55 pp    2.9x            REAL
12 rules -> 24 rules                           +1.40 pp    1.6x            LIKELY REAL
pair concepts -> facet concepts                +0.98 pp    1.1x            BORDERLINE
AR_L1 (answer-row sparsity)                    +0.48 pp    0.5x            NOT DISTINGUISHABLE
facet concepts -> 11 concepts (drop dead)      -0.56 pp    0.6x            NOT DISTINGUISHABLE
24 rules -> 48 rules                           +0.17 pp    0.2x            NOT DISTINGUISHABLE
facet wiring -> cross-facet wiring             +0.07 pp    0.1x            NOT DISTINGUISHABLE
unordered -> semantic wiring                   +0.02 pp    0.0x            NOT DISTINGUISHABLE
```

So of roughly ten accuracy claims made today, **four survive the noise** (three structural, one likely), and six do not.
The structural effects were always the big ones: word order, using both inputs, and the size of the middle layer.

## What does NOT change
The non-accuracy results were measured on single runs too, so they carry the same limitation - but several are far larger
than their own spread and are independently checkable:

- **Concept honesty**: mean 78.0% (std 1.7) with **no concept at 50.0%** any more, and the impossible label removed.
- **Per-rule ablation**: the 96-rule layer's median rule is worth 0.000 points; the 24-rule layer's is worth ~0.35-0.50. That
  gap is much larger than the spread, and it is a structural property of the layer size, not a tuning accident.
- **No contradictory rules exist** under cross-facet wiring - true by construction, not a measurement.
- **The harmful rule shrank** from -0.550 to -0.125 after the impossible label was dropped. One seed for each, so the
  direction is suggestive and the magnitude is not established.

## The honest headline for this layer
```
11 named concepts -> 24 rules (2 connections each) -> answer
    accuracy 15.08% +/- 0.46 over 3 seeds, against a 5.42% trivial baseline (+9.66)
    concept honesty 78.0% +/- 1.7 balanced; no dead concept
    every rule is load-bearing: median per-rule ablation loss 0.35-0.50 points
    the readable middle layer costs 1.57 +/- 0.51 accuracy points
```

## What this says about method, plainly
Three of today's wrong numbers came from measurement defects (an inflated accuracy metric, a silently unwired mask, an
empty concept label), and now a fourth class: **reporting single-seed differences as effects**. The rule that follows is
the one this project keeps re-learning - state the spread before claiming a difference, and only call something real when
it is several times the noise.
