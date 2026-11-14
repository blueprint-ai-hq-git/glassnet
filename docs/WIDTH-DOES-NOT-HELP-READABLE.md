# Doubling the width changes almost nothing for the readable chat model

Date: 2026-11-14 19:15. Readable arms only in this document: `nodirect9_d512_s0` (d512) and `nodirect1024_d512_s0` (d1024), both
9 epochs, seed 0, 600k windows of the full corpus, both forced to answer through the concept vector alone. The d1024 **control**
(direct pathway ON) was still training when this was written, so **no accuracy-gap claim is made here** - that waits for the
matched pair. Predictions `lens/runs/NODIRECT-D1024-PREDICTIONS.txt` were written before launch.

## Result

```
                          readable d512   readable d1024   change
final VAL accuracy             19.95%          19.95%        0.00
final VAL ppl                 292.65          291.04        -1.61
held-out best (by position)    21.45%          (in log)        -
next-token, 2000 positions     21.90%          22.45%       +0.55
distinct tokens predicted          44              49          +5
concept-only accuracy          21.60%          21.35%       -0.25
concept agreement with full     77.35%          82.30%       +4.95
concepts firing (>0.5)       15 of 36        15 of 36           0
concept fidelity audit        34 of 36        34 of 36            0
explain() exact              100 of 100      100 of 100          0
mean explain residual        5.634e-07       6.217e-07          -
concept_share               99.999996%      99.999996%          0
```

**Final validation accuracy is identical to two decimals (19.95% at both widths).** Next-token accuracy moves +0.55, diversity
moves +5 tokens (44 -> 49), concept firing is 15 of 36 at both, and the explanation stays exact on 100 of 100 windows at both.

## Why this is a confirmation, not a surprise

The bottleneck was diagnosed as concept capacity: 15-18 of 36 concepts fire on a median window, so 36 half-saturated numbers cannot
address 12,002 output tokens (`docs/history/DIVERSITY-COLLAPSE-CAPACITY.md`). **If the concept layer is the choke point, then adding
hidden width above it should buy nothing - and that is exactly what happened.** The two independent facts agree:

```
earlier: concept share of the decision SHRINKS with width on a direct-path model (18.00% -> 8.89%)
now:     a readable model gains ~nothing from the same width increase (19.95% -> 19.95%)
```

On the direct-path model, width bought accuracy by routing *around* the concepts. On the readable model, where routing around is
impossible, width has nowhere to go. That is the strongest evidence yet that the 36-concept layer, not the hidden width, is the
binding constraint.

## Predictions, scored so far

- **P2 HOLDS** (readable diversity improves in absolute terms: 44 -> 49 distinct tokens, while staying well under the control).
- **P3 HOLDS** (explain remains exact at d1024: mean residual 6.217e-07, `concept_share` 99.999996%, 100 of 100 windows).
- **P1 NOT YET SCORED**: it is about the accuracy gap to the d1024 control, which is still training. No claim is made about it.

## What it means for the next step

Widening the hidden size is measured out as a route to a better readable chat model. **Widening the concept layer is the remaining
lever**, which is the owner's decision because it changes the concept definitions every audit compares against.
