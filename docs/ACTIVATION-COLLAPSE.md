# Why the readable model talks narrowly: measured mechanism, and a hypothesis killed (2026-11-15)

The readable chat model speaks in 86-135 distinct tokens where its direct-path control uses 226. Widening the concept set from 36 to 53
bought only +8 tokens (docs/history/CONCEPT-WIDENING-53.md). This document measures WHY, and it kills the first explanation I had.

## Hypothesis 1: the answer map's RANK is the ceiling - REFUTED

The readable model's answer layer is a matrix of shape (53, 12002), so every logit vector it can produce is `b + a @ W` with `a` in
R^53: a 53-dimensional family inside R^12002. The control has the same structure over 36 concepts **plus** a direct term of rank up to
1024 (`emb @ dmix`, 32768 x 1024). It looked like a clean, mechanical explanation.

```
model                     answer.W shape   numerical rank   effective rank (participation ratio)
readable 53-concept         (53, 12002)          53                     52.26
direct-path 36-concept      (36, 12002)          36                     35.97  (+ rank-1024 direct term)
```

Then I sampled the SAME 53-dimensional family at random, which is the test that decides whether the family is the limit:

```
observed predictions, 4,000 held-out windows :  135 distinct tokens
random a ~ N(observed mean, observed sd),
40,000 draws                                 :  341 distinct tokens
uniform a inside the observed per-dim range  :  212 distinct tokens
observed a + 3sd jitter                       :  132 distinct tokens
```

**The same family reaches 341 distinct tokens when sampled randomly; the model only uses 135.** So the rank of the answer map is NOT
what caps vocabulary. The hypothesis is dead and the planned "rank per concept" surgery was not built on it.

## Hypothesis 2: the concept activations collapse into a few directions - SUPPORTED

Measurements on 4,000 held-out windows:

```
model                     concept dims   effective dimensionality   variance in top-5 directions
readable 53-concept            53             10.37 of 53 (20%)                 54%
direct-path 36-concept         36              2.97 of 36  (8%)                 82%
```

**The readable model's 53 concept numbers move, in practice, through about 10 effective directions, and 54% of their variance sits in
five.** No dimension is dead (0 of 53 have vanishing variance), which is why the concept audits still pass - the neurons do fire on
their names - but collectively they describe a much smaller space than their count suggests. That is why breadth barely moved when the
count grew: more names did not buy more independent directions.

The direct-path model's concept layer is even MORE collapsed (2.97 of 36), which is consistent with everything else measured here: its
concepts are carried along for the ride, and its breadth comes from the direct path, not from them.

## What this changes

- **Not more concepts, and not more rank per concept.** Both target a ceiling that was measured not to exist.
- **The target is the spread of the concept activations**, which is a property of how the concept layer is trained, not of its size.
- The cheapest lever that touches it is the concept supervision weight; the principled one is a term that penalises correlated concept
  activations. The supervision lever is config-only, so it runs first: `lens/runs/ACTIVATION-COLLAPSE-PREDICTIONS.txt`.
