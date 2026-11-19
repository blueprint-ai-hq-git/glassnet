

---

# THE MEAN-SINGULAR-VALUE REPAIR FAILED BADLY: -11.28 ACCURACY POINTS WITH A LOWER LOSS (2026-11-19 12:47)

The repair came from a measurement: the memory operator `diag(1-tanh^2) @ Wh` had a **mean singular value of 0.5613**, so the average direction
lost 44% of its perturbation per step and 16-step retention was predicted at 0.01%, while the spectral radius was a satisfied 0.9. The repair was
`renorm_mean_sv`, lifting the mean to 0.95 with the largest singular value capped at 2.0.

Predictions written before the run: P1 the post-training operator mean lands near 0.95 instead of 0.56; P2 held-out within 1.5 points of the
control; P3 delayed recall at 16 steps improves. **Falsifier: P2 failing by more than 1.5 points means instability.**

```
arm     what it does                        HELD-OUT            loss      train ema   gap
svoff   spectral-radius clamp (0.9)         17.70%             9.5525     7.0570    +2.4955
svon    mean singular value -> 0.95         6.42%              8.6336     7.4039    +1.2297

difference                                 -11.28 points       -0.92
```

**The falsifier fired on its most severe branch: -11.28 points, not -1.5.**

## The interesting part: accuracy collapsed while LOSS IMPROVED

`svon` has a **lower held-out loss** (8.6336 vs 9.5525) and a **far lower top-1 accuracy** (6.42% vs 17.70%). A lower loss with worse accuracy
means the model's probability mass is spread more sensibly across many plausible tokens while it almost never puts the single best token on top.
Raising the mean singular value appears to have flattened the readout's confidence: better calibrated in aggregate, useless for greedy prediction.

So the repair did not "fail to help" - it **changed what the model optimises toward**, and the metric that got better is not the one that matters
for a model that has to answer with one token.

## Third appearance today of the same trap

This is the third arm today whose **generalisation gap shrank while the model got worse**: plain-vs-rarity (gap +2.87 -> +0.77 as accuracy fell
2.91 points), the tied readout at 600 steps, and now this (+2.4955 -> +1.2297 as accuracy fell 11.28 points). A smaller held-out-minus-train gap
is a consequence of fitting less, not evidence of generalising better. Any future arm that reports a shrinking gap must report accuracy beside it.

## What this does NOT refute

The retention measurement stands: the operator's mean singular value **is** 0.5613 and **does** imply ~0.01% retention over 16 steps. What failed is
the assumption that forcing that number toward 1.0 during training is free. It is not - it costs 11 accuracy points. The memory limitation is real;
this particular repair for it is not viable.

## Next

The remaining path from that measurement is to raise the mean singular value **only at inference/recurrency** (or with a much gentler target and a
tighter cap) rather than constraining the matrix every training step, and to measure retention directly with the delayed-recall probe rather than
trusting next-token accuracy as a proxy. At a gentler setting the question becomes "how much accuracy does each step of mean singular value cost",
which is measurable as a curve instead of a single failed point.
