# The readable image model, scored on all four questions (2026-11-18)

The readable-vs-plain comparison is now settled on four separate questions at 8 seeds each. Two are wins, one is parity, one is a loss, and they are reported together because a
single headline would hide most of this.

```
                                    readable        plain
mean accuracy (8 seeds)              84.14%         84.44%      gap -0.30 points
seed-to-seed sd                      0.49           1.26        readable is 2.6x tighter
worst seed                           83.36%         81.52%
best seed                            84.78%         85.80%
spread                               1.42 points    4.28 points

epochs to reach a threshold (mean over 8 seeds)
   70%      1.0             1.0          1.00x
   75%      1.1             1.5          1.33x
   80%      3.2             2.9          0.88x
   82%      5.9             6.4          1.09x
   84%     14.9            10.7          0.72x
```

## 1. Mean accuracy: parity

**-0.30 points over 8 seeds.** With a gap sd of 1.22 the standard error is 0.43, so the mean gap is well under one standard error from zero. The readable model is neither
reliably better nor reliably worse than a plain model of the same size.

## 2. Seed-to-seed stability: a real advantage, smaller than first measured

Readable spread is **1.42 points against the plain model's 4.28**, a standard deviation of **0.49 against 1.26** - **2.6x tighter**. The 3-seed run had estimated 5.4x, and
prediction P2 ("plain sd stays >= 1.5") **failed** at 1.26, so the direction held and the magnitude shrank. It is reported as 2.6x, from 8 seeds, not as 5.4x from 3.

Why it makes sense: 21 units pinned to named formulas cannot drift far from the functions they are supervised to compute, while one wide free hidden layer lands in a different
local minimum on every initialisation.

## 3. Learning speed: no advantage

Epochs-to-threshold shows **no trend in either direction** - readable is faster at 75% (1.33x) and slower at 80% and 84% (0.88x, 0.72x). A readable model does not learn faster
than a plain one. It is a **negative result** and it contradicts the intuition that structured representations should train quicker at this scale.

## 4. Transparency: the thing the architecture exists for

- **20 to 21 of 21 concepts pass their own honesty audit** on every seed (95-100%).
- Zeroing the concept layer drops accuracy from 84.62% to **9.98%**; shuffling it drops it to **9.55%**, so the layer carries the work and the answer layer reads per-image
  content rather than the layer's average.
- Every middle neuron is a formula a person can recompute by hand, and the model publishes them: `glassnet-readable-vision-21` on the Hub carries all 21 concept formulas in
  `config.json` alongside the weights.

## The honest one-line summary

**At matched parameters this architecture is not more accurate and not faster to train; it is markedly more predictable seed-to-seed, and it is fully readable.** Anyone claiming a
readable model wins on accuracy at this scale is claiming something this project measured and did not find.
