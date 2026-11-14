# The concept-supervision knob: 4.22 accuracy points buy most of the model's honesty, and AUROC was flattering both ends

Date: 2026-11-14. Accuracy from `gates/phase2_train.py` (K=14, R=32, keep=3, lam=0.005, 10 epochs, batch 128, lr 0.1, 3 seeds,
predictions in `lens/runs/ALPHA-SWEEP-PREDICTIONS.txt`). Honesty from `tools/phase2_reaudit.py` on the seed-0 checkpoints, whose
instrument is **balanced accuracy and rank correlation per concept**, not AUROC. Output saved to
`lens/runs/ALPHA-HONESTY-AUDIT.txt`.

## Accuracy and AUROC across alpha

```
alpha   accuracy (3 seeds)              per-seed              AUROC
1.0     82.68%                          82.84 82.54 82.67      0.805
3.0     78.46%                          82.76 79.13 73.50      0.879   <- shipped
6.0     74.34%                          75.57 73.02 74.42      0.896
10.0    63.51%                          56.88 61.52 72.14      0.918
```

Predictions: **P1 FAILED** - I expected the total fall from alpha 3 to 10 to be under 2 points; it is **14.95**. **P2 held in
direction** (AUROC rises monotonically, alpha 10 at 0.918, still under 0.92). **P3 FAILED** - no arm reached AUROC >= 0.90 with
accuracy >= 77.5. The falsifier did not trigger: AUROC is not flat.

Two things stand out beyond the means: **alpha 1.0 is remarkably stable** (82.84 / 82.54 / 82.67, spread 0.30) while alpha 3.0
swings 82.76 / 79.13 / 73.50 (spread 9.26) and alpha 10.0 swings 15.26. Stronger concept pressure buys honesty at the price of
seed-to-seed reliability.

## The honest instrument changes the verdict

`tools/phase2_reaudit.py` on the same two checkpoints (seed 0), measuring each concept neuron against **its own label**:

```
                                    alpha 1.0        alpha 3.0 (shipped)
mean |rank corr| with own label       0.561              0.732
concepts below 0.50                   7 of 14            2 of 14
concepts tracking ANOTHER feature     5 of 14            2 of 14
  worst offenders                    symmetry, vertical_edges, quad_UR_ink,
                                     quad_LR_ink, tall_shape    symmetry, vertical_edges
per-rule ablation median             +0.50 pts           +1.07 pts
dead rules                            1 of 32            0 of 32
concept interaction                   0.0%               0.0%
rule-pair lift median                1.07 (9 of 32 < 1.0)  1.12 (12 of 32 < 1.0)
```

**On the honest metric, alpha 1.0 is not usable for the readable-model goal: half its concept layer (7 of 14) does not track the
name attached to it, and 5 neurons predict a different feature better than their own.** AUROC said 0.805 vs 0.879 - a 0.074
difference that reads as "slightly less honest". The label audit says 7 vs 2 mislabeled neurons, which reads as "half the middle
layer is wrong". AUROC was the flattering instrument, exactly as the reaudit tool's own docstring warns.

So the alpha knob is not a free lunch in either direction:

```
alpha 1.0    82.68% accuracy     7 of 14 concepts mislabeled    <- accurate, not honest
alpha 3.0    78.46% accuracy     2 of 14 concepts mislabeled    <- the shipped compromise
alpha 6.0    74.34% accuracy     AUROC 0.896, label audit not yet run
alpha 10.0   63.51% accuracy     AUROC 0.918, label audit not yet run
```

**Honesty costs 4.22 accuracy points and still leaves 2 of 14 concepts lying.** That is the real state of the readable image
model, and it is a long way from "100% honest".

## Two structural facts the audit measured for free

- **Concept interaction is 0.0% in both arms**: the sum of individual concept effects equals the effect of removing all concepts.
  So the model is **purely additive** in its concepts - which is good news for `explain()` (contributions really do add up and
  there is no hidden interaction to disclose) and bad news for expressiveness.
- **Rules combine concepts that rarely appear together**: median rule-pair lift 1.12, with 12 of 32 rules below 1.0 at alpha 3.0
  (9 of 32 at alpha 1.0). A rule that fires on co-occurrence patterns the data almost never shows is hard for a human to check.

## What the next honest improvement looks like

Since more pressure makes accuracy collapse faster than it fixes labels, the lever is **the concept definitions**, not the
pressure: replace or refine the concepts that are demonstrably mislabeled (`symmetry`, `vertical_edges` track other features at
both alphas), then re-run this same audit. That is the only path to "every middle neuron means its name" that does not simply pay
in accuracy.

The next run should therefore audit alpha 6.0 and 10.0 on the label instrument before either is considered - AUROC alone must not
be used to claim honesty for this model again.
