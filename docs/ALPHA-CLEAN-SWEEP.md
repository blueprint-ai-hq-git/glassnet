# Re-tuning alpha on clean labels: 2.56 accuracy points are available, at the price of one mislabeled concept in one seed

Date: 2026-11-14. Tool `gates/phase2_train.py` with the de-duplicated 14-concept set (`--concept-names`), honesty from
`tools/phase2_reaudit.py`. Predictions in `lens/runs/ALPHA-CLEAN-PREDICTIONS.txt`, raw output in
`lens/runs/ALPHA-CLEAN-SWEEP.txt` and `lens/runs/ALPHA-CLEAN-AUDITS.txt`. Architecture fixed: R=32, keep=3, lam=0.005, 10 epochs,
batch 128, lr 0.1, 3 seeds, 11,800 parameters.

## Why re-tune at all

alpha=3.0 was chosen when the concept labels correlated up to 0.916 with each other. On redundant labels a weakly supervised
neuron can drift to whichever nearby feature is easiest, which is exactly what happened (alpha 1.0 gave **7 of 14** mislabeled
concepts on the old label set). With the de-duplicated set (pairwise |corr| median 0.071, max 0.587) there is no nearby feature to
drift to, so the prediction was that honesty would become cheap.

## Accuracy falls monotonically as alpha rises

```
alpha 0.5  83.31 / 82.71 / 83.36   mean 83.13  sd 0.29
alpha 1.0  82.51 / 82.50 / 83.26   mean 82.76  sd 0.35
alpha 2.0  82.00 / 80.29 / 81.18   mean 81.16  sd 0.70
alpha 3.0  81.29 / 79.53 / 79.78   mean 80.20  sd 0.78   <- the shipped setting
```

## And honesty costs less than it used to, but not nothing

```
alpha  own-label corr (seeds 0/1/2)   below 0.50   mislabeled        rule-pair lift
0.5    0.711 (seed 0)                 1 of 14      1 of 14                0.87 (20 of 32 bad)
1.0    0.786 / 0.772 / 0.775          0 of 14      0 / 0 / 1 of 14        0.79 (19 of 32)
3.0    0.846 / 0.847 / 0.848          0 of 14      0 / 0 / 0 of 14        0.92 (19 of 32)
```

**The honest reading is a choice, not a winner.** alpha 1.0 buys **+2.56 accuracy** (82.76 vs 80.20) and keeps 0 concepts below
0.50 in every seed, but its mean own-label correlation drops from 0.846 to 0.772-0.786 and one seed has one concept tracking a
different feature better than its own. alpha 0.5 is faster still (83.13) and clearly worse on every honesty measure. alpha 3.0
remains the only setting with **0 mislabeled concepts in all three seeds**.

Predictions, scored:
- **P1 PARTLY HELD**: the >= +1.5 accuracy gain held (+2.56), the "at most 1 mislabeled" held (0/0/1), but my stated bar of
  "mean own-label corr >= 0.80" **failed** at 0.772-0.786.
- **P2 HOLDS**: the accuracy spread from alpha 0.5 to 3.0 is 2.93 points, under my 3-point bar - but only just, so the curve is
  not as flat as predicted.
- **P3 HOLDS**: median rule-pair lift stays <= 1.05 in every arm (0.79-0.92), confirming rule plausibility is **independent** of
  concept honesty - the two problems are separate.
- The falsifier (alpha 1.0 producing >= 2 mislabeled concepts) did **not** trigger.

## Where the readable image model now stands

```
matched plain control (10,345 params)                     84.37% +/- 1.04
readable, 14 clean concepts, rules, alpha 1.0 (11,800)    82.76% +/- 0.35    -1.61
readable, 14 clean concepts, rules, alpha 3.0 (11,800)    80.20% +/- 0.78    -4.17
readable, no rule layer at all (A3)                       83.13% +/- 0.42    -1.24
```

**At alpha 1.0 the full readable stack - 14 honest concepts, 32 rules, L1-trained sparse wiring - is 1.61 points behind a plain
network of the same size**, against 4.24 before the concept labels were fixed. The remaining 1.61 is now the honest price of
`explain()` on this task.

## Two honest notes

1. **Rule plausibility is unsolved and independent of alpha**: 19-20 of 32 rules wire together concepts that co-occur less than
   chance would predict (median lift 0.79-0.92). Every concept can be honest while the rules remain hard for a human to
   justify. This is the next honesty problem, not a side note.
2. **Reproducibility is not exact**: the identical alpha 3.0 configuration measured 80.13% in the earlier run (`SET14`) and 80.20%
   here. A ~0.07-point wobble across identical configs on the GPU backend, likely non-determinism in the kernels. Small, but it
   means differences under ~0.2 points should not be treated as real on this task.
