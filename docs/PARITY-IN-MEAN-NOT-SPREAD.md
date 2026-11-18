# The parity replicated in the mean, not in the spread - and the spread is the control's fault (2026-11-18)

The parity arm was replicated at three seeds. The honest result is **not** "readable is at parity and stable". It is: **readable and plain have the same average accuracy, and the
plain model's accuracy swings far more between seeds.**

```
seed   readable    plain      gap
0       84.78%    84.86%    -0.08
1       84.21%    85.80%    -1.59
2       83.96%    81.52%    +2.44

readable: mean 84.32  sd 0.34  range 0.82
plain   : mean 84.06  sd 1.84  range 4.28
gap     : mean +0.26  sd 1.66   (bootstrap CI with 3 seeds is just the min/max: [-1.59, +2.44])
plain sd / readable sd = 5.4x
```

## Predictions, scored

- **P1** "3-seed mean gap between -0.6 and +0.6" - **HOLDS** at +0.26.
- **P2** "sd across seeds at most 0.6" - **FAILED**, at 1.66. The gap is a coin flip per seed.
- **P3** "honesty >= 85% every seed" - **HOLDS** (95.2, 95.2, 100.0).
- **FALSIFIER** "mean at or below -1.0" - **did not fire** at +0.26, so the parity claim stands for the mean and the earlier ~-1.5 figure is superseded **for this configuration
  only**.

## Why the gap swings, and why that is the finding

The readable model is tightly clustered (84.78, 84.21, 83.96 - a spread of 0.82 points) while the plain control jumps 4.28 points between seeds (84.86, 85.80, 81.52). Since every
seed trains both arms on the same data with the same recipe, this is not a data difference. The plain model's single wide hidden layer lands in very different local minima depending
on initialisation; the concept layer's 21 units are pinned to named formulas, which constrains them and stabilises what they learn.

A per-seed spread difference of 5.4x with n=3 seeds **is not a result** - three samples cannot establish a variance difference, and one lucky or unlucky seed moves the estimate.
It is recorded here as a **hypothesis with a pre-registered test**, not as a claim.

## What this changes in the project's story

The old story was "readability costs 1.5 points". The measured story is now:

- **Average accuracy: parity** (+0.26 points over 3 seeds, mean gap CI includes zero).
- **Seed-to-seed reliability: possibly much better (-5.4x spread), untested.**
- **Concept honesty: 95-100%.**
- **The layer is load-bearing**: zeroing it drops 84.15% -> 10.17%, shuffling it -> 9.55%.

## The test, pre-registered before it runs

`lens/scripts/run_capadv_seeds_more.sh` runs seeds 3-7 so there are **8 seeds per arm**:

- **P1** readable sd stays at or below 1.0 point
- **P2** the plain control's sd stays at or above 1.5 points
- **P3** the 8-seed mean gap stays within +-0.7 points
- **FALSIFIER** the plain sd falls below 1.0 with 8 seeds: then the 3-seed spread was an accident, there is no stability advantage, and **this page gets the retraction**.

The radius-1.0 chat arm is unaffected by any of this and continues on cores 4-7.
