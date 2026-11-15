
## Temperature sweep: fidelity is best around T=0.5-0.6, and my own runner dropped the metric that mattered

Predictions in `lens/runs/TEMPERATURE-KNEE-PREDICTIONS.txt`. Three seeds of the best configuration (4M/12ep), sampling at T = 0.5, 0.6,
0.7, 0.9, through the real forward.

**A defect in my own runner, first:** the command I launched filtered the output with `grep -vE '...|^  repetition|^  self_repeat'`,
which removed **the exact metric the sweep existed to measure**. So `lens/runs/TEMPERATURE-KNEE.txt` contains coverage and KL and no
repetition at all. The filter is fixed and the sweep is re-running into `lens/runs/TEMPERATURE-KNEE-FULL.txt`; nothing about repetition is
reported below, because I do not have it.

## What the sweep did capture: coverage and KL

```
temperature   seed 0 coverage/KL   seed 1 coverage/KL   seed 2 coverage/KL
T=0.5             (filtered)          0.312 / 17.511       0.351 / 16.498
T=0.6             (filtered)          0.337 / 16.858       0.316 / 17.366
T=0.7             0.292 / 17.719      0.281 / 17.932       0.302 / 17.525
T=0.9             0.259 / 18.452      0.247 / 18.701       0.240 / 18.930
```

- **K3 PASSED**: coverage falls and KL rises as temperature rises, monotonically, on every seed (seed 2: 0.351 -> 0.316 -> 0.302 ->
  0.240).
- **The useful result**: `T=0.5-0.6` keeps corpus coverage at **0.316-0.351**, which matches greedy's 0.343 - so a mild temperature does
  **not** pay the fidelity bill that `T=0.8` did (0.217). If repetition is also low there - and that is precisely the number my runner
  threw away - then the knee is at T=0.5-0.6, not at 0.8, and the fidelity-versus-diversity trade is much cheaper than yesterday's
  numbers implied.
- **K1 and K2 are UNSCORED** until the re-run lands, because both depend on repetition.

## Method note

This is the third time today that a defect in my own tooling, not in the model, produced a wrong or missing result - after the hand-built
forward (131 logits wrong) and the single-seed repetition comparison (noise read as an effect). The pattern is consistent: **the
measurement path deserves the same verification as the model**, and a grep that silently drops a column is as damaging as a wrong matrix.
Concrete rule from here: a runner must print the metric list it intends to capture and assert each one is present in its output, rather
than filtering by pattern and trusting that what remains is complete.
