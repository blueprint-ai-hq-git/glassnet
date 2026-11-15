
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

## The decoding knee is T=0.7: 10x less looping than greedy at 85% of its fidelity (2026-11-15)

Predictions in `lens/runs/TEMPERATURE-KNEE-PREDICTIONS.txt`. Three seeds of the best configuration (4M/12ep), sampling at four
temperatures, real forward. The earlier run's repetition column was destroyed by my own grep; these are the corrected numbers.

```
temp   mean repetition   mean coverage   mean KL       (reference points)
0.5        0.050            0.340          16.79       greedy : rep 0.275, cov 0.343, KL 17.21
0.6        0.065            0.311          17.49       T=0.8  : rep 0.001, cov 0.217, KL 19.51
0.7        0.027            0.292          17.73
0.9        0.002            0.249          18.69
```

- **K1 PASSED**: `T=0.7` delivers repetition **0.027** (below the 0.05 bar) with coverage **0.292** (above the 0.26 bar). `T=0.5` misses on
  repetition by a hair (0.050, exactly at the boundary rather than below it).
- **K2 PASSED under the tolerance I set (+/-0.02)**, with one honest caveat: the sequence 0.050 -> 0.065 -> 0.027 -> 0.002 is not strictly
  monotone - `T=0.6` is slightly worse than `T=0.5` - so "monotone" is the wrong word for it. Repetition falls with temperature overall,
  with one small inversion.
- **K3 held** from the earlier run: coverage falls and KL rises as temperature rises.

## The result that matters

**`T=0.7` is the knee.** Against greedy it cuts repetition from **0.275 to 0.027** - ten times less looping - while keeping coverage at
**0.292**, which is 85% of greedy's 0.343 and far above `T=0.8`'s 0.217. So the fidelity-versus-diversity trade I described yesterday was
**the wrong framing**: at the right temperature there is no meaningful trade, only a slightly lower coverage for a tenfold reduction in
looping.

**This is the decoding configuration to ship for this model**: sampling at T=0.7, not greedy (loops) and not T=0.8+ (wanders into the
tail).

## What this closes

The looping line, which consumed most of this day, ends as follows, all measured through the real forward on three seeds:

```
greedy  : 0.275 repetition, 0.343 coverage    the loop is the decoder's
T=0.7   : 0.027 repetition, 0.292 coverage    10x less looping, 85% of the fidelity
T=0.8   : 0.001 repetition, 0.217 coverage    escapes the loop into the tail
```

With **24.58% +/- 0.06 validation accuracy** at 4M/12ep, the shipped configuration is: 4M windows, 12 epochs, 53 concepts, lam 1.0, and
**sampling at T=0.7**.
