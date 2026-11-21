

---

# TWO GAPS, NOT ONE: THE MODEL CANNOT READ BY POSITION AND CANNOT COMPUTE (2026-11-21 13:35)

Two runs finished. Neither produced the verdict I wanted, and together they changed the diagnosis again - this time into something
sharper and testable.

## 1. The memory precondition is BLOCKED by something more basic

`lens/runs/RECALL-r1.txt`, seed 0, one trained model scored at every retrieval distance 0..19:

```
near (q=0..2) mean  8.3%     far (q=16..19) mean  8.1%     Spearman(q, accuracy) -0.102
OUTCOME: CURVE_INCONCLUSIVE
```

**Flat at chance for every distance, including distance 0.** The model was not asked to reach far back - at q=0 it only had to repeat
the most recent digit - and it still failed. So this is not a long-range memory problem, and L1's precondition **cannot be measured
until reading-by-position works at all**.

## 2. The salience certificate still cannot issue a verdict, but its margin is consistently positive

Two attempts on `max2`, six arms:

```
attempt/steps   seed   AUROC    free norm   margin    Spearman   probe accuracy   verdict
2000           0      0.9357    0.8897     +0.0460    0.0079*       92.0%        SALIENCE_WORKS
2000           1      0.8826    0.8700     +0.0126    0.0627*       88.0%        INVALID_INSTRUMENT
2000           2      0.7193    0.7954     -0.0762    0.0957*       60.0%        INVALID_INSTRUMENT
5000           0      0.8843    0.8627     +0.0216    0.4334        90.0%        SALIENCE_FAILED (P1 bar 0.90)
5000           1      0.8393    0.8267     +0.0126    0.4308        82.0%        INVALID_INSTRUMENT
5000           2      0.8048    0.7573     +0.0476    0.3914        86.0%        INVALID_INSTRUMENT
```
\* the 2000-step Spearman values are invalid - see the bug below.

- **The margin over the free baseline is positive in 5 of 6 arms** (+0.0460, +0.0126, +0.0216, +0.0126, +0.0476). The layer does carry
  information the hidden norm does not.
- **AUROC ranges 0.72-0.94 and crosses the 0.90 bar once.** Not reliable.
- **Spearman 0.39-0.43** against a 0.50 bar - close, not met.
- **The instrument is the problem:** `max2` probe accuracy swings **60-92%** and the script refuses any verdict below 90%. Lowering that
  bar would be tuning the gate until it passes, so instead the certificate moved to `copy1`, which solved the same family of task at
  **98.0%**.

## 3. My bug, and why the magnitudes caught it

P3 reported Spearman **0.0079** on the arm whose AUROC was **0.9357**, while the measured separation was **0.3773** for important
positions against **0.0107** for decoys - a 35x gap. A layer ranking that cannot have an uncorrelated slope. The cause: the correlation
paired predictions for the EVAL sequences with the ground truth of the FIRST n_eval TRUTH sequences, two different sets. Both arrays had
the right shape, so nothing complained. After the fix: **0.43 / 0.43 / 0.39**. The bug was found by asking why two measurements
disagreed, not by reading the code.

## What this changes

The single story "the model cannot do arithmetic" is too coarse. There are at least **two separate gaps**:

1. **Reading by position** - the model cannot retrieve a token because of WHERE it is, even one position back.
2. **Arithmetic** - even when selection works (max2 92.0%), combining two values by addition fails (8.0%).

`copy1` (98.0%) and `max2` (92.0%) both work because they select by **content**: the informative tokens are digits (0-9) among decoys
(10-19), so no positional addressing is needed. `sum2fixed` puts the operands at **fixed positions** and fails at **2.0%** - the failure
appears exactly when position has to be used.

Running now: `copylast` (digit at the last position) against `copyfirst` (the same digit at position 0, with decoys after it). Identical
job, only the distance differs, so the comparison is causal. P1 copylast >= 90%, P2 copyfirst <= 25%; if BOTH pass, distance is not the
blocker and the arithmetic failures are purely arithmetic.
