# 2026-11-25 - data scale still pays (1M beats 200k), plus the harness that ends the instrument-bug class

## 1. Data scale: 5x the pool wins at equal steps

Source: lens/runs/DATA-SCALE.out (numbers printed before the verdict block crashed - see 3).

    pool          steps 25,600   steps 51,200   epochs at 51,200
      200,000        3.92173        3.79662          2.05
    1,000,000        4.00662        3.73593          0.41

At 51,200 steps the 1M pool is BETTER by 0.0607 even though each window had been seen only 0.41 times against 2.05 for the
200k pool. So distinct data was still paying at 5x the previous ceiling, and the 1M arm was nowhere near converged when it
won.

IMPORTANT COMPARABILITY WARNING: these absolute values (3.73-4.01) must NOT be compared with earlier runs (4.12-4.20) from
this morning. exp_data_scale.py draws its held-out windows from a different token range (2,000,000+ instead of 700,000+),
so the held-out SET is different and only within-run comparisons are valid. This is exactly the failure mode the new harness
removes: one fixed held set, recorded in the result file.

## 2. The harness (lens/harness.py, 214 lines)

One place where a held-out number can be produced, so the class of bug that produced twelve wrong results cannot recur:

  * `Corpus` owns the split. Train and held windows come from DISJOINT token ranges, asserted at construction (measured:
    train_hi=1,998,268 < held_lo=2,033,349, and zero shared start positions).
  * `TrainSet` / `HeldSet` are distinct types. `evaluate()` accepts only `HeldSet` and raises on `TrainSet` - the exact
    bug that voided the capacity run.
  * `train_steps()` accepts only `TrainSet` and raises on `HeldSet`.
  * Every result is written with corpus path, seeds, pool sizes, backend and parameter count.

Regression tests: tests/test_harness_regressions.py, 8 assertions, ALL PASS. Each names the bug it prevents.

MY OWN TEST WAS WRONG ONCE, which is worth recording: I first asserted that the curve {4.18790, 4.15636, 4.14550} must not be
called flat because it was "still paying". Its last interval gained only 0.0109, below the 0.02 materiality bar, so calling
it flat IS correct and the test expectation was the error. Corrected to assert the honest reading in both directions.

## 3. Bug 9 happened AGAIN, on the same verdict block

exp_data_scale.py was derived from exp_data_ceiling.py, which still had the hardcoded c[6400] / c[12800] / c[25600] verdict
keys. Changing MARKS to [25600, 51200] therefore crashed it with KeyError: 6400 AFTER printing valid numbers. Same bug, same
shape, second file. That is the argument for moving every experiment onto the harness instead of copying scripts forward.
