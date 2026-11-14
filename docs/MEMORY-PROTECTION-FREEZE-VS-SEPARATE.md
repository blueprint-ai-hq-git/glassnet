# Protecting memorised copy: freezing works here, and the "separate memory table" arm was void twice

Date: 2026-11-14. Question (one of the project's standing goals): can a model keep what it has memorised while it keeps learning - and does
giving the memory its OWN table do that better than freezing the matrices?

## The one result that survived: freezing protects copy
Base `early_s0_ep1000` (copy: 83.0% taught / 71.0% held-out), 4000 steps of language-only batches, 2 seeds each:

```
arm                          held-out copy            language
                             seed0  seed1  mean       seed0  seed1  mean
freeze ptrW/Wq/Wk            67.2   67.3   67.2       16.20  16.65  16.42
no protection (control)      56.7   61.3   59.0       17.70  17.62  17.66
difference                   +10.5  +6.0   +8.2       -1.50  -0.97  -1.24
```

Freezing the pointer matrices keeps **+8.2 points** of held-out copying at a cost of **1.24 points** of language.
Same direction as the earlier five-seed result (+2.07), larger here. Source: `logs/split51m/frz2_s*.log`,
`logs/split51m/non2_s*.log`.

## The arm that was supposed to test the design was void, twice

**Attempt 1 void:** all six arms reported taught copy **0.0%** - impossible for a learnable copy task. Cause: I built the
command from memory and started from a LANGUAGE checkpoint (`matched_d512_s0_best`) instead of the copy base, at
`--corpus-frac 0.5` instead of `1.0`. This project had already measured that language training destroys copying.

**Attempt 2 void:** the "separate memory" arms came out identical to the control **to the decimal** - held-out 56.7 /
61.3 and language 17.70 / 17.62 on both seeds, and **no** `SEPARATE embedding` line in either log. That is the signature
of a flag that does nothing. Cause, found by reading the trainer: the `if a.sep_emb:` block was **nested inside
`if a.pointer:`**. My arms copied `COMMON` from the freeze script, which never passes `--pointer`, so the block was
skipped and both arms were the same arm.

**And the obvious fix - just add `--pointer` - destroys the thing being measured.** Inside that block the trainer also
sets `conf_gate`/`amp_gate`/`route` from the flags and rebuilds `ptrW`/`Wq`/`Wk`. Measured on this base:

```
without --pointer   before training: taught 83.0%  held-out 71.0%
with    --pointer   before training: taught 19.2%  held-out  0.0%
with --pointer --keep-pointer-weights (new)  taught 70.3%  held-out 0.0%   (re-reading the gates alone breaks it)
```

So the two conditions were mutually exclusive: the table only existed when the pointer was re-attached, and
re-attaching wiped the memory the experiment is about.

## The code changes, and why they fail loudly instead of silently
`lens/scripts/train_copy_task.py`:
- the sep-emb block **moved out** of `if a.pointer:`; `--sep-emb` now either builds the table or `SystemExit`s with an
  explanation, because a flag that silently does nothing is worse than one that refuses to run;
- new `--keep-pointer-weights` so an arm can re-read the pointer flags without rebuilding the trained matrices;
- verified after the change: `--sep-emb` alone prints `SEPARATE embedding ... pemb (12002, 512)` and the base stays at
  **83.0% taught / 71.0% held-out**.

## Predictions, scored
- P1 (sep holds copy within 1 point of freeze): **VOID** - the arm was the control.
- P2 (sep costs less language than freeze): **VOID** for the same reason.
- P3 (both beat control by >= 1.5): freeze **+8.2 PASSED**, sep **VOID**.
- Falsifier (sep > 1 point worse than freeze on copy): could not be evaluated.

## Running now
`lens/scripts/run_split_mem_test_v3.sh`: the sep arm re-run properly (`--sep-emb` alone, table engaged, base intact at
83.0/71.0) against a matched control, 2 seeds, with P2 requiring sep to hold held-out copy **at least 3 points** above
its control. Falsifier: within 1 point means the separate table does not protect memory in this configuration and the
idea needs a different mechanism, not a bigger table.
