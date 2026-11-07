# The missing cell: the 20k baseline at seed 2 (predictions, before the run)

2026-11-07 10:0x IST, written before `lens/scripts/run_datascale_20k_s2.sh` is launched.

## Why this arm exists

The data-scale table has a hole. Seeds 0 and 1 each have a matched pair (20k and 200k at the same seed), so
their gains are **within-seed**: +5.25 pp and +6.92 pp. Seed 2 has only the 200k arm (VAL 20.50%,
`docs/history/DATASCALE-SEED2-RESULT.md`), so its "gain" had to be quoted against the *mean* of the two other
seeds' 20k baselines - a cross-seed comparison that every document currently flags.

This arm fills the hole: the 20k baseline at seed 2, same command as the wall A/B's `off` arm, seed 2. Then
seed 2's gain becomes within-seed like the others, and the three-seed mean gain stops leaning on a borrowed
baseline.

## Predictions

B1  BASELINE BAND. Seed-2 20k VAL lands in **12.10% - 12.90%**, i.e. within 0.3 pp of the two existing 20k
    values (12.18% at seed 0, 12.79% at seed 1; their spread is 0.61 pp).
    FALSIFIER: outside that band. Below 12.10% would make seed 2 an unusually weak baseline and would inflate
    its apparent gain; above 12.90% would deflate it.

B2  WITHIN-SEED GAIN. Seed-2's 200k value (20.50%) minus this 20k value lands in **+5.0 to +9.0 pp**, the range
    the two existing paired gains define (+5.25, +6.92).
    FALSIFIER: outside that range. It would not be wrong - it would mean the gain varies more across seeds than
    two draws suggested, and it would be reported as that.

B3  NO RE-GRADE. This arm does not change seed 2's S1 band failure (20.50% was 0.32 pp above 15.18-20.18) and
    does not touch the 200k numbers. It adds the baseline the table was missing.

## What will not be claimed

* No significance test: three paired gains are still a handful.
* No headline change: 37.37% (d=512, full corpus, 6 epochs) remains the best number.
* Cost: ~7 min of a free GPU at ~400 MiB, one epoch-3 checkpoint written by the trainer.
