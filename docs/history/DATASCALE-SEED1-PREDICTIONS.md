# DATASCALE-HONEST seed 1: does the 200k gain - and the concept jump - reproduce? (predictions, before the run)

2026-11-06 20:1x IST, written before `lens/scripts/run_datascale_honest.sh 1` is launched. Seed 0 result:
`docs/history/DATASCALE-RESULT.md`.

## What is already measured, and what is not

Seed 0 at 200k windows: VAL 17.43%, ASST 18.61%, ppl 207.29, concepts **36/36** (`DATASCALE-HONEST.txt:230`).
The 20k baseline: VAL 12.18%, ASST 13.00%, ppl 399.50, concepts 20/36 (`WALL-AB.txt:201`). That is +5.25 pp,
and a concept audit that went from 20/36 to 36/36 at the same 3,811,534 parameters.

One seed. The public page says so, and it says the number is a lead until it replicates. This run is that
replication, and it changes nothing in the command except `--seed 1`.

## Predictions

R1  ACC REPLICATES. VAL acc lands again inside **15.18% - 20.18%** (the same band D1 used for seed 0),
    i.e. the 10x-data gain is reproduced and not a seed-0 accident.
    FALSIFIER: VAL outside that band. Inside 12.18-15.18 would mean a real but smaller gain; below 12.18
    would mean the seed-0 gain was the outlier; above 20.18 would mean seed 0 undersold it.

R2  LOSS REPLICATES. Final VAL ppl below **300** (seed 0: 207.29; 20k baseline: 399.50).
    FALSIFIER: ppl >= 300.

C1  CONCEPT JUMP REPLICATES. The concept audit passes for at least **30 of 36** kept concepts.
    FALSIFIER: fewer than **25 of 36**. Then the 20/36 -> 36/36 jump at seed 0 was seed-specific, and the
    sentence that data scale (not the concept design) limits the concept line on STATUS.md must be withdrawn.

R3  SEED SPREAD, MEASURED. |seed 1 VAL - seed 0 VAL| is recorded whatever it is. If it exceeds 1.5 pp, then
    this rung's seed-to-seed spread is larger than the 1.03 pp run-to-run spread measured at 20k windows
    (`REF-REPRO.md`) and a two-seed mean cannot be treated as tighter than about +/-0.75 pp.
    This is a measurement, not a pass/fail - it is written down so it cannot be chosen after the fact.

## What will still not be claimed

* No "smart", no talk quality, no ship.
* No headline change: 37.37% (d=512, full corpus, 6 epochs) stays the best number.
* Two seeds is a replication, not a significance test. The result is reported as two values and their
  difference, never as "p < ..." invented afterwards.
