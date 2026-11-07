# DATASCALE-HONEST seed 2: a third draw lands just above the predicted band; the mean rises to +6.73 pp

2026-11-07 09:48:07 IST. Third draw at 200k windows, seed 2, one flag different from seeds 0/1 (`--seed`).
Predictions were written before the launch (`docs/history/DATASCALE-SEED2-PREDICTIONS.md`); graded by
`lens/scripts/datascale_seed2_grade.py`, which re-verifies every pinned source line on each run. Log:
`lens/runs/DATASCALE-HONEST-s2.txt`.

## The three draws

| run | seed | VAL acc | ASST acc | final ppl | concepts | source |
|---|---:|---:|---:|---:|---:|---|
| `datascale_honest_s0` | 0 | 17.43% | 18.61% | 207.29 | 36/36 | `DATASCALE-HONEST.txt:230` |
| `datascale_honest_s1` | 1 | 19.71% | 20.82% | 184.32 | 36/36 | `DATASCALE-HONEST-s1.txt:262` |
| `datascale_honest_s2` | 2 | **20.50%** | **21.47%** | **180.30** | 36/36 | `DATASCALE-HONEST-s2.txt:256` |

    20k baselines (seeds 0 and 1 only): 12.18% and 12.79%, mean 12.485%
    three-seed mean VAL: 19.213%          mean gain over the 20k baseline mean: +6.73 pp
    paired gains: +5.25 pp (seed 0), +6.92 pp (seed 1)
    seed-to-seed range over three draws: 20.50 - 17.43 = 3.07 pp

Seed 2's own path: VAL 15.67 → 17.55 → 20.50 (ppl 232.28 → 200.86 → 180.30), ASST 16.69 → 18.80 → **21.47**,
still improving at its last epoch - the same shape seed 1 had.

## The verdicts

    S1  seed-2 VAL inside 15.18-20.18       20.50%  -> OUTSIDE (0.32 pp above the top)  FAIL
    S2  three-seed mean inside 16.5-20.5    19.213% -> PASS   (mean gain +6.73 pp, inside +4.3..+8.3)
    S3  seed-2 concepts >= 30/36            36/36   -> PASS

S1 failed, and it failed in the direction the prediction doc had already named: *"Above it means the band was
too conservative."* So the honest reading is not that the result weakened - it is that the band derived from
a single seed-0 draw at 17.43 was too narrow, which is exactly the failure mode `REF-REPRO.md` warned about
when it showed three runs of one configuration spanning 1.03 pp.

## What three draws now say, and what they do not

* The effect is real and larger than the first seed suggested: **+5.25, +6.92 and (against the baseline mean)
  +8.02 pp** for seed 2, i.e. a three-seed mean gain of **+6.73 pp** and a mean VAL of **19.21%**.
* The seed-to-seed range is **3.07 pp** - three times the 1.03 pp run-to-run noise at the 20k rung. So the
  spread between seeds, not the measurement noise, is what limits how precisely this rung can be quoted, and
  no ± interval from two or three draws is trustworthy.
* The concept audit is **36/36 on all three draws**, so that effect is stable, not a seed accident.
* Checkpoint provenance: `verify_wall_ab_ckpt.py` on the seed-2 log -> 1 arm verified, 0 problems; `best.npz`
  is epoch 3 as the log named, 3,811,534 parameters (same architecture as every arm in this family).

## Not claimed

No significance test from three seeds, no headline change (37.37% at d=512 remains the best number), no talk
or ship claim, and the 6-epoch arm is still excluded from any averaging (E4).

## CORRECTION - 2026-11-07 10:1x

Two statements above are superseded, and the numbers they carried are replaced, not deleted:

* "20k baselines (seeds 0 and 1 only)" - false. Seed 2's 20k baseline existed all along as `wall_rep_off_s2`
  (`lens/runs/WALL-REP.txt:329`, VAL 13.40%), and it was re-measured on 2026-11-07 with an identical result.
* "+8.02 pp for seed 2 ... against the baseline mean" - a cross-seed comparison, now replaced by the
  within-seed **+7.10 pp** (20.50 - 13.40). Three paired gains: +5.25, +6.92, +7.10, mean **+6.42 pp**.
  Full record: `docs/history/DATASCALE-20K-S2-RESULT.md`.
