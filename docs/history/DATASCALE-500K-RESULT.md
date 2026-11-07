# 500k windows at seed 0: VAL 20.81%, all four predictions pass, and the curve has not flattened

2026-11-07 11:06:45 IST, `DATASCALE-500K.txt`, `exit=0`, marker `DATASCALE-500K DONE 2026-11-07 11:06`.
Predictions were written before the launch (`docs/history/DATASCALE-500K-PREDICTIONS.md`); graded by
`lens/scripts/datascale_500k_grade.py`, whose thresholds were fixed in that file and whose predecessor pins are
re-read and re-verified on every run.

## Result

    VAL 20.81%   ASST 21.82%   ppl 171.47   concepts 36/36   (ep 3/3, loss 4.9487; DATASCALE-500K.txt:392)
    path: ep1 20.03 (ppl 185.17) -> ep2 20.05 (173.29) -> ep3 20.81 (171.47)
    best checkpoint = epoch 3, held-out ASST 21.82% (`NEW BEST ... at epoch 3`, best.npz 15,262,004 bytes)

Provenance: `verify_wall_ab_ckpt.py` on this log -> **1 arm verified, 0 problems**; `best.npz` is the epoch the
log named. Same 3,811,534-parameter architecture, same 87,035-window held-out set, same command as the 200k arm
with one flag changed (`--max-windows 200000` -> `500000`).

## The verdicts

    C1  VAL >= 18.93% (200k + 1.5 pp)   20.81%  -> PASS, +3.38 pp over the 200k arm
    C2  ppl < 207.29                    171.47  -> PASS
    C3  concepts >= 36 of the kept set  36/36   -> PASS
    C4  ep3 >= ep2                      20.05 -> 20.81 -> PASS

No falsifier fired. Nothing was reinterpreted after seeing the number.

## The curve at seed 0

    20k    12.18%   ppl 399.50
    200k   17.43%   ppl 207.29     +5.25 pp for 10x data  (3.32 doublings -> 1.58 pp/doubling)
    500k   20.81%   ppl 171.47     +3.38 pp for 2.5x data (1.32 doublings -> 2.56 pp/doubling)

The marginal gain per doubling went **up**, not down (1.58 -> 2.56 pp/doubling), on a single seed. So the honest
statement is narrow and strong: **between 200k and 500k the data lever did not flatten** - and the "is 2M worth
hours of GPU?" question this arm was launched to settle now has evidence on the yes side, with the caveat that
one seed cannot distinguish a real steepening from seed variation.

This is also the highest d=256 value measured so far (20.81% against 19.71 / 20.50 at 200k and a 19.21%
three-seed mean), but the comparison that counts is the paired one: **17.43% -> 20.81% at seed 0**.

## Not claimed

* One seed. No interval, no significance test, and the steepening is not established at n=1.
* Not a headline: **37.37%** (d=512, full corpus, 6 epochs) is still the best number the project has, and this
  arm is 16.6 pp below it.
* No extrapolation to 2M. The next arm measures that; this one only says the curve was still rising at 500k.
* Concept count is reported, not reconciled: the audit prints 36/36 and this run kept 36.
