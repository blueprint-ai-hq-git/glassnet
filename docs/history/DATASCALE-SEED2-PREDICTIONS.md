# DATASCALE-HONEST seed 2: a third draw on the mean the public page now quotes (predictions, before the run)

2026-11-07 09:3x IST, written before `lens/scripts/launch_datascale.sh 2`.

## Why a third seed now

The public result is two seeds: 200k windows gives VAL 17.43% (seed 0) and 19.71% (seed 1) against 20k
baselines of 12.18% and 12.79%, i.e. +5.25 and +6.92 pp, mean +6.09 (`docs/history/DATASCALE-SEED1-RESULT.md`).
Those two seeds differ by **2.28 pp**, which is larger than the 1.03 pp run-to-run spread measured at the 20k
rung. So the mean on the public page rests on two numbers that disagree by more than the measurement noise,
and the honest way to tighten it is another draw, not a wider interval invented from two.

The schedule question is closed: 6 epochs bought +1.00 pp of endpoint and no better trajectory
(`docs/history/DATASCALE-EP6-RESULT.md`), so more steps are not what this rung needs.

## Design - identical to seeds 0 and 1

Same corpus (`prepared_mix_ctrl_honest`), same command, same 36 kept concepts, same 3,811,534 parameters, no
bypass, `--max-windows 200000 --epochs 3`, same 87,035-window held-out set (`n=44418` / `n=58191`).

    ONLY CHANGE: --seed 2

Baselines for this seed: there is no 20k arm at seed 2. The 20k baselines exist at seeds 0 and 1 only
(12.18%, 12.79%), so this arm is compared to the 200k distribution, not to a matched 20k run. A matched 20k
arm at seed 2 is a separate decision and is not run here.

## Predictions and their falsifiers

S1  BAND. VAL lands inside **15.18% - 20.18%** again (the band D1 used, and the one both existing seeds fell
    inside).
    FALSIFIER: outside that band. Above it means the band was too conservative; below it means the rung's seed
    variance is wider than two draws suggested.

S2  MEAN. The three-seed mean VAL lands in **16.5% - 20.5%**, i.e. the mean gain over the two 20k baselines
    stays between +4.3 and +8.3 pp.
    FALSIFIER: outside. A mean below 16.5% would say the effect is smaller than the two published values
    imply; above 20.5% would say seed 0 was the outlier low draw.

S3  CONCEPTS. The audit passes >= 30 of 36 (both existing seeds scored 36/36).
    FALSIFIER: < 25/36 would mean the concept jump is not a stable property of the configuration.

## What will not be claimed

* No significance test: three seeds are still a handful, and no p-value will be invented after the fact. What
  is published is the three values and their mean.
* The 6-epoch arm is not averaged in (E4), and this arm is not averaged with it either.
* No headline change: 37.37% (d=512, full corpus, 6 epochs) remains the best number.
* Cost: ~20 min of a free GPU at ~400 MiB.
