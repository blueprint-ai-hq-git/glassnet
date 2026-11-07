# 2M windows at seed 0: does the lever keep paying past 500k? (predictions, before the run)

2026-11-07 11:3x IST, written before `lens/scripts/run_datascale_2m_s0.sh` is launched.

## Where this sits

    seed 0, d=256, batch 256, 3 epochs, honest mix, one flag changing each time:
      20k    12.18%  ppl 399.50
     200k    17.43%  ppl 207.29     +5.25 pp for 10x   = 1.58 pp/doubling
     500k    20.81%  ppl 171.47     +3.38 pp for 2.5x  = 2.56 pp/doubling

The 500k arm was launched to answer "has the curve flattened?" and answered no, with the marginal gain per
doubling rising instead of decaying - on one seed, so the slope is not established. This arm is the next rung on
the same ladder: 4x the 500k data, ~2.1 h at the measured 10.6 min/epoch.

The project already has a 2M-window result (24.20% at batch 1024, full-corpus sampling). It is **not
comparable** to these arms - different batch and a different sampling path - so this run does not replace it and
must not be confused with it.

## Predictions

T1  STILL PAYING. VAL > 20.81%, i.e. strictly above the 500k arm at the same seed.
    FALSIFIER: <= 20.81% means the lever saturated between 500k and 2M.

T2  HOW MUCH. VAL lands in **21.5% - 24.0%** (+0.7 to +3.2 pp), i.e. **1.5 to 3.0 pp per doubling** over the two
    doublings from 500k to 2M. The 500k arm alone would suggest 2.56 pp/doubling (=> ~25.9%), but that slope
    rests on two points with one seed, so the band is deliberately wider than the point estimate rather than
    equal to it. Stated plainly: I expect decay relative to the naive extrapolation and continued gain in
    absolute terms.
    FALSIFIER: below 21.5% (decay faster than the band) or above 24.0% (the lever is stronger than the band).

T3  PERPLEXITY. ppl < 171.47.

T4  CONCEPTS. Audit >= 36 of the kept set.

Context that is NOT a prediction: at 2M x 32 = 64M tokens for 3,811,534 params this arm is **16.79
tokens/param**. Observed in its own header after the launch:
`tokens per parameter: 16.79  (well trained)   [windows/params 0.525; windows 2,000,000 x 32 tokens]`
(`DATASCALE-2M.txt`) - the first rung in this family the trainer calls "well trained", and the first run to
print the figure correctly (the old label would have said 0.53, under the "UNDER-TRAINED" line;
`docs/history/TOKENS-PER-PARAM-MISLABEL.md`).

## What will not be claimed

* One seed; no interval, no significance test, and no statement about the 2M/seed-1 or batch-1024 rungs.
* No headline change unless it beats 37.37% (d=512, full corpus, 6 epochs). It will not.
* Cost and guards: 2.1 h, ~490 MiB VRAM, checkpoints every epoch (~42 min, inside the 2 h rule), CPU pinned to
  2 cores with `taskset -c 0,1` so the machine stays usable if Roblox is started; GPU-CLAIM held.
