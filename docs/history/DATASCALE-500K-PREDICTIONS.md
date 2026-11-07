# Does the data lever keep paying past 200k? 500k windows at seed 0 (predictions, before the run)

2026-11-07 10:3x IST, written before `lens/scripts/run_datascale_500k_s0.sh` is launched.

## Why 500k and not 2M

The measured curve at seed 0 at `d=256`, batch 256, 3 epochs is: **12.18% at 20k windows, 17.43% at 200k**
(`WALL-AB.txt:116`, `DATASCALE-HONEST.txt:230`). Ten times the data bought +5.25 pp. The project also has a
2M-window result at the same width but a **different batch** (1024) and 6 epochs, which cannot be paired with
these. So between 200k and the only larger rungs there is either a large unmeasured gap or a batch change -
and a 2M run at batch 256 costs ~3.4 h, which is too much to spend on a guess.

500k costs ~50 min, is one flag different from the 200k arm, and answers the question that decides whether the
next long run is worth it: **is the curve still rising at 200k, or has it flattened?**

## Predictions (paired to seed 0 only)

C1  LEVER STILL PAYING. 500k VAL beats seed 0's 200k value by >= +1.5 pp, i.e. **VAL >= 18.93%**.
    FALSIFIER: <= 17.43% would mean more data hurt; 17.43-18.93% would mean the curve has flattened into the
    seed spread and that 2M is not obviously worth hours of GPU.

C2  PERPLEXITY. ppl < 207.29 (seed 0's 200k value), same direction.

C3  CONCEPTS. The audit stays >= 36 of the kept set (the 200k arms kept 36 and scored 36/36; the 20k arms kept
    37 and scored 22/37, so the count is reported, not assumed).

C4  MONOTONE PATH. Epoch 3 >= epoch 2, as in every datascale arm so far.

## What will not be claimed

* Not comparable to the 200k *three-seed mean* of 19.21% - one seed only, so the pair is 17.43% -> 500k.
* No claim about 2M: C1 decides whether that run is scheduled, it does not measure it.
* No headline change unless it beats 37.37% (d=512, full corpus, 6 epochs), which 500k at d=256 will not.
* Batch stays 256 and epochs stay 3, so exactly one thing changes: `--max-windows`.
