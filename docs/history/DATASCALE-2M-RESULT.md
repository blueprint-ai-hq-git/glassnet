# 2M windows: the lever bends flat - +0.48 pp for 4x the data, and the last epoch goes backwards

2026-11-07 13:21:12 IST, `DATASCALE-2M.txt`, `exit=0`, marker `DATASCALE-2M DONE 2026-11-07 13:21`.
Predictions written before the launch in `docs/history/DATASCALE-2M-PREDICTIONS.md`; graded by
`lens/scripts/datascale_ladder_grade.py 2m`, whose thresholds live in a spec table and are never edited after a
result.

## Result

    VAL 21.29%   ASST 22.60%   ppl 149.14   concepts 36/36      (ep 3/3, loss 4.8734; `DATASCALE-2M.txt:1142`, final VAL at `:1226`)
    path: ep1 21.10 (ppl 159.78) -> ep2 21.69 (149.90) -> ep3 21.29 (149.14)
    best checkpoint = epoch 2: held-out ASST 22.82% (`NEW BEST ... at epoch 2`), and ep3 is 0.40 pp BELOW ep2

Provenance: `verify_wall_ab_ckpt.py` -> 1 arm verified, 0 problems; `best.npz` is byte-identical to `_ep2.npz`
(`cmp` clean, 15,262,096 bytes each), which is what "best" should mean. Same 3,811,534 params, same 87,035-window held-out set,
one flag changed from the 500k arm (`--max-windows 500000` -> `2000000`).

## The verdicts

    T1  VAL > 20.81%                      21.29%  -> PASS (+0.48 pp)
    T2  VAL lands 21.5-24.0%              21.29%  -> FAIL, 0.21 pp BELOW the band floor
    T3  ppl < 171.47                      149.14  -> PASS
    T4  concepts >= 36                    36/36   -> PASS
    C4  ep3 >= ep2                        21.69 -> 21.29 -> FAIL

T2 failed on the low side, which is the direction my own prediction doc flagged as the risk: the naive
2.56 pp/doubling extrapolation implied ~25.9% and the doc said plainly that decay relative to it was expected,
with a band that assumed decay, not collapse. The band was wrong in the same direction as the naive slope, just
much less so. And C4 failed: the third epoch did not help - the best checkpoint is epoch 2.

## The curve at seed 0, and what it says

    20k     12.18%   ppl 399.50
    200k    17.43%   ppl 207.29    +5.25 pp / 3.32 doublings = 1.58 pp per doubling
    500k    20.81%   ppl 171.47    +3.38 pp / 1.32 doublings = 2.56 pp per doubling
    2M      21.29%   ppl 149.14    +0.48 pp / 2.00 doublings = 0.24 pp per doubling

    100x data in total buys +9.11 pp (12.18 -> 21.29), and the last 4x buys 5.3% of that.
    Perplexity keeps falling (399.50 -> 149.14) while accuracy stops rising - the model is getting more
    confident about the same answers.

So the answer to the question this arm was launched for is **no**: the data lever does not keep paying past 500k
at d=256, batch 256, 3 epochs. It bends flat between 500k and 2M, and 0.24 pp/doubling is inside the seed spread
(3.07 pp across three seeds at the 200k rung, `docs/history/DATASCALE-SEED2-RESULT.md`), so the last rung cannot
be distinguished from noise on one seed.

## A defect in my own pre-registration, admitted

`docs/history/NEXT-ARM-DECISION.md` branched on `>24.0` (B1), `21.5-24.0` (B2) and `<=20.81` (B3). The result
landed at **21.29 - inside a gap the branches never covered**. The branches did not tile the outcome space, which
is the same class of error as a band that cannot be satisfied. Both B2's premise (the lever still pays, but
decayed) and B3's (the ceiling is near) are partly true here, so the next arm has to be chosen with an argument
rather than by looking it up, and that choice is therefore weaker than the earlier ones - stated rather than
dressed up.

## Not claimed

* One seed. No interval, no significance test, and the 0.48 pp gain is not distinguishable from seed variation.
* Not a headline: 37.37% (d=512, full corpus, 6 epochs) stands, and this arm is 16.08 pp below it.
* Not comparable to the project's older 2M result at batch 1024 (24.20%): different batch, different sampling
  path. Two numbers with the same window count and the same name are not the same experiment.
