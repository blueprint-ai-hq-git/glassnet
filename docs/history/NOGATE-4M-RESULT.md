# 4M without the gates: 23.74% - the ungated lever is still paying, and it plateaued inside the run

2026-11-07 16:58:29 IST, `lens/runs/DATASCALE-4M-NOGATE.txt:69`, `exit=0`, marker
`NOGATE-4M DONE 2026-11-07 16:58`. Predictions written before the launch in
`docs/history/NOGATE-4M-PREDICTIONS.md`; graded by `lens/scripts/datascale_ladder_grade.py nogate4m`.

## Result

    VAL 23.74%   ASST 25.71%   ppl 95.99   train 34.68%   concepts 36/36
    path: ep1 23.11 (ppl 106.06) -> ep2 23.75 (97.08) -> ep3 23.74 (95.99)
    best checkpoint = epoch 3 (ASST 25.71%); 484/490/483 s per epoch; 3,811,534 params

Provenance: `verify_wall_ab_ckpt.py` -> 1 arm, 0 problems. Mechanical: 0 agreement-step lines, 0 gate-step lines,
`METHOD: OURS STEP RULE, GATES OFF`.

## The verdicts

    V1  VAL > 23.12%                         23.74%  -> PASS (+0.62 pp)
    V2  lands 23.4-24.6%                     23.74%  -> INSIDE (PASS)
    V3  ppl < 104.71                         95.99   -> PASS
    V4  mechanical                           0 / 0   -> PASS
    V5  concepts >= 36                       36/36   -> PASS
    C4  ep3 >= ep2 (generic check)           23.75 -> 23.74  -> FAIL by 0.01 pp

All five predictions pass. The one failure is the epoch-over-epoch check, and it is worth stating rather than
rounding away: **ep2 and ep3 are the same number to two decimals.** Three epochs at 4M have plateaued - the run
was still improving on perplexity (97.08 -> 95.99) while accuracy did not move.

## The ungated ladder

    500k   22.05%  (ppl 140.27)   72 s/epoch
    2M     23.12%  (ppl 104.71)   +1.07 pp for 4x = 0.54 pp/doubling
    4M     23.74%  (ppl  95.99)   +0.62 pp for 2x = 0.62 pp/doubling

The per-doubling rate has not decayed (0.54 then 0.62) - the ungated data lever is still paying at 4M, which is the
opposite of what the gated ladder said at 2M this morning. Cost grows linearly with data: 2x windows cost 1.95x
epoch time (487 s against 249.5 s).

23.74% is the best d=256 number **in the capped-window ladder** (<=4M windows, 3 epochs), 2.45 pp above the gated 2M arm
and 13.63 pp below the d512 headline (37.37%). **It is not the best d=256 number in the project**: full-corpus d=256
runs reach **35.72% by position** (`docs/history/STATE.md:146`, 16.9M windows, 6 epochs), 10.01 pp higher on the
comparable metric. The earlier wording here and on the front page said "best on record", which was wrong and is
corrected in the same edit.

## What the plateau inside the run cautions

Every arm so far has been 3 epochs. At 4M, ep2 = ep3 on accuracy, so the arm is limited by **steps**, not by data
volume - 15,625 steps x 3. The next honest question is therefore not "8M windows?" but "more epochs at 4M, or
larger batches?", and those are different experiments. Nothing here answers it.

## Not claimed

* One seed. The 4M arm's +0.62 pp is smaller than the 3.07 pp seed spread measured at 200k.
* Not a headline: 13.63 pp below 37.37%, and the d512 step-size confound (`D512-PROBE-RESULT.md`) is untouched.
* The gates were not measured at this rung; only the gated/gates-off comparison at 500k and 2M exists.
