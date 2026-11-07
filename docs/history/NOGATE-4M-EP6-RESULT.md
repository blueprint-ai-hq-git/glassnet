# Six epochs at 4M: +0.18 pp. Doubling the steps is worth a third of doubling the data

2026-11-07 18:07:39 IST, `lens/runs/DATASCALE-4M-NOGATE-EP6.txt:73`, `exit=0`, marker
`NOGATE-4M-EP6 DONE 2026-11-07 18:07`. Predictions written before the launch in
`docs/history/NOGATE-4M-EP6-PREDICTIONS.md`; graded by `datascale_ladder_grade.py nogate4m_ep6`.

## Result

    VAL 23.92%   ASST 25.92%   ppl 93.72   train 36.08%   concepts 36/36
    path: 23.11 (106.06) -> 23.71 (97.29) -> 23.82 (95.89) -> 23.51 (95.36) -> 23.73 (94.31) -> 23.92 (93.72)
    best checkpoint = epoch 6; 481-486 s per epoch; 93,750 updates; 3,811,534 params

Provenance: `verify_wall_ab_ckpt.py` -> 1 arm, 0 problems. Mechanical: 0 agreement-step lines, 0 gate-step lines,
`METHOD: OURS STEP RULE, GATES OFF`.

## The verdicts

    W1  best VAL >= 24.04%           23.92%  -> FAIL, 0.12 pp short
    W2  best in 24.0-25.5%           23.92%  -> OUTSIDE (low)
    W3  ppl < 95.99                  93.72   -> PASS
    W4  mechanical                   0 / 0   -> PASS
    W5  concepts >= 36               36/36   -> PASS

**The hypothesis this arm was built to test was falsified.** I wrote that epochs 2 and 3 being equal meant the arm
was step-limited, and that doubling the updates should therefore buy at least 0.3 pp. It bought **0.18 pp**. The
arm is not step-limited, or at least not mostly.

## The comparison that matters, at one rung (4M, seed 0, gates off)

    doubling the DATA   2M -> 4M          23.12 -> 23.74   +0.62 pp
    doubling the STEPS  46,875 -> 93,750  23.74 -> 23.92   +0.18 pp

Data is worth **3.4x** what steps are worth at this margin, and neither closes the gap to the project's d256 record:
the full-corpus d256 run reaches **35.72% by position** (`docs/history/STATE.md:146`), **11.80 pp** above this arm.

Accuracy also wobbles across epochs (23.82, 23.51, 23.73, 23.92) while perplexity falls monotonically (95.89 ->
93.72) - the same pattern as every arm today. The model keeps getting more confident on the same answers; the
accuracy metric has stopped moving with it.

## What this rules in and out for the next arm

* **Out:** more epochs at 4M. Doubling them bought 0.18 pp and needed 49 minutes; a further doubling is unlikely to
  pay for itself, and W1's own falsifier said so in advance.
* **In:** the data axis is still the best value per GPU-minute (+0.62 pp for 2x data at ~25 min), and 8M windows is
  now the natural next rung.
* **Unmeasured, and needed:** the 11.80 pp gap to the archive's d256 number. Those runs used 16.9M windows *and* 6
  epochs, and may differ in harness details; nothing here says the gap is closed by data alone.

## Not claimed

* One seed. The epoch-to-epoch wobble is ±0.3 pp and the deltas here are 0.18 and 0.62, so only the *ratio* is
  worth reading, not either number's size.
* Not the project's d256 record; not a headline. 37.37% (d512, full corpus, 6 epochs) is 13.45 pp above this arm.
* The gates were not tested at this rung (measured only at 500k and 2M).
