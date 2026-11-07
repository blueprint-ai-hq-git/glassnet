# DATASCALE-EP6: is the 200k rung truncated by its 3-epoch schedule? (predictions, before the run)

2026-11-07 08:4x IST, written before `lens/scripts/run_datascale_ep6.sh` is launched.

## Why this is the next question, and not a new topic

The 200k data-scale arms both beat their 20k baselines by a lot (+5.25 and +6.92 pp, mean +6.09 -
`docs/history/DATASCALE-SEED1-RESULT.md`). But their epoch paths disagree about the schedule:

    seed 0, 200k, 3 epochs:  VAL 17.48 -> 17.90 -> 17.43   ppl 232.63 -> 214.13 -> 207.29   (flattened)
    seed 1, 200k, 3 epochs:  VAL 16.24 -> 19.22 -> 19.71   ppl 239.50 -> 194.83 -> 184.32   (still rising)

Seed 1 gained +0.49 pp on its last epoch and its loss was still falling. That is the signature of a run that
was cut off, not of one that converged. If more steps help, the 200k rung is step-starved and every 3-epoch
number here understates what data scale buys. If they do not, the 3-epoch schedule is fine and the gain is
purely a data effect.

## Design - one knob, seed 0

Identical to `datascale_honest_s0` (`docs/history/DATASCALE-RESULT.md`): same corpus, same command, same
36-concept set, same 3,811,534 parameters, no bypass, 200,000 windows, same 87,035-window held-out set.

    ONLY CHANGE: --epochs 3 -> 6

Baseline for this exact arm: seed 0 at 3 epochs, VAL 17.43% / ASST 18.61% / ppl 207.29
(`lens/runs/DATASCALE-HONEST.txt:230`).

## Predictions and their falsifiers

E1  MAIN. 6 epochs lifts final VAL above **18.43%**, i.e. at least +1.0 pp over the 3-epoch value of 17.43%.
    FALSIFIER: final VAL <= 17.43%. Then the last three epochs bought nothing at seed 0, the 3-epoch schedule
    was not truncating, and no further schedule work at this rung is warranted.

E2  LOSS. Final ppl at epoch 6 is below **207.29** (the 3-epoch value).
    FALSIFIER: ppl >= 207.29. A flat or rising loss while accuracy is flat means the extra epochs overfit
    rather than fit.

E3  PARTIAL. If VAL lands inside 17.43-18.43 (a gain of 0-1 pp) the honest reading is "small benefit, not
    worth 2x the compute at this rung", and it is reported as that rather than rounded up to E1.

E4  NO POOLING. A 6-epoch arm is not the same experiment as the 3-epoch arms. Its number is reported on its
    own and never averaged into the +5.25 / +6.92 replication, which stands as measured.

## What will not be claimed

* No "smart", no talk quality, no ship. Next-token accuracy at 200k windows is all this measures.
* No headline change: 37.37% (d=512, full corpus, 6 epochs) remains the best number.
* One seed. This decides a schedule question at seed 0 only; if E1 passes, a seed-1 confirmation is a
  separate, later decision.
* Cost: ~40 min of a free GPU at ~400 MiB, checkpoints written every epoch by the trainer.
