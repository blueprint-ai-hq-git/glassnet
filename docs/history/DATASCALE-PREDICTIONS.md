# DATASCALE-HONEST: is data scale the lever at this rung? (predictions, written before the run)

2026-11-06 19:1x IST. Written before `lens/scripts/run_datascale_honest.sh` is launched, in response to an
unscheduled 200k-window run at 19:05 that is not comparable to anything (`UNSCHEDULED-DATASCALE-RUN.md`).

## The question

`PLAN.txt` names DATA SCALE as the next lever once the concept-wall route is exhausted ("the only lever
that has ever moved this rung"). The wall block did close that route: the bypass was not worth 1-6 points,
it *cost* 1.4-3.4 pp on four seeds (`WALL-REP-RESULT.md`). What has never been measured on the honest mix is
whether 10x more of the same data moves this rung at all.

## Design - one thing changes

Identical to the wall A/B's `off` arm (`run_wall_ab.sh:113-117`), same corpus
(`data/chat/prepared_mix_ctrl_honest`, 294,225,090 cached tokens, 740,666 conversations), same
d 256 / window 32 / 3 epochs / batch 256 / lr 0.4 / lr-decay 0.5 / lam-concept 0.3 / lam-act 0.0 /
`--assistant-only --train-turn-end --gate-tol 0.05 --method ours --opt-lr 1.0`, no bypass, seed 0.

    ONLY CHANGE: --max-windows 20000 -> 200000   (10x)

Baseline to compare against, same config, same corpus, already measured: `wall_off` seed 0 at 20k windows,
**VAL 12.18% / ASST 13.00% / ppl 399.50** (`WALL-AB.txt:116`).

## Predictions and their falsifiers

D1  MAIN. 200k windows raises VAL acc by **+3.0 to +8.0 pp** over 12.18%, i.e. lands in **15.18% - 20.18%**.
    FALSIFIER A: gain <= +1.0 pp (VAL <= 13.18%). Then data scale is not the lever at this rung either, and
    the plan's stated branch is wrong - that would be the third plan assumption to fall today.
    FALSIFIER B: gain > +8.0 pp (VAL > 20.18%). Then the 20k rung was data-starved and every lever comparison
    made at 20k (attention, window 64, the bypass) was made in a regime that the project never intended to
    ship from.

D2  PPL. Final VAL ppl falls below **300** (baseline 399.50), consistent with D1.
    FALSIFIER: ppl >= 399.50 - more data did not even improve the loss, which would point at the gate or the
    optimiser rather than the data.

D3  NO SIGNIFICANCE CLAIM. One seed. The measured run-to-run spread at this rung is 1.03 pp
    (`REF-REPRO.md`), so a gain under ~1 pp is not separable from noise and will not be reported as one. With
    D1 predicting 3-8 pp, the expected signal is 3-8x the spread - that is why a single arm is worth running.

## What this run does not do

* It does not touch the wall arms, their logs, or their verdicts, and its number is not pooled with them:
  200k windows is a different experiment, not a bigger sample of the same one.
* It does not settle the P1 gate reading or the n-seed rule question, which are still the maintainer's to decide.
* It makes no "smart", talk-quality or ship claim. Next-token accuracy at 200k windows is all it measures.
* It costs ~65 min of a free GPU at ~400 MiB. Checkpoints are written every epoch by the trainer.
