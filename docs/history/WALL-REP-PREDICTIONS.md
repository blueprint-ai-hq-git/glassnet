# WALL-REP: does the bypass result replicate on fresh seeds? (predictions, written before any arm ran)

Written 2026-11-06 18:2x IST by the chain turn woken by label `wall_ab_resume2` (a stale wake: that label
finished at 17:51 and its six arms were already graded at 17:51:32). This document exists BEFORE
`lens/scripts/run_wall_rep.sh` is run, and it grades nothing that was already graded.

## Why a new block instead of reading the old one again

The A/B's six arms are done and their verdict is in `lens/runs/WALL-AB-GRADE.txt`. Two things now stand in
the way of treating `off - on` as a number:

1. The control gate as coded FAILs (band built from ASST values, applied to VAL - `docs/history/WALL-AB-P1-METRIC.md`),
   and a gate must not be re-read after the result.
2. The rung's own noise is larger than anyone claimed: three runs of the SAME command and seed give VAL
   11.32 / 10.79 / 10.29, a 1.03 pp span (`docs/history/REF-REPRO.md`). The `off - on` deltas are +1.39
   (seed 0) and +3.11 (seed 1) - 1.3x and 3.0x that span, on one seed each.

One seed per arm cannot separate a 1.4 pp effect from a 1.0 pp noise floor. The cheapest honest move is
two INDEPENDENT seeds of the two arms that matter, run now, with the direction stated first.

## Design (nothing else changes)

Same trainer command as `run_wall_ab.sh:113-117`, same data, same d 256 / window 32 / 3 epochs / batch 256 /
lr 0.4 / lr-decay 0.5 / lam-concept 0.3 / lam-act 0.0 / 20,000 windows / gate-tol 0.05 / method ours /
opt-lr 1.0 / `--assistant-only --train-turn-end`:

    arm `wall_rep_off` : no bypass                       seeds 2 and 3
    arm `wall_rep_on`  : `--direct --direct-mode tied`   seeds 2 and 3

Four arms, ~8 min each, ~400 MiB VRAM, `nice 19` + `ionice -c3` + `taskset -c 0,1`. New log
`lens/runs/WALL-REP.txt`, new tags, so nothing here can touch `lens/runs/WALL-AB.txt` or its arms.

## Predictions, and what falsifies each

R1  DIRECTION REPLICATES. `off` is ABOVE `on` on both new seeds, i.e. `off - on > 0` at seeds 2 and 3.
    FALSIFIER: `off - on <= 0` on either seed. Then the A/B's direction is seed-specific and the honest
    statement becomes "1 of 2 seed pairs agree", not "the bypass hurts".

R2  MAGNITUDE. Both new deltas land between +1.0 and +4.0 pp, the range the two existing deltas define
    (+1.39, +3.11). FALSIFIER: a delta outside that range on either seed - the effect is not a
    constant of this rung and no single number describes it.

R3  SPREAD. Each arm's four values (2 old + 2 new) span no more than 1.5 pp on VAL acc, i.e. the 1.03 pp
    span seen for one configuration is the size of the noise, not a one-off. FALSIFIER: a span above
    1.5 pp - then even four draws do not bound this rung's noise and no delta from it may be quoted at all.

R4  NOT POOLED. This block is reported on its own. It does NOT re-open P1, does NOT change any threshold in
    `WALL-AB-PREDICTIONS.md`, and its arms are not added to the six-arm table as if they were part of that
    registered design. FALSIFIER: any commit that mixes the two tables without labelling which block each
    row came from.

## What will NOT be claimed

* No "the concept wall is lossy" and no "the bypass is harmful" as a settled finding - four arms on two
  new seeds either strengthen a lead or kill it.
* No talk number, no "smart", no ship.
* No significance test invented after the fact: the report is the four numbers, their spread, and whether R1
  and R2 held, written before the runs.
