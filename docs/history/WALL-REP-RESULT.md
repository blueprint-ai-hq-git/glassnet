# WALL-REP result: the bypass direction replicated on two fresh seeds, and the noise bound missed by 0.03 pp

2026-11-06 18:56 IST. Four arms, seeds 2 and 3, block launched 18:31:30 as `glassnet-wallrep.service`.
Runner marker: `WALL-REP DONE 2026-11-06 18:56 arms_done=4 of 4` (`lens/runs/WALL-REP.txt`). Predictions
R1-R4 were written before any arm ran (`docs/history/WALL-REP-PREDICTIONS.md`); this file reports them, it
does not re-write them. Grader: `lens/scripts/wall_rep_grade.py` (every number below is a log line).

## The four arms

| arm | seed | VAL acc | ASST acc | ppl | concepts | log |
|---|---|---:|---:|---:|---:|---|
| wall_rep_off | 2 | **13.40%** | 13.83% | 384.39 | 22/37 | `WALL-REP.txt:329` |
| wall_rep_on | 2 | 9.98% | 10.65% | 654.01 | 23/37 | `WALL-REP.txt:110` |
| wall_rep_off | 3 | **13.26%** | 13.72% | 378.53 | 19/37 | `WALL-REP.txt:517` |
| wall_rep_on | 3 | 11.21% | 11.99% | 628.40 | 21/37 | `WALL-REP.txt:705` |

Checkpoint provenance, checked in this run with `lens/scripts/verify_wall_ab_ckpt.py lens/runs/WALL-REP.txt`:
**4 arms verified, 0 provenance problems** - each `best.npz` is the epoch the log named (on s2 ep2, the
other three ep3), sha and arrays match, and the `__meta__` seed/mode inside each file agrees with its arm.
The same script on the six-arm log still reports 6 arms / 0 problems (no regression from this run's edit).

## The verdicts, as the grader printed them

    R1 DIRECTION  s2: off 13.40 - on  9.98 = +3.42      PASS
                  s3: off 13.26 - on 11.21 = +2.05      (direction replicates on both new seeds)
    R2 MAGNITUDE  s2 +3.42, s3 +2.05, both inside 1.0..4.0    PASS
    R3 SPREAD     off draws 12.18, 12.79, 13.40, 13.26 -> span 1.22 pp   <= 1.5  ok
                  on  draws 10.79,  9.68,  9.98, 11.21 -> span 1.53 pp   >  1.5  FAIL
    R4 NOT POOLED six-arm table has 6 rows, 0 of them from this block    PASS

R3 fails by **0.03 pp** (1.53 against a pre-registered bound of 1.5). That is a real miss, not a rounding
artefact, and its own falsifier is explicit: *"a span above 1.5 pp - then even four draws do not bound this
rung's noise and no delta from it may be quoted at all."*

## What the four seeds say, stated without moving the goalposts

    seed 0 (A/B):      off-on = +1.39 pp
    seed 1 (A/B):      off-on = +3.11 pp
    seed 2 (this):     off-on = +3.42 pp
    seed 3 (this):     off-on = +2.05 pp

Four seeds, four positive deltas, range +1.39 to +3.42, while the per-arm spread over four draws is 1.22 pp
(off) and 1.53 pp (on). The direction is consistent and the prediction's opposite sign has now been seen
four times.

**And yet the pre-registered rule still forbids quoting a delta**, because R3 - the noise bound written
before these runs - was not met. This is the second, independent reason the wall A/B's deltas stay unquoted:
P1's band mixes two columns (`WALL-AB-P1-METRIC.md`), and the rung's own noise (1.53 pp on the `on` arm over
four draws) is the same size as the smallest effect (1.39 pp at seed 0). Both are recorded rather than
argued away.

## What would settle it, and whose call that is

The honest next step is not another pair of seeds - four already exist - but a decision on what the gate
should be, given a measured per-seed effect of 1.4-3.4 pp against a measured per-arm spread of 1.2-1.5 pp:
either a pre-registered n-seed rule with a stated statistic, or accepting that this rung can only support
"the no-bypass arm was ahead on all four seeds" and nothing more precise. That is Yahya's call; this block
does not decide it, and no commit from this run writes a delta as a result.
