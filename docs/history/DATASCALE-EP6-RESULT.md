# DATASCALE-EP6: doubling the epochs at 200k bought exactly the minimum, and the schedule is unstable

2026-11-07 09:14:27 IST. One arm, seed 0, 6 epochs instead of 3, everything else identical
(`docs/history/DATASCALE-EP6-PREDICTIONS.md` written before the run). Runner marker
`DATASCALE-EP6 DONE 2026-11-07 09:14`; log `lens/runs/DATASCALE-EP6.txt`. Grader:
`lens/scripts/datascale_ep6_grade.py`, which re-verifies the 3-epoch baseline lines on every run.

## The question and the answer

    seed 0, 3 epochs:  VAL 17.48 -> 17.90 -> 17.43            (flattened)
    seed 1, 3 epochs:  VAL 16.24 -> 19.22 -> 19.71            (still rising)
    seed 0, 6 epochs:  VAL 17.48 -> 17.90 -> 17.43 -> 15.13 -> 19.68 -> 18.43

The first three epochs of the 6-epoch run reproduce the 3-epoch run **exactly** (17.48/232.63, 17.90/214.13,
17.43/207.29), so the arms differ only in where they stop - the extra epochs do not perturb the trajectory
before them. What they add after epoch 3 is not a climb but a swing: 15.13, then 19.68, then 18.43.

## The verdicts

    E1  final VAL above 18.43%     final VAL 18.43%   -> PARTIAL (E3), exactly at the threshold, not above it
    E2  final ppl below 207.29     ppl 201.69         -> PASS
    E4  never pooled with the 3-epoch arms             -> not pooled, reported alone

By the prediction's own wording E1 needed *above* 18.43 and the run landed *on* it: the gain over the
3-epoch arm is **+1.00 pp** and **-5.60 ppl**. E3 was written exactly for this case - "small benefit, not
worth 2x the compute at this rung; not rounded up to E1" - so it is recorded as PARTIAL.

## What the endpoint hides, stated as context and not as a re-grade

The run's best checkpoint is epoch 5, not epoch 6: ASST **20.83%** at ep5 (NEW BEST line), against the
3-epoch arm's best of **18.87%** at its epoch 2. Final-epoch VAL for ep6 is 18.43%, so a reader who looks only
at the endpoint sees +1.00 pp, while the best saved weights are +1.96 pp better on the assistant metric.

Both are true and neither re-grades E1: the pre-registered metric was the final epoch. What the gap between
them says is that **this rung is unstable epoch-to-epoch** (a 4.55 pp swing between ep4 and ep5), so a single
final-epoch number carries more variance than the 3-epoch reads suggested. That is worth knowing before
anyone compares future arms by endpoint alone.

## Checkpoint provenance

`verify_wall_ab_ckpt.py lens/runs/DATASCALE-EP6.txt` -> **1 arm verified, 0 provenance problems**:
`best.npz` is the epoch the log named (ep 5), sha and arrays match, meta says seed 0 / no bypass, and the
parameter count off the stored arrays is 3,811,534 - the same architecture as every other arm in this family.

## What this settles

* Doubling the schedule at 200k is **not** worth 2x the compute: it buys +1.00 pp of final VAL, and most of
  what it adds is a better *best* checkpoint rather than a better endpoint.
* The 3-epoch data-scale result stands unchanged (+5.25 pp seed 0, +6.92 pp seed 1, mean +6.09), and its
  numbers are not averaged with this arm.
* If the project wants more from this rung, the evidence points at **more seeds or wider models**, not at a
  longer schedule. That is a decision for Yahya, not a default.
