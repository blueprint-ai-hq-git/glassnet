# WALL A/B arm 1: `best.npz` provenance repaired (2026-11-06 17:2x)

One file was wrong on disk and is now correct. Nothing was deleted; nothing was re-trained;
no trainer code was touched while the A/B is still executing.

## The defect (measured, not inferred)

`lens/runs/chat/wall_off_s0_best.npz` was written twice in one minute:

| time | writer | line in `lens/runs/WALL-AB.txt` | content |
|---|---|---|---|
| 17:07 | attempt 1, epoch 3 (the real best) | `:119` `NEW BEST held-out 13.00% at epoch 3 ... -> saved wall_off_s0_best.npz` | ep3 weights |
| 17:11 | attempt 2, epoch 1 (a redundant restart) | `:285` `NEW BEST held-out 8.07% at epoch 1 ... -> saved wall_off_s0_best.npz` | ep1 weights |

`train_chat.py` overwrites `<out>_best.npz` unconditionally at every new best, so the redundant
attempt-2 restart (started because `run_wall_ab_resume.sh`'s skip test looked for `gpu_wall_<arm>_`
while these runs save `wall_<arm>_`) replaced the 13.00% epoch-3 file with its own epoch-1 file
before it was stopped by hand at 17:10:39.

Proof it was ep1 and not ep3, by hash before the repair:

    714d15c6c861435f9a74eb9b685edb037468a2da2d7b828a11eef083b7fc60b0  wall_off_s0_best.npz
    714d15c6c861435f9a74eb9b685edb037468a2da2d7b828a11eef083b7fc60b0  wall_off_s0_ep1.npz
    40d4120a08b5e86a82498386165411754031fb29d80ae413b6058f859e854c16  wall_off_s0_ep3.npz

and by content: `numpy.array_equal` over all 6 arrays said best == ep1 was bit-identical, and
best carried `__meta__` with `direct_mode: off` (the arm-1 config, so it was not a foreign file).
ep3 differs from ep2 in at least one array and holds 3,811,534 parameters.

## The repair

1. The wrong file was archived, not removed:
   `lens/runs/archive/wall_ab_bestfix_2026-11-06/wall_off_s0_best.npz.OVERWRITTEN-714d15c6`
   sha256 `714d15c6c861435f9a74eb9b685edb037468a2da2d7b828a11eef083b7fc60b0` (verified after copy).
2. `wall_off_s0_best.npz` was restored from `wall_off_s0_ep3.npz` — which is what the trainer
   itself would have written at epoch 3, and what it did write at 17:07.
3. Verified after: `sha256sum` gives `40d4120a...` for both `best.npz` and `ep3.npz`, and
   `array_equal` over all arrays is True.

## Why this is the honest reading, not a re-grading

The arm's numbers do not change. They were always read from the log (`:116` epoch 3, `:201`
final: VAL ppl 399.50, VAL acc 12.18%, concepts 20/36) and from `ep3.npz`, never from
`best.npz`. The repair only makes the stored artifact agree with the log it came from, so a
later analysis that loads `best.npz` cannot silently pick up an 8.07% checkpoint and call it
this arm's best.

## Still open (code, deliberately not touched)

`train_chat.py`'s unconditional `best.npz` overwrite is the underlying defect. It is NOT fixed
here: `run_wall_ab_resume2.sh` restarts the trainer from that file for every remaining arm, and
the six arms must run byte-identical code (P6, `docs/history/WALL-AB-PREDICTIONS.md`). The fix
belongs immediately after the A/B finishes.
