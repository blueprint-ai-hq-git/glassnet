# WALL-AB: does anything get through the concept wall? (predictions, written before any arm ran)

Written 2026-11-06 16:2x IST by the autonomous wake turn, BEFORE the script `lens/scripts/run_wall_ab.sh`
existed in runnable form and before any of the six arms started. Source of the question:
`docs/history/PROBLEMS.md` OPEN #1, "Concept-wall bottleneck test (the biggest open lever for 'smart')".

## The question, in one sentence

The answer layer can only see the concept layer, so any signal the concepts cannot express is lost.
`--direct` is the bypass that was added to fix exactly that (its comment in `lens/model.py:154-163` cites
concepts-only 5.9% vs position-aware 77.6% on a toy identity task), and it is ON in every recent arm -
but nobody has ever measured what removing it costs at the current 20k-window rung. Without that number
the "concept wall is lossy, build a wider residual" plan is an assumption, not a measurement.

## Design (one change per arm, all else matched)

Config of every arm, identical: d 256, window 32, 3 epochs, batch 256, lr 0.4, lr-decay 0.5,
lam-concept 0.3, lam-act 0.0, `--assistant-only --train-turn-end`, `--max-windows 20000`,
`--gate-tol 0.05 --method ours --opt-lr 1.0`, seeds 0 and 1. This is field-for-field the config of the
stored 20k reference in `lens/runs/CTX-ATTN.txt` (final val 11.96/10.08, mean 11.02%).

  arm 1 `wall_off` : concepts only, NO bypass                                  seed 0, 1
  arm 2 `wall_on`  : `--direct --direct-mode tied`  (full-rank per-position mix, +2,097,152 params)   seed 0, 1
  arm 3 `wall_r64` : `--direct --direct-mode lowrank --direct-rank 64` (+1,292,416 params)            seed 0, 1

Three points on a capacity axis (0 / 1.29M / 2.10M extra parameters) rather than two, because a two-arm
test cannot tell "the channel was missing" from "the arm had 2.1M more knobs".
The reference is re-run inside this block (arm 2) instead of being compared across jobs: the measured
run-to-run spread on this GPU path is 0.01-0.02 pp (`docs/history/D512-REPRODUCIBILITY.md`), so a
same-block control is the only clean comparison.

## Predictions (P1-P5), and the number that falsifies each

P1  CONTROL SANITY. `wall_on` s0/s1 land within 10.0-12.5% final val acc, i.e. reproduce the stored
    reference band. If not, the block is not comparable to the stored rung and no delta is quoted.

P2  MAIN. `wall_off` is LOWER than `wall_on` on BOTH seeds. Predicted delta: -1.0 to -6.0 points final
    val acc. FALSIFIER: if `wall_on - wall_off` is smaller than 0.5 pp on both seeds, the bypass earns
    nothing at this rung, the "concept wall is lossy" premise is NOT established, and the wider-residual
    build in PROBLEMS.md #1 is cancelled rather than run.

P3  CAPACITY DOSE. `wall_r64` lands within 1.0 pp of `wall_on` on both seeds. If the 0.8M-cheaper
    rank-64 residual matches the full-rank mix, then more bypass capacity is not the lever and any future
    residual work should buy rank, not width. FALSIFIER: `wall_r64` more than 1.0 pp BELOW `wall_on` on
    both seeds means bypass capacity itself is the lever, and rank 64 is under-provisioned.

P4  OVERFIT DIRECTION. `wall_off` shows the SMALLEST train-minus-val gap at epoch 3 of the three arms
    (fewest parameters). If instead `wall_off` has the largest gap, the gap is not driven by capacity and
    my capacity reading of this project's +26-point gap is wrong.

P5  FALSIFIER AGAINST MY OWN EXPECTATION. If `wall_off` is ABOVE `wall_on` by more than 1.0 pp on both
    seeds, the bypass is actively harming the concept path (logit dilution) and that is reported as a
    finding that contradicts the plan, not spun as noise.

## What will NOT be claimed from this

* No talk number, no ship, no "smart". Next-token accuracy at 20k windows is the only measured quantity.
* No separation of "channel access" from "parameter count" beyond the three-point dose above - the arms
  differ in both, deliberately, and the dose is the evidence, not an ablation of one factor.
* No extrapolation to 2M windows or to d512.

## RESUME ADDENDUM - 2026-11-06 17:0x (added before the resumed launch; nothing above edited)

Status when this was written: of the six arms, ONE was actually training (off seed 0, orphaned trainer pid
3015702, launched 17:01:13). Three launchers of run_wall_ab.sh had each been SIGTERM'd with the agent turn
that started it (chain rc=143, three separate records in lens/runs/WALL-AB.txt), and each death took the
loop that owns the remaining arms with it while leaving the trainer alive. So no result existed yet; the
"three aborts" in the log are launcher deaths, not training failures.

The resumed run (lens/scripts/run_wall_ab_resume.sh) is the SAME experiment: same six arms, same seeds,
byte-identical trainer commands, same log, same completion markers. P1-P5 above stand unedited and are
graded exactly as written. One extra prediction, made before launching, about what I expect to see:

P6  SUPERVISION ONLY. The resumed run changes no arm and no hyperparameter, so the six numbers should be
    indistinguishable from what run_wall_ab.sh would have produced had its launcher lived. The ONLY
    predicted difference is that off/seed 0 arrives first (it was already ~3 min in when the supervisor
    attached) and the other five follow at ~7-9 min each. FALSIFIER: if any arm's recorded command line in
    the log differs from run_wall_ab.sh's, the resume is invalid and its numbers must not be pooled with
    the original three-arm design.

Cost note: seeds are per-arm, so resuming cannot inflate significance; there is no repeated seed here.

## CITATION CORRECTION - 2026-11-06 18:4x (provenance only; nothing above is edited)

The design section names the stored reference as `lens/runs/CTX-ATTN.txt` (final val 11.96/10.08, mean
11.02%). **That file is wrong, and so is the column.** Re-verified this run:

    grep -cE "11\.96|10\.08" lens/runs/CTX-ATTN.txt   -> 0

The two numbers are the `ASST(by-position)` values at epoch 3 of the 20k arms in `lens/runs/LEARNCURVE.txt`:

    :4773  lc_w20000_s0  VAL ppl 552.99 acc 11.32%  ...  ASST(by-position) acc 11.96%
    :4960  lc_w20000_s1  VAL ppl 683.12 acc  9.53%  ...  ASST(by-position) acc 10.08%

so the same two runs' `VAL acc` values are 11.32 and 9.53, and it is `VAL acc` that `wall_ab_grade.py`
reads. P1's band was therefore derived from one column and applied to another; the numbers, both columns
and both commands are in `docs/history/WALL-AB-P1-METRIC.md`.

This addendum changes NO prediction, NO threshold and NO verdict: P1-P5 above stand exactly as written and
the coded P1 verdict stays FAIL. It is here so that the next reader does not go looking for 11.96 in a file
that does not contain it, and so nobody re-derives the band from the wrong column a second time.

