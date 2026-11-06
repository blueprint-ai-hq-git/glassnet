# WALL-AB verdict: all six arms, graded as pre-registered

Written 2026-11-06 17:5x IST, after the runner wrote its own completion marker. Nothing here is a new
threshold: P1-P5 are the predictions written before any arm ran (`docs/history/WALL-AB-PREDICTIONS.md`),
graded by `lens/scripts/wall_ab_grade.py`. Every number below is copied from the log or from the
grader's own output; no number is estimated.

## What finished

    lens/runs/WALL-AB.txt line 1226:  ===== WALL-AB DONE 2026-11-06 17:51 arms_missing=0

`lens/runs/WALL-AB-RESULTS.tsv` (sha256 d1e4f0af...) has 6 rows, one per arm, each with the log line it
came from. `lens/runs/WALL-AB-GRADE.txt` (sha256 ac55e35b...) is the grader's output on those 6 rows.

## The six arms (final VAL at epoch 3, d 256, window 32, 20k windows, batch 256)

| arm | seed | final ppl | val acc | concepts | train acc | gap | log |
|---|---|---:|---:|---:|---:|---:|---|
| wall_off (no bypass) | 0 | 399.50 | **12.18%** | 20/36 | 19.43% | +7.25 | :116, final :201 |
| wall_on (tied full-rank, +2,097,152 params) | 0 | 575.00 | 10.79% | 20/36 | 39.03% | +28.24 | :392, final :477 |
| wall_r64 (rank 64, +1,292,416 params) | 0 | 550.85 | 11.28% | 24/36 | 39.23% | +27.95 | :580, final :664 |
| wall_off (no bypass) | 1 | 395.80 | **12.79%** | 26/36 | 22.23% | +9.44 | :765, final :850 |
| wall_on (tied full-rank) | 1 | 628.41 | 9.68% | 22/36 | 39.39% | +29.71 | :952, final :1036 |
| wall_r64 (rank 64) | 1 | 521.68 | 10.87% | 26/36 | 38.98% | +28.11 | :1138, final :1222 |

Stored 20k-window reference for the same config: 11.96 / 10.08 (mean 11.02%), `lens/runs/CTX-ATTN.txt`.

## The verdicts, in the grader's words

P1 CONTROL SANITY - band 10.0-12.5% final val acc for `wall_on`.

    on s0 10.79%  (inside)
    on s1  9.68%  (0.32 pp BELOW the floor)
    -> FAIL

This is the one that constrains what may be said. As coded, the control arm did not land in the band:
on s1 9.68% is 0.32 pp below the floor. **But the band itself was built from the wrong column** - 11.96 and
10.08 are the `ASST(by-position)` values of `lc_w20000` at `LEARNCURVE.txt:4773` and `:4960`, while the
grader reads the `VAL` column (the same two runs scored VAL 11.32 and 9.53, and the file the predictions
cite, CTX-ATTN.txt, contains neither number). Compared on the same column the control reproduces within
0.55 pp on both seeds. Full evidence, both columns, both commands and the five matching config-header
lines: `docs/history/WALL-AB-P1-METRIC.md`. So: the coded verdict is FAIL and no delta against the stored
rung is quoted, but that FAIL is a metric mismatch, not a drifted rung.

P2 MAIN - predicted: `wall_off` LOWER than `wall_on` by 1.0-6.0 points.

    s0: off 12.18% - on 10.79% = +1.39
    s1: off 12.79% - on  9.68% = +3.11
    -> OPPOSITE SIGN: the predicted loss is NOT reproduced. Removing the bypass did not cost the
       concept path 1-6 points; on both seeds it was 1.39 and 3.11 points AHEAD.

P3 CAPACITY DOSE - predicted: rank 64 within 1.0 pp of the full-rank mix.

    s0: r64 11.28% - on 10.79% = +0.49  (inside 1.0)
    s1: r64 10.87% - on  9.68% = +1.19  (outside 1.0)
    -> MIXED: not below `on` on either seed, so "rank 64 is under-provisioned" is not supported, but the
       dose reading (rank 64 == full-rank mix) is not clean at seed 1 either.

P4 OVERFIT DIRECTION - predicted: `wall_off` has the smallest train-minus-val gap.

    off gaps +7.25 / +9.44   mean +8.35
    on  gaps +28.24 / +29.71 mean +28.98
    r64 gaps +27.95 / +28.11 mean +28.03
    -> PASS: smallest mean gap is `off`, by 20.63 pp against `on`.

P5 FALSIFIER AGAINST MY OWN EXPECTATION - `wall_off` ABOVE `wall_on` by more than 1.0 pp on both seeds.

    off - on: s0 +1.39  s1 +3.11
    -> CONTRADICTION CONFIRMED: on both seeds the no-bypass arm is ahead; reported as a finding that
       contradicts the plan, not spun as noise.

P6 SUPERVISION ONLY (resume addendum) - falsifier: any arm's recorded command line differs from
`run_wall_ab.sh`'s. Checked by reading both scripts' trainer invocations and the seven `----- <arm> seed
<S> :` markers in the log: the flags are field-for-field identical (`off` none, `on` `--direct
--direct-mode tied`, `r64` `--direct --direct-mode lowrank --direct-rank 64`); the only difference is
shell quoting of the seed (`$SEED` vs `"$SEED"`), same values. -> not triggered; the resume is valid.

## What this does and does not settle

* It does NOT show "the concept wall is lossy". The reverse direction appeared on both seeds.
* It does NOT licence the wider-residual build: P2's premise (the bypass is worth 1-6 points) did not
  reproduce; the bypass arm is 2.1M params heavier and 1.39-3.11 pp worse on both seeds, and it was also
  the most overfit (gap +28.98 vs +8.35 mean).
* It does NOT change the project's headline number: best held-out next-token accuracy is still 37.37%
  (d512, 6 epochs), untouched by this block.
* It DOES say the P4 reading of this project's large train-minus-val gap is capacity-driven: the arm with
  2.1M fewer parameters has a 20.63 pp smaller gap at the same rung.
* It does NOT separate "channel access" from "parameter count" - that was the point of the 0 / 1.29M /
  2.10M dose and the dose came back mixed at seed 1.

Because P1 failed, the honest next step is a decision, not an automatic launch: either re-run the control
until `wall_on` reproduces the 10.0-12.5 band at this rung, or accept the block as a within-block pair
and move to the lever the plan already names (data scale, the only lever that has moved this rung).

## Grade-script defect found and fixed in this run

`lens/scripts/wall_ab_grade.py` P2 computed `on - off` but graded the pre-registered band, which is
`off - on` (-1.0..-6.0). Two consequences, both fixed at 17:47:28:

* the falsifier `on - off < 0.5` was true for every negative delta, so a real 1.39/3.11 pp LOSS was
  reported as "bypass earns under 0.5 pp" - right numbers, wrong sentence;
* the band test was only reachable when every `on - off` was positive, so a perfect result would still
  have printed "magnitude OUTSIDE the band".

The fix computes `off - on`, keeps the no-effect test as `|off - on| < 0.5`, and prints both differences
on every line. The decision it implies is unchanged (do not build the wider residual), only the reason is
now the true one. Written up separately in `docs/history/GRADER-P2-SIGN.md`.
