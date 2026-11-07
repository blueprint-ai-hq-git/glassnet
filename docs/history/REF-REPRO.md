# The 20k reference does not reproduce: three runs of the same command span 1.03 pp

2026-11-06 18:15 IST. One arm, seed 0, launched by the chain (`lens/scripts/run_ref_repro.sh`, label
`ref_repro`), log `lens/runs/REF-REPRO.txt` + `lens/runs/REF-REPRO-run.txt`, tag `ref_repro_s0`,
`DONE 2026-11-06 18:15:22 rc=0`. Command identical to `run_learncurve.sh:19-22` for the 20k arm (only
`--out` differs) and identical to the wall A/B's `on` arm.

## Predictions written before the run, and what happened

    P1  reproducible -> VAL 11.2-11.4, ASST 11.96 +/- 0.1
    P2  not reproducible -> VAL 10.8 +/- 0.1
    got VAL 10.29, ASST 10.92

Neither band was hit. The run landed 0.50 pp BELOW `wall_on` and 1.03 pp below the stored reference, so
this is neither "the reference reproduces" nor "the reference is the anomaly". P1 and P2 are both FAILED
as written and are recorded that way.

## The three draws of one configuration

| run | log | VAL acc | ASST acc |
|---|---|---:|---:|
| `lc_w20000_s0` (stored reference) | `LEARNCURVE.txt:4773` | 11.32 | 11.96 |
| `wall_on` seed 0 (A/B control) | `WALL-AB.txt:392` | 10.79 | 11.42 |
| `ref_repro_s0` (this run) | `REF-REPRO-run.txt`, ep 3/3 | 10.29 | 10.92 |

    VAL  spread (max - min) = 1.03 pp
    ASST spread (max - min) = 1.04 pp

Same command, same seed, same data, same config header on all three
(`assistant-only 213,046/300,000`, `chat_concepts_wide 43 defined`, `36 kept 7 dropped`,
`train-turn-end 173`, `windows 20,000 train / 87,035 val`, `vocab 12,002`, `window 32`).

## What this corrects

`STATUS.md` and `docs/history/D512-REPRODUCIBILITY.md` carry "run-to-run spread 0.01-0.02 pp". That figure
was measured on the d512 / 2M-window path and it does NOT transfer to this 20k-window rung: here the same
command spans 1.03 pp. Anything that used the small figure to argue "differences under 0.5 pp are noise,
differences over 1 pp are real" at this rung has to be re-argued.

## What it does to the wall A/B

* `off - on` on VAL acc is +1.39 (seed 0) and +3.11 (seed 1), against a measured spread of 1.03 pp: 1.3x and
  3.0x the spread, on one seed each. The direction is the same on both seeds, but with one seed per arm
  this is not a significance claim and must not be written as one.
* `wall_on`'s own two seeds differ by 1.11 pp (10.79 vs 9.68 on VAL), which is the same size as the
  three-run spread above - i.e. the seed-to-seed and run-to-run variation at this rung are the same order.
* P1's band (10.0-12.5) is +/-1.25 around a single draw (11.02). If the distribution is ~1 pp wide, that
  band is a coin flip for any single arm, so P1 cannot function as a control at n=1 seed. That is a rules
  question - what the gate should require - and it belongs to the maintainer, not to this worker.

## What was not done

Nothing was averaged, back-filled or re-run to make a band fit. The failed predictions are recorded as
failed, the new number stands as a third draw, and the A/B's arms were not touched.
