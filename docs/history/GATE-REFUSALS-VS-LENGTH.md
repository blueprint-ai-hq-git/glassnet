# The agreement gate refuses more steps as an arm trains - not as the corpus grows

2026-11-07 12:0x IST, from the logs already on disk. No GPU used; the 2M arm was mid-epoch-1 throughout.

## The claim this note exists to block

Read straight off the logs, the numbers look like a regime change:

    six-arm 20k log, best arm:   refused 1 / 225 steps    = 0.4%
    200k arm, final:             refused 948 / 2325 steps = 40.8%
    500k arm, final:             refused 3907 / 5850 steps = 66.8%
    2M arm, in flight:           refused 3633 / 5600 steps = 64.9%

"At 20k the gate is off; above 200k it refuses most steps" would be a comfortable, publishable-sounding story - and
it is wrong. The counters are cumulative per run and the runs are different lengths: the 20k arms are **225-234
steps**, the 200k arm 2325, the 500k arm 5850, the 2M arm 23,438. Comparing final values compares a 234-step run
with a 5850-step run.

## The comparison that is fair: matched step counts

Refusal fraction for the agreement gate, at the same number of optimisation steps (the 20k arms never run past
~225 steps, so no cell beyond 200 is legitimate for them):

| step | 20k arms (seven segments, min-max) | 200k | 500k | 2M |
|---:|---:|---:|---:|---:|
| 100 | 0.0 - 16.0% | 12.0% | 12.0% | 8.0% |
| 150 | 0.0 - 18.7% | 14.7% | 12.0% | 10.0% |
| 200 | 0.0 - 19.5% | 14.0% | 11.5% | 10.0% |
| 1000 | not run | 22.1% | 21.3% | 31.7% |
| 2000 | not run | 35.9% | 36.6% | 42.1% |
| 4000 | not run | ended | 59.3% | 57.1% |
| 5850 | not run | ended | 66.8% | not yet run |

At every step where more than one family actually ran, the data-scaled arms sit **inside** the 20k arms' own
spread, and the 200k and 500k arms agree with each other to within 1 pp at steps 500, 1000 and 2000 (15.0/15.8, 22.1/21.3, 35.9/36.6). The refusal
fraction is a function of **how long the arm has trained**, almost identically across data caps.

## Two traps I fell into on the way, recorded because they are the house defect

1. **First pass reported "20k refuses 0.4%".** The parser kept the last line seen at each step across a log
   holding six arms, so it silently reported whichever arm printed last - the `on` arm (1/225), not the family.
   Re-parsed per arm segment: the 20k arms range **0.4% to 20.0%** (`off` seed 0 refuses 45/225).
2. **First pass also reported the 200k arm refusing 40.8% at step 4000 and 5000**, where that arm has no data -
   its run ends at step 2325 and the table carried the final value forward. Both artifacts produced a cleaner
   story than the truth, which is exactly why they were worth catching.

## What it does and does not imply

* **Does not** mean architecture results measured at 20k were taken with the gates disabled. At matched early
  steps the 20k arms span the same 0-20% band.
* **Does** mean the late-training refusal figures in the 200k/500k/2M logs must not be compared across arms, and
  that "the gate refuses two thirds of steps" describes the *end* of a 5850-step run, not the data scale.
* **Hypothesis, not a finding:** an arm that spends most of its late steps refused may convert extra data into
  less gain than a linear extrapolation suggests. That is consistent with widening the 2M band
  (`docs/history/DATASCALE-2M-PREDICTIONS.md`, T2: 21.5-24.0 while the naive slope implied ~25.9), but the
  500k arm still beat its predecessor, so the hypothesis is untested. It would need a gate-off control at 500k,
  which has not been run.

Source logs: `lens/runs/WALL-AB.txt` (seven segments: `off` s0 twice - one aborted at step 100 - `on` s0,
`r64` s0, `off` s1, `on` s1, `r64` s1), `DATASCALE-HONEST.txt`, `DATASCALE-500K.txt`, `DATASCALE-2M.txt`.
