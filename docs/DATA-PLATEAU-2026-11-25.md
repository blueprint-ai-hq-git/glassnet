# 2026-11-25 - the 200k pool flattens: total gain only 0.042 over 25,600 steps

Source: lens/runs/DATA-CEILING-LONG.out (clean run, exit 0), lens/runs/DATA-CEILING.json. Verified cupy path.

    pool 200,000      steps 25,600   38,400   51,200
    held-out          4.18790        4.15636  4.14550

    total change over 25,600 extra steps: -0.04240
    interval gains: 25,600->38,400 = 0.03154 ; 38,400->51,200 = 0.01086   (a 3x reduction)

## The honest reading

The curve is STILL falling but the gains are collapsing, and the total available from doubling the step budget again is
small. So: with 200,000 distinct windows, extra steps are worth about 0.01-0.03 each doubling. That is the same shape as the
20k pool flattening earlier, just at a higher level. Growing the pool moves the whole curve down (4.72662 -> 4.18625 at
6,400 steps in the controlled test); growing the step budget only slides along it.

## MY OWN VERDICT BLOCK WAS WRONG, and it printed FLATTENED on a falling curve

`flat = abs(last - c[best_mark]) <= 0.05` is always true when the best IS the last point, so the run declared FLATTENED
while the numbers show it still falling. That is the ELEVENTH instrument bug in this project. Fixed to compare the last
INTERVAL's gain against the first interval's, so a curve that is still paying cannot be labelled flat. The numbers above
were printed before the verdict line and are unaffected; only the label was wrong.

## What this changes

No training-budget claim will be made from this. The next question is not "more steps" but "does a bigger model make better
use of the same data", because if capacity binds next then the answer is a wider model on the same pool, not a longer run.
That test is queued: same pool, same steps, d=128 vs d=256.
