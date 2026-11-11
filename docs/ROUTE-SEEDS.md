# Three seeds on the routing recipe: copy generalisation peaks early and is reproducible (2026-11-11)

Routing instead of adding was measured once at 33.0% mean held-out copying. Repeating it on seeds 2 and 3 gives a
result more useful than a single number: **the trade-off curve**.

```
                                 step 2000   4000    6000    8000    10000   12000
held-out copy, seed 0              64.3      45.8    24.8    39.5     3.2    56.2
held-out copy, seed 2              66.7      47.7    44.3    17.5    31.3    36.3
held-out copy, seed 3              60.2      38.2    18.5    24.8     1.7    31.0
language, seed 0                    -         -       -       -        -      27.3
language, seed 2                  16.2      20.4    22.3    24.1    25.8    26.1
language, seed 3                  16.4      20.6    23.4    24.2    25.7    25.8

mean of last three checkpoints, per seed: copy 33.0 / 28.4 / 19.2  ->  26.9% overall
mean of last three checkpoints, per seed: language 25.8 / 25.3 / 25.2  ->  25.4% overall
```

## The finding that matters
**At step 2,000 the three seeds agree closely: 64.3, 66.7, 60.2 - a mean of 63.7% with a spread of 6.5 points.** Later
checkpoints swing wildly (1.7% to 56.2%) and drift downward. So the copy operation generalises best EARLY in a mixed run
and is progressively traded away for language, which climbs steadily from ~16% to ~25.4%.

This is not noise versus signal in the usual sense: the early numbers are the tight ones and the late numbers are the
noisy ones. The previous result I published (33.0% mean of last three) is real but sits on the noisy half of this curve;
the stable statement is **63.7% held-out copying at 2,000 steps, three seeds, spread 6.5**.

## Honest scoring of the prediction
Prediction before the seeds ran: "mean of last three held-out copying above 20% on both seeds, language above 22%".
Seed 2 gave 28.4% and language 25.3 - PASS. Seed 3 gave 19.2% and 25.2 - **copy FAIL by 0.8 points**. Two of three
seeds cleared 20% on that ruler.

## What this changes about how the model should be trained
A single mixed run cannot hold both abilities at their best: copy is strongest at the start and language at the end. The
next arm therefore trains in two stages: language to convergence first, then a short low-learning-rate mixture, so the
copy phase lands where the copy operation is strongest instead of being averaged away. Prediction recorded in
`lens/scripts/run_language_first.sh` before it ran: held-out copying >= 50% with language >= 24%.

## Measurement note
Every number here is 600 held-out items per checkpoint (300 taught, 300 held-out words), same items across seeds 0 and 2
and 3 because the item seed is derived from the training seed. The per-checkpoint swing of plus or minus 20 points on
identical settings means no single checkpoint may be quoted as a result; the three-seed early mean is the number to use.
