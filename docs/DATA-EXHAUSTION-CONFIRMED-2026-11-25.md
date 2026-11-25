# 2026-11-25 - DATA EXHAUSTION CONFIRMED: same steps, only distinct data differs

Source: lens/runs/DATA-CEILING.out, lens/runs/DATA-CEILING.json. Backend: verified cupy path.

Design: identical model, seed, learning rate, step budget and held-out set (256 windows, never trained). The ONLY change
between arms is how many DISTINCT training windows exist. Each arm gets a fresh model and an epoch shuffle over its own pool.

    pool          steps 6,400   steps 12,800   steps 25,600   shape
      20,000        4.72662       5.71944        7.64552      RISES
     200,000        4.68625       4.32718        4.20032      falls

    P1 20k reproduces the rise        MET  (4.72662 -> 7.64552)
    P2 200k does not rise             MET  (4.68625 -> 4.20032)
    P3 200k beats 20k at 25,600 by .20 MET  (+3.44520)

## What this settles

The earlier long-curve collapse was not the optimiser, not the learning rate, and not a sampler bug. Ten times more distinct
data makes the SAME step budget keep improving instead of degrading. Distinct data volume is the binding constraint at this
scale, and this is now a controlled comparison rather than a correlation.

It also explains the project's history: long runs kept failing to improve because they were re-reading a small pool until the
model memorised it. Capacity, depth and step budget were all being blamed for a data problem.

## What it does NOT say

Nothing about language ability. This is an MLP language model at vocab 2,000 on a 64-token window; it is a controlled
instrument for the data/step question, not the chat model. And 4.20032 held-out loss is still far above what any talkable
model needs.

## Next, directly implied

Continue the 200k arm past 25,600 steps to find where it actually flattens (51,200 next, about 60 s on this backend). Only
then is a training budget claim honest.
