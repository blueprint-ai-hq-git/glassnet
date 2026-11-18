# 6000 steps: the plateau holds, and a silent no-op bug in the resume path (2026-11-18)

## The 6000-step verdict, measured at the corrected protocol

```
step  3,000   4,500   5,000   5,500   5,750   6,000      (104 chunks, SE 0.50)
acc   23.95%  25.35%  25.24%  25.76%  26.01%  25.89%
loss  6.3261  6.1303  6.1148  6.0826  6.0662  6.0640
```

- **Original P1** ("held-out >= 24% at 6000 steps") - **MET**: 25.89%.
- **Restated P1** ("held-out >= 26% at 6000 steps", restated after the eval correction moved this arm's baseline from 21.03% to 23.95%) - **MISSED by 0.11
  points**: 25.89% against 26.00%. Recorded as missed. A 0.11-point miss is 0.2 standard errors and therefore a tie in substance, but the gate was stated as a number and it was not
  reached, and redefining it after the fact is exactly what this project forbids.
- **Plateau band, steps 4500-6000: 25.24% to 26.01%**, a spread of **0.77 points** against a **0.50-point** standard error.
- **3000 -> 4500 gained +1.40 points (2.8 SE). 4500 -> 6000 gained +0.54 points (1.1 SE).** The returns fell from nearly three standard errors to barely one over the same
  1500-step span, and the second interval **includes zero**.

**Conclusion: at radius 0.9 with a 32-token window, this recipe plateaus at ~25.5-26% after roughly 4,500 steps (about 9M tokens) and the next 1,500 steps buy nothing resolvable.**
Note this is not a data-exhaustion plateau: one epoch of the corpus is 294M tokens and the model has seen 12.3M, about 4% of one pass. The limit is not how much text exists.

## The bug this wake found: a resumed arm could train nothing and report success

The trainer's `STEPS` was an **absolute end step**, and the loop was `range(START_STEP + 1, STEPS + 1)`. The learning rate divided by `STEPS - 1`:

```
for st in range(START_STEP + 1, STEPS + 1):
    lr = LR0 * (1.0 - 0.9 * (st - 1) / max(1, STEPS - 1))
```

The memory-channel arm queued an hour earlier resumes from **step 3000** with **STEPS=1500**. That builds `range(3001, 1501)` - **an empty loop**. The arm would have:

1. loaded the resumed weights,
2. trained **zero** steps,
3. written a checkpoint and printed held-out numbers **belonging entirely to the model it started from**,
4. and reported a `COMPLETE` line.

A silent no-op that produces plausible numbers is the worst failure mode this project has, and it was one queue slot away from running. With a larger `START_STEP` the
schedule would additionally have produced a **negative learning rate**.

**Fixed:** when `RESUME` is set, `STEPS` now means the number of **additional** steps, and both the loop end and the schedule are relative to the resume point. A fresh run is
numerically identical to before. Verified both directions:

```
fresh run   : steps=2 (from 0 to 2)          <- unchanged
resumed run : steps=1500 (from 3000 to 4500) <- was an empty loop
```

Plus a guard that **refuses to run** if the resume point is at or beyond the computed end, rather than quietly doing nothing.

## The pattern across this session

Three of the defects found today were not wrong maths but **wrong measurement or silent no-ops**: an eval too small to see the effect it was testing, a cross-implementation gate that
could never pass, and now a resumed arm that could report a model it never trained. The maths has been the reliable part.
