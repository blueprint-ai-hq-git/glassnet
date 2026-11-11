# CORRECTION: the separate embedding table result, and a metric that swings too much to rank arms

**What I published and then had to correct (2026-11-11 18:25).** I reported that giving the copy pathway its own
embedding table was worse than sharing it, quoting 3.3% held-out copying at step 4,000 against the unbounded-gate arm's
14.8% at step 12,000. That comparison was 4,000 steps against 12,000 - a mid-run checkpoint against a finished one.
The finished separate-embedding arm reads:

```
step    language    taught copy   held-out copy
 6000     29.35%       94.7%           7.0%
 8000     29.62%       86.5%           9.3%
10000     30.23%       95.7%           7.5%
12000     30.15%       86.7%          15.7%
```

So its final held-out figure, **15.7%**, is the highest single reading of any arm today, higher than the unbounded
gate's **14.8%**, and its language figure, **30.15%**, is also the highest.

## But neither arm is actually better, and the reason matters more than the ranking
The held-out copy number swings between checkpoints on identical training settings: 7.0, 9.3, 7.5, 15.7 for one arm and
13.0, 16.5, 14.8 for the other. At 600 evaluation items, a few points of that is sampling noise, and the rest is the
optimiser wandering. Comparing single checkpoints ranks coin flips. On the last three checkpoints each:

```
arm                                    mean of last 3 held-out readings
unbounded gate                          14.8%
separate embedding                      10.8%
```

Mean of last three favours the gate; final checkpoint favours the separate table. **The honest conclusion is that these
two variants are indistinguishable at this sample size**, and the experiment as built cannot rank them. What IS
established is the direction: from 0.2% when the pointer is frozen, to 3.8% with a bounded gate, to 14.8-15.7% with an
unbounded gate, which is a large and repeatable effect.

## Rule adopted from this
No arm gets ranked on a single checkpoint again, and no mid-run number gets compared with a finished one. Every
comparison in this project now uses the mean of the last three evaluation checkpoints, and the number of evaluation
items is reported next to it so the noise floor is visible.
