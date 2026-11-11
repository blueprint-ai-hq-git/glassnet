# Two-stage training fails in all three forms (2026-11-11)

Idea: teach language to convergence first, then add a short copy stage, so the copy phase lands where copying is
strongest. Measured three ways. **All three failed.**

```
variant                                     stage 1 language   stage 2 taught   stage 2 held-out copy
unfrozen pointer                                 26.82%            84.5%                6.8%
frozen pointer matrices                          17.02%            84.0%                3.2%
frozen matrices + separate frozen embedding      18.40%            83.2%                2.5%
```

Predictions on record before each run were held-out copy >= 40-50% with language >= 24%. Worst miss was 37 points.

## What the three failures have in common
Taught-word copying always recovers to 83-85%: the model can re-fit the specific words in a few hundred steps. Held-out
copying, the part that generalises to words never seen, does not come back at all. And freezing made language *worse*
(17.0% and 18.4% against 26.8%), so the protection cost capability without buying any.

That is consistent with `docs/POINTER-AIM-VS-VOTE.md` and `docs/COPY-PEAK.md`: the copy operation is strongest at the
start of a mixed run (69.8% held-out at step 1,000) and decays monotonically as language training proceeds. Training
language first puts the model past that window, and no later mixture returns to it.

## The honest conclusion
A single mixed run cannot hold both abilities at their best, and staging the training does not fix it either. The next
attack has to be architectural - a copy route that language training cannot overwrite - not another schedule, freeze or
learning-rate choice. Six schedules and protection schemes have now been measured and published as failures.
