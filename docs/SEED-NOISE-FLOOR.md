

---

# SEED NOISE IS 2.80 POINTS, SO A 0.5-POINT MEMORY EFFECT IS UNMEASURABLE BY SEEDS ALONE (2026-11-19 14:30)

Three controls now exist for the same configuration (K=0, 3000 steps, tied readout, identical settings, only the seed differs):

```
seed 0  win3kwinoff_K0_ep3000   14.75%
seed 1  s1off_K0_ep3000         16.03%
seed 2  s2off_K0_ep3000         10.50%

mean 13.76%   sd 2.80 points   range 5.53 points
```

**The arm's own seed-to-seed spread is 11x the +0.50 effect the window bank was credited with this morning.** Combined with the corrected protocol, where all
three arms sit at 12.80-12.97% against a 13.14% constant, the picture is unambiguous: these arms cannot resolve the effect they were built to find.

## What it would take, computed rather than guessed

Paired differences have sd ~ sd*sqrt(2) = 3.96 points. To detect a paired effect at 95% confidence:

```
effect 0.5 points  ->  242 seeds per arm
effect 1.0 points  ->   61 seeds per arm
effect 2.0 points  ->   16 seeds per arm
effect 3.0 points  ->    8 seeds per arm
```

So a sub-point memory effect **cannot be established by adding seeds** - not with any number of runs this project can afford. There are exactly two honest
routes:

1. **Only claim large effects.** A memory mechanism must move next-token accuracy by **3+ points** to be detectable with 8 seeds. Anything smaller is
   unmeasurable at this budget and must not be reported as a result, positive or negative.
2. **Use a sharper instrument.** Next-token accuracy is a poor probe of memory because the model can score without using its memory at all. The
   delayed-recall probe already exists and has enormous dynamic range on the same architecture: **100.00% at delays 1, 4 and 8 against a 3.12% chance**,
   and **81.25% at delay 16**. That instrument separates "the memory works" from "the memory does nothing" by ~97 points instead of 0.5.

## The strategic conclusion

**The memory bank should be judged by the delayed-recall probe, not by next-token accuracy.** Every arm today graded memory on a metric whose noise floor is
11x the effect size, which is why the results kept landing on "indistinguishable from zero" - partly the mechanism, but partly the instrument. The probe is
already certified (BPTT 6.789e-08) and already measured at four delays.

That does not excuse the mechanism: on the probe, the bank holds a fact for 8 steps perfectly and 16 steps at 81.25%, and the real-text question (does it
help a 1.68M model generate) is still open. But the next memory measurement should be on the sharp instrument, and next-token accuracy should only be used
to confirm that a mechanism does not COST anything.
