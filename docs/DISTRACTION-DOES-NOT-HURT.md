

---

# DISTRACTION DOES NOT HURT IT AT ALL — SO "IMPORTANCE" NEEDS A DIFFERENT TEST (2026-11-22 13:55)

The pre-registered question was: does junk input actually damage this model? If it does, an importance mechanism has a job.
If it does not, building one would be decoration.

## Measured, "find the one real digit and copy it", decoys growing 8 -> 64

```
 8 positions   accuracy 100.0%   (chance 10%)
16 positions   accuracy 100.0%   (chance 10%)
32 positions   accuracy 100.0%   (chance 10%)
64 positions   accuracy 100.0%   (chance 10%)
```

**P1's condition was met exactly:** accuracy never fell below 95%, so **the model already ignores distractors perfectly**, and an
importance gate would have nothing to improve. Recorded as **IMPORTANCE IS NOT THE BOTTLENECK for behaving correctly.**

## But something real is still missing, and it is the thing the owner actually asked for

Two measured facts now sit side by side:

- **it behaves as if it knows what matters** - 100.0% with 64 positions of junk around the single real digit;
- **it cannot report what mattered** - the probe that tried to read that knowledge out scored **77/100** against a bar of 90.

So the model *acts* on importance and cannot *say* what it acted on. For a project whose entire purpose is that a person can read
why an answer was given, that gap is the real problem - not accuracy.

## The next test, built on this project's own discovery

The machine could not add until it was allowed to **write the working out**. So the same medicine, applied to explanation: do not
bolt on a probe head - **ask the model to write down which position it used, as part of its answer.**

```
  input :  positions, exactly one holds a real digit, the rest are decoys
  output:  [the position of that digit] [the digit itself]
```

If it can emit the position, the explanation is **the model's own answer, in the open, checkable by a person**. That is what
"readable" is supposed to mean, and it is a far stronger standard than an AUROC score from a side-probe.

Bars: position >= 90%, digit >= 95%, both >= 85%. Falsifier: position < 80% means it cannot state its own reason even when asked
to write it, and the mechanism must be different.
