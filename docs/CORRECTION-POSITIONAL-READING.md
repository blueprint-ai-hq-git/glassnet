

---

# CORRECTION (2026-11-21 13:55): THE MODEL CAN READ ITS MOST RECENT POSITION PERFECTLY

Two hours ago I published "the model cannot read by position at all, even one position back". **That was wrong**, and the run that
started after it proves it.

## The contradicting measurement

`copylast` - one digit at the LAST position, six decoys before it, answer = that digit, no query token:

```
copylast  seed 0   100.0%
copylast  seed 1   100.0%
```

**100%, twice.** Reading the most recent position works perfectly. `copyfirst` (the same digit moved to position 0, seven decoys after
it) was still running when this correction was written; it is the comparison that actually isolates distance.

## So what did the recall run actually fail at?

The recall task had a **query token** that states how far back to look (`token 10+q` means "q positions back"). Distance 0 there failed
at chance, even though the underlying job - repeat the last digit - is exactly `copylast`, which scores 100%. The difference is the
query token: the model has to **decode a symbol into an offset** and then use it. That is a computation, not a read.

**The corrected statement of the gap:** the model can read the most recent position (100%), and cannot turn a symbolic query into a
positional offset (chance at every distance). Positional reading is not broken; **addressing arithmetic** is.

## The salience certificate on copy1, two of three arms

```
seed   AUROC    free norm   margin    Spearman   probe acc   verdict
 0     0.7130    0.6988    +0.0142    0.3955      94.0%      SALIENCE_FAILED (all three bars missed)
 1     0.9542    0.6490    +0.3052    0.5080      98.0%      SALIENCE_WORKS (all three MET)
 2     (running)
```

Seed 1 clears every bar decisively: AUROC **0.9542**, margin **+0.3052** over the free baseline, Spearman **0.5080**. Seed 0 misses all
three. **So the layer is not yet certifiable - it works on one seed and not on another, and a layer that works sometimes is exactly what
"100% on every layer" forbids.** L3 stays UNCERTIFIED and the honest next step is to make the certificate itself reproducible before
claiming the layer: more truth sequences, more head training steps, and a check of whether the head's 400-step fit is the unstable part.

## Why this correction is written out

Three times today a number that looked like a conclusion was overturned by the next run: the one-seed capacity win, the fake loss that
was really padding, and now "cannot read by position". Each time the mistake was mine and each time a control measurement caught it.
Writing the correction on the page, next to the claim, is the only way the record stays usable.
