# A trap in our own headline test, found before it could fool us (2026-11-10)

Run 9 adds a learned per-unit gain to the readout. Before letting it run, the gain path was exercised on a
deliberately untrained artefact: run 8's checkpoint with `gain_ro` set to **1.0 for every unit** and nothing
else changed. That is RMS normalisation applied to the state, with no learning whatsoever.

```
checkpoint                                 F1 KL      F4 trained vs noise    G2 entropy
scan8_ep1000 (as trained, no norm)         0.0151     0.0151 vs 0.0153 (0.99x)   99.9% of uniform
same weights, gain_ro = 1.0 (no training)  0.9590     0.9590 vs 0.9604 (1.00x)   46.9% of uniform
```

## What this shows
Simply rescaling the state into the readout multiplies the headline KL by **63x** - 0.0151 to 0.9590 - and it
would pass the F1 bar (> 0.05) that eight previous runs failed. No training happened. Nothing was learned.

But the trained-vs-noise control tells the truth: at gain 1.0 the trained readout scores 0.9590 and a readout
of **random numbers** scores 0.9604. Training contributes exactly nothing; the whole apparent improvement is
scale. Any readout, including noise, separates the six prompts once its input has unit RMS.

## Why this matters more than the number
Without F4, run 9 would have produced a headline we could have published in good faith and it would have been
a mirage. F1 alone is gameable by arithmetic. F4 - the trained readout must beat a random readout **at the same
scale** by 2x - is the only criterion in the set that cannot be satisfied by fiddling with magnitudes.

So the rule for run 9 stands and tightens: a high F1 with F4 at 1.00x is a **FAIL and a scale artefact**, and
it must be reported as one. G2 accompanies it: entropy falling from 99.9% of uniform to 46.9% shows the same
mechanism from the other side, sharper distributions with no new information.

## What would count as a real win
F4 above 2x, with G1 showing the gain has not run away and G2 showing entropy has not collapsed toward zero.
That combination cannot be produced by rescaling alone, because the noise control is rescaled too.
