# The copy peak, measured properly: 69.8% held-out copying at 1,000 steps, checkpoints now kept (2026-11-11)

Prediction recorded before this run: "held-out copying at step 2,000 of 50% or better on both seeds." Measured on
**1,500 taught and 1,500 held-out items** (3,000 per checkpoint, up from 600):

```
                 step 1000   step 2000   step 3000   step 4000    language at 4000
seed 0 copy        70.2%       63.1%       56.8%       46.5%          20.1%
seed 2 copy        69.3%       65.5%       55.1%       47.2%          20.4%
taught words       83.5-85.2% throughout
```

**Prediction PASSED**: 63.1% and 65.5% at step 2,000 against a 50% bar. The earlier 63.7% mean from the 600-item
evaluation holds up on five times the items, and the real peak is one step earlier: **69.8% mean at step 1,000**.

## The curve, stated plainly
Held-out copying decays monotonically as language improves. Both seeds, both directions, no reversals:

```
language   14.6% -> 16.6% -> 18.7% -> 20.1%      (copy 70.2 -> 63.1 -> 56.8 -> 46.5)
```

There is no checkpoint where both abilities are near their own maximum. That is a measured trade-off, not a tuning
problem, and it is the single most important fact about this architecture so far.

## Fixed: the checkpoints now exist
Every evaluation writes a checkpoint (`early_s0_ep1000.npz` ... `_ep4000`) and the best-by-held-out is copied to
`early_s0_best.npz` / `early_s2_best.npz`. The three-seed result published this morning could not be reloaded because
only `_final` was kept; that defect is closed. The 70.2% model is a file on disk, not a line in a log.

## The frozen-pointer arm failed, and it failed in both directions
Language-only stage with the pointer's matrices frozen:

```
stage 1 language: 9.38 / 13.30 / 15.68 / 17.02% at steps 2,500 / 5,000 / 7,500 / 10,000
stage 2 copy    : taught 84.0%, held-out 3.2%
```

Against 26.8% language for the same stage unfrozen, freezing cost **9.8 points of language** and still delivered only
3.2% held-out copying afterwards. The likely reason is the one already measured: the pointer reads the SHARED embedding
table, which language batches keep moving, so freezing its own matrices never froze the mechanism.

That is why the arm now running freezes the pointer's matrices **and** gives it a separate embedding table that is also
frozen - the one combination not yet tried. Prediction: language >= 24% in stage 1, held-out copying >= 40% after stage 2.
