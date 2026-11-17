
## The zero-gain norm fixes the explosion and does not fix the stability (2026-11-17)

Option 3 from the previous entry is built: a third RMSNorm at the block output, blended in by a per-feature gain `g3` that starts at **exactly zero**, so the block is
still an exact no-op at step 0 while the scale of what feeds `concept.W` comes under control as the gain grows.

**Gradient and no-op certification** (`tools/block_selftest.py`, now including `g3`):

```
Wq 2.8e-09  Wk 1.9e-09  Wv 2.2e-09  Wo 2.3e-09  Wg 3.6e-09
Wu 3.9e-09  Wd 2.4e-09  n1 3.2e-09  n2 1.7e-09  g3 9.6e-10     all PASS
step-0: max |block(x) - x| = 0.000e+00                          exact
learns:  loss 2.856814 -> 1.311742 (54.1%)                      unchanged
```

**Warm-start artifact rebuilt**: 15 tensors, **29,882,337 parameters** (the extra 1,024 are `g3`), step-0 equivalence still 1.624e-06.

## The pilot, after the fix

```
lr 5e-4, clip 1.0, 20 steps
  step  0  train loss 6.5691   |g| Wo 136.4  Wd 170.2  emb 4.4
  step 10  train loss 1.8151   |g| Wo 308.0  Wd 360.3  emb 9.3
  step 19  train loss 5.0908   |g| Wo 196.5  Wd 220.3  emb 5.5

net descent 6.5691 -> 5.0908   (no explosion: the previous runs went to 10.1)
val loss 6.5568 nats            shipped model on the same text: 6.06 nats
```

**What improved**: the explosion is gone. The loss recovers instead of diverging, and it ends below where it started.

**What did not**: the loss still oscillates between 1.8 and 5.1 across steps, and **validation loss is 6.5568 against 6.06 for the shipped model** - so after 20 steps the
arm is *worse than the model it started from*, on held-out text. Nothing here is learning anything that generalises yet.

## The honest reading

Two separate problems were hiding behind one symptom. The first - an unnormalised 32768-vector feeding a 53-wide projection - is fixed and certified. The second is
optimisation: with gradients this large (`Wo` at 308, `emb` at 9.3) and only 2 windows per step, a clipped single-rate update moves the model around without settling.
That is an optimiser problem, not an architecture problem, and it needs the same treatment every other arm got: a proper rate schedule, larger batches, and an
evaluation every epoch rather than a single 20-step pilot.

**No claim is made about whether depth helps.** The falsifier written for this arm - breadth still under 0.20 with depth in place means the concept bottleneck K is the
binding constraint - remains untested, because the arm has not yet run in a state where its numbers mean anything.

## Cost picture, unchanged

0.95 s/step at batch 2 on two cores = about 127 windows/minute. A real arm needs the GPU; that is arithmetic, not preference.
