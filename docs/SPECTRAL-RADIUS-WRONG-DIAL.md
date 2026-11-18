# The spectral radius was the wrong dial: memory dies on the AVERAGE singular value (2026-11-18)

Unclamping the recurrence to spectral radius 1.0 gave 5.4x more memory at k=16 - but only 4.73%, against a 10% gate. So the clamp was **a** cause, not the whole cause. This
probe finds the rest.

`tools/saturation_probe.py` on `scan10_r10_ep1500.npz` (the radius-1.0 checkpoint, 2,560 positions of real corpus):

```
Wh spectral radius        1.0000
Wh largest singular value 1.9601
Wh MEAN singular value    0.8424
memory operator diag(1 - tanh^2) @ Wh, mean singular value   0.7618

tanh saturation:  |preactivation| > 2.0  ->  0.00% of units
                  |preactivation| > 3.0  ->  0.00% of units
                  mean |preactivation| 0.269, 90th percentile 0.558
```

## What this rules out, and what it leaves

**Tanh saturation is not the problem - it is not happening at all.** Zero units sit in the flat region; the activations are tiny (mean 0.269). The "memory cannot live inside a
saturating nonlinearity" hypothesis, which motivated the whole gated-cell detour, is **not** what limits this model. That detour's premise was wrong.

**The weight matrix is the problem, and the spectral radius is the wrong summary of it.** The operator that actually carries a perturbation one step forward is
`diag(1 - tanh^2) @ Wh`, and its **mean singular value is 0.7618**: the average direction is multiplied by 0.76 every step. Sixteen steps of that is `0.7618^16 = 0.0129`, i.e.
**1.29% predicted retention** - and 4.73% was measured, the same order. A matrix can have spectral radius exactly 1.0 and still shrink almost everything, because spectral radius
constrains only the largest eigenvalue modulus, not the behaviour of the average direction. Non-normal matrices do this routinely.

`renorm_recurrent(p, 1.0)` was therefore always going to produce a model that forgets: it held the one number that does not govern memory while leaving the one that does
unconstrained.

## The change, and its prediction

`lens/scan_batched.py` gains **`ortho_recurrent(p, scale)`**: project `Wh` onto an orthogonal matrix every step with a deterministic-sign QR, so **every singular value is
exactly 1**. Verified: on a random 64x64 matrix the projection moves mean/min/max singular values from 6.78 to 1.000000 / 1.000000 / 1.000000.

Two arms (`lens/scripts/run_ortho.sh`), 1500 steps, waiting for the machine to be free first:

- **P1** k=16 memory rises **above the 10% gate** at scale 1.0 - the first arm whose mechanism should reach it
- **P2** scale 1.0 trains worse than 0.98 (a perfectly expansive loop is harder to fit); 0.98 stays within 2 points of the radius-1.0 arm's 13.23%
- **P3** the memory operator's mean singular value rises above 0.90 on the finished checkpoint
- **FALSIFIER** k=16 memory stays **below 6%** with an orthogonal matrix: then the weight matrix is not the explanation either, the decay comes from the embedding or the
  readout path, and the next probe belongs there.

## The honest state of the memory line

```
radius 0.9   k=16  0.878%
radius 1.0   k=16  4.730%   <- clamp proven to be a cause; 3.04 points of accuracy cost
gate             10.000%
orthogonal   running
```
Every number above is measured on a checkpoint, not predicted. The 10% gate has not been reached by any configuration yet.
