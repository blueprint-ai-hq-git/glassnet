
## The four ops a transformer block needs are gradient-correct before any model code is written (2026-11-17)

Gate result for `tools/gradcheck_newops.py`. Every op was derived, coded, and checked against central finite differences in float64, with every input entry
perturbed separately. Absolute error, and the same error normalised by the scale of the gradient:

```
op                         check          max abs err    rel-to-scale
RMSNorm (+ gain)           d/dx, d/dgain   6.3e-10       2.1e-10   PASS
RoPE                       d/dx            2.4e-09       4.5e-10   PASS
SwiGLU (gate/up/down)      d/dx + 3 d/W    1.4e-08       2.6e-10   PASS
GQA attention, causal      d/dx + 4 d/W    1.1e-08       5.5e-11   PASS
```

**Twelve checks, all passing at 1e-8 or better.** These are exactly the mechanisms our shipped model lacks: RMSNorm (no normalisation), RoPE (no relative
position), SwiGLU (no gated MLP), grouped-query attention (no attention at all). They can now enter the engine.

## The harness had to be fixed before it could be trusted

The first run reported attention's `d/dx` as **FAIL at 9.8e-05 relative error**. It was the metric, not the gradient: attention gradients contain entries that are
legitimately near zero, and a relative-error denominator of 1e-12 makes any such entry look catastrophic. Diagnostic before the fix:

```
max ABS error              9.5e-09
worst relative error       2.5e-08
eps sweep 1e-4..1e-7       analytic tracks numeric at every step size
```

So the check now judges on **absolute** error with the relative figure reported for information, and it prints both. A metric that fails correct code is exactly as
dangerous as one that passes broken code - and this project has been bitten by both directions this week.

## And the harness proves it can fail

A deliberately corrupted gradient (0.1% error injected into the RMSNorm `d/dx`) is **caught**: absolute error 6.1e-03 against a 1e-7 threshold, relative 5.0e-04.
A check that cannot fail is not a check, so that run is part of the script.

## Where this leaves the build

The architecture step now has its foundation proven: the four missing mechanisms are correct in isolation. The remaining work is composition - assemble them into a
block, prove that a stack with the stack-depth zeroed reproduces the current model exactly (step-0 equivalence), then train candidate B from the spec (d384, 8
layers, 24.5M parameters).

Two bugs of mine were found and fixed inside the derivation itself before any of this passed: the RMSNorm backward had the wrong contraction (`g*dy` instead of
`g*dy*x`) and the gain gradient summed the wrong axis. Both were caught by the numeric check doing its job, which is the entire reason the check exists.
