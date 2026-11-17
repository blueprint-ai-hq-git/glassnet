
## A working transformer block, certified three ways (2026-11-17)

`lens/block.py` is now real, trainable code: pre-norm attention and pre-norm SwiGLU, both residual, with explicit forward and backward in plain NumPy so the
arithmetic is readable line by line. Certification is `tools/block_selftest.py`, and it checks three things because each has failed separately this week.

```
1. BLOCK-LEVEL GRADIENTS (finite differences, every parameter)
   Wq 3.0e-09 | Wk 2.3e-09 | Wv 3.0e-09 | Wo 2.4e-09 | Wg 2.8e-09
   Wu 2.8e-09 | Wd 1.6e-09 | n1 2.3e-09 | n2 7.8e-10        all PASS

2. STEP-0 NO-OP (zero-initialised block)
   max |block(x) - x| = 0.000e+00                            exactly zero

3. DOES IT LEARN? causal block must report the PREVIOUS position's vector
   loss 2.856814 -> 1.311742 after 400 SGD steps            54.1% reduction
```

**The third check is the one that matters.** Reporting the previous position's vector from the current position is impossible for a bag-of-positions model - it
requires attention to actually reach backwards. A 54% loss reduction means the attention path is wired correctly end to end, not merely gradient-correct on paper.

## Why three checks and not one

Each of these failed for me separately this week:

- **ops correct in isolation, composed wrongly**: the RMSNorm backward contracted `g*dy` instead of `g*dy*x`, and the gain gradient summed the wrong axis. Both
  passed nothing until the numeric check caught them.
- **a gate that cannot fail**: my first step-0 test opened the gate while the output projections were still zeros, so both states produced identical output and the
  test "passed" for the wrong reason.
- **a model that does not learn despite correct gradients**: yesterday's auction of supervision settings, where every arm was internally consistent and none of them
  moved the thing being measured.

So the standard for the new architecture is: correct gradients, exact no-op start, and demonstrated learning - and all three have to be produced by tool output,
not asserted.

## What this means for the build

The piece of the design that did not exist yesterday now exists, is certified, and is readable. What remains is wiring it into `lens/model.py` behind its own flag
with the equivalence assertion in the trainer, then the **warm-start arm**: shipped weights plus this block, 29,881,313 parameters, starting from 24.58% +/- 0.055
instead of from noise.

The falsifier for that arm is already written and has not changed: if breadth stays under 0.20 with depth in place, then the concept bottleneck K is the binding
constraint rather than depth, and the next experiment targets K.
