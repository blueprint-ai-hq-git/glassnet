
## Radius 1.0 gives a real but small free-running gain, and it fails my own memory gate (2026-11-17)

The 12-chunk sweep suggested radius 1.0 beat the shipped 0.9 by 0.84 points of accuracy. Confirmed properly on **5,120 tokens** for prediction and **1,600 paired tokens** for
generation:

```
r      acc%    entropy      top1   sum(pr)   free-run acc%      rep
0.9   16.70%     6.6505    0.1256  1.000000          5.00%     0.915
1.0   17.23%     6.7473    0.1138  1.000000          6.75%     0.917

accuracy  : +0.53 points, per-chunk sd 8.08, SE 1.28  ->  WITHIN NOISE
free-run  : +1.75 points, paired bootstrap 95% CI [+0.44, +3.38]  ->  RESOLVABLE
repetition: 0.915 -> 0.917  ->  unchanged
```

## What is real and what is not

**Not real:** the accuracy gain. +0.53 points against a standard error of 1.28 on this evaluation is indistinguishable from zero, and the earlier +0.84 came from a 1,536-token slice.
**The "nearly free point" from the sweep does not survive confirmation and is withdrawn.**

**Real:** free-running accuracy rises from **5.00% to 6.75%**, paired by start, with an interval that excludes zero. Relaxing the spectral clamp does improve generation slightly - exactly the
direction the memory analysis predicted.

**Real but fatal to the framing:** repetition is **unchanged** (0.915 vs 0.917). The loop signature did not move, and a generation gain that leaves repetition untouched is not addressing
the mechanism that produces the loop.

## Against my own gate

The gate written before this work: *k=16 perturbation shift above 10% while held-out accuracy holds at or above 16%.*
At radius 1.0 the shift is **1.864%** and accuracy is **17.23%**.

**Accuracy holds; the memory condition fails by a factor of five.** So the honest verdict is that radius 1.0 is a **small real improvement** that **does not pass its own pre-registered gate**, and
therefore does not establish that memory length was the binding constraint. Free-running 6.75% is also below the 10% promotion bar used throughout this line.

## Where this leaves the clamp

Two facts now stand together and they are both mine to hold:

- The **cause** of the exponential forgetting is `renorm_recurrent(p, 0.9)` running every step - a clamp below 1, which contracts perturbations by construction. **Confirmed.**
- **Relaxing that clamp modestly is not the fix**: 0.9 -> 1.0 buys 1.75 points of free-running accuracy and nothing on repetition, while pushing to 3.4 (where the memory gate does pass) destroys
  accuracy to 1.50%.

The clamp explains the decay; it does not follow that it is the lever. A mechanism that keeps information **selectively** is still what the evidence points at - a gate, not a global scalar - and any
such mechanism must be trained with the clamp raised or removed, or it will be normalised back to radius 0.9 after every step and report no effect.
