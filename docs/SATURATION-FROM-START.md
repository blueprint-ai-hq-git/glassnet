
## The saturation is present from the first checkpoint: the readout is born one-hot (2026-11-17)

`tools/saturation_trajectory.py`, same held-out tokens at every checkpoint:

```
ckpt     logit_rng  max|lg|   entropy    %unif    acc%
ep250         11.6      9.1    0.0108     0.12%    7.10
ep500         11.8      9.8    0.0088     0.09%   14.91
ep750         11.8     10.6    0.0092     0.10%   16.28
ep1000        12.1     11.3    0.0074     0.08%   17.84
ep1250        12.1     11.6    0.0078     0.08%   17.45
ep1500        12.1     11.7    0.0079     0.08%   17.97
```

## Hypothesis B, and it is not close

Two stories were written down before measuring. **A**: the readout's scale grows during training and drags the logits apart. **B**: the logits are huge from the start.

**B, decisively.** The logit range moves **11.6 -> 12.1, a factor of 1.05**, across 1,250 training steps. Entropy never rises above **0.0108 nats** at any checkpoint. Meanwhile accuracy climbs **7.10% -> 17.97%**.

So the model is not drifting into overconfidence. **It starts there.** A logit range of ~12 over 12,002 words is a softmax ratio of e^12 = 162,000 to 1 between the top word and the bottom, which is one-hot for practical purposes, and it is already nearly that at step 250.

## What this changes about the fix

The interesting quantity is not "why does confidence grow" - it does not. It is **"what does training do with a saturated readout"**: it raises accuracy from 7% to 18% **without ever softening a single distribution**. The model is learning *which* word to be confident about while remaining certain about all of them.

That is the signature of an objective with a very strong incentive to commit: cross-entropy on a 12,002-way vocabulary pays nothing for hedging, so the cheapest way to reduce loss is to sharpen. The failure was never a lack of sensitivity in the readout - **F4's implicit theory was that the answer layer was blind, and the truth is the opposite: it is over-committed.**

## The lever this points at, to be pre-registered rather than tried

1. **Label smoothing** - makes hedging pay a little, directly opposing one-hot commitment.
2. **Readout temperature / scale control at initialisation** - start the logits near the uniform regime instead of near one-hot.
3. **A confidence penalty tied to the measured entropy** - but runs 1-8 already showed penalties on this model tend to be erased by training, so this is the weakest of the three.

Whichever is tried, the gate is now measurable and cheap: **entropy on held-out real text must rise above ~0.5 nats while accuracy holds near 18%,** and the falsifier is that if entropy stays under 0.05 nats, the lever does not touch the mechanism.

## One thing this does not say

It does not say the architecture is wrong. The states decorrelate, the readout beats noise 859x, and accuracy is 3.2x its bar. **What is broken is calibration, not capacity** - and calibration is the first defect in this project that has a clean, cheap, falsifiable test from the very next run.
