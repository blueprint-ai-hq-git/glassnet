# Shift augmentation removes the 27-point collapse, at a measured cost of 4.4 points of clean accuracy (2026-11-18)

The robustness report showed the readable image model collapsing under a **2-pixel shift**: 84.60% clean, **56.77%** shifted, a **27.83-point** loss. The diagnosis was structural -
a fixed patch grid is alignment-sensitive by construction. So the arm `shiftaug` trained the same architecture with **2-pixel shift augmentation**.

## The comparison that makes this attributable

The first version of this comparison was invalid: the collapse was measured on a model with **4 patches of 14x14**, the augmented arm runs **4 x 56**. Different configs, so the
clean accuracies were not comparable and nothing could be attributed. A matched control was therefore measured - **the same parity configuration, same seed, no augmentation**
(`lens/runs/image/fixc3_s0_ep24.npz`, 15,977 params), against the augmented model (`shiftaug_s0_ep24.npz`, same config, 2px augmentation):

```
input                           no augmentation   2px shift aug    difference
clean                             84.47%            80.03%          -4.44
brightness +0.10                  82.90%            77.83%          -5.07
brightness +0.20                  77.63%            71.60%          -6.03
brightness -0.10                  82.97%            78.23%          -4.74
brightness -0.20                  78.40%            73.23%          -5.17
contrast x1.30                    83.53%            78.77%          -4.76
contrast x0.70                    81.90%            77.47%          -4.43
gaussian noise sd 0.05            84.17%            80.30%          -3.87
gaussian noise sd 0.10            83.63%            79.50%          -4.13
blur 3-tap binomial               83.20%            80.10%          -3.10
shift 1 px right                  76.93%            79.80%          +2.87
shift 1 px down                   78.33%            79.60%          +1.27
shift 2 px right                  57.00%            78.10%         +21.10
```

## What it bought and what it cost

**Bought: the collapse is gone.** Under a 2-pixel shift - the condition a real user produces - the augmented model scores **78.10%** against the control's **57.00%**: a **+21.10-point**
improvement, and the shift penalty falls from **-27.47 points to -1.93**, a **14x** reduction. The augmented model under a 2-pixel shift (78.10%) is closer to the control's *clean* score
(84.47%) than to the control's *shifted* score (57.00%).

**Cost: 4.44 points of clean accuracy**, and every other perturbation (brightness, contrast, noise, blur) moves down with it by 3.1 to 6.0 points. That is the honest shape of the trade:
augmentation does not add robustness on top of accuracy, it **moves accuracy from pristine inputs to realistic ones**.

**The one asymmetry worth naming:** shift robustness came at the price of brightness robustness. Brightness +0.20 was -6.83 for the control and -8.43 for the augmented model, i.e. **1.60
points worse**. Training on shifted copies of the same images does not teach anything about global intensity.

## The arm's own accuracy delta is uninterpretable

`shiftaug` also reported readable-vs-plain **-1.01 points** at **one seed**. One hour earlier the 3-seed fixcentre arm measured the gap distribution as **mean +0.12, sd 2.02**. A single
-seed -1.01 is **inside that spread and carries no information**. It is recorded and not interpreted, and the arm's `concepts passing 21/21` is consistent with the fixcentre result.

## Why this is the strongest result of the session

Every other line this session ended in "not resolvable" or "withdrawn". This one is a **21-point improvement in the condition that actually occurs**, measured against a **matched
control**, with the cost stated in the same table. It also confirms the structural diagnosis: the model was alignment-sensitive exactly as the fixed patch grid predicts, and teaching it
about alignment fixed it.
