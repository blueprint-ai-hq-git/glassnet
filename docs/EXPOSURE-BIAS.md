
## Why generation loops: exposure bias, measured (2026-11-17)

The model predicts the next token correctly **16.70%** of the time on held-out text with a broad, mildly under-confident distribution - and then produces `"the and of the"` forever
when asked to generate. That gap is not calibration and not saturation. It is the difference between the condition the model is trained in and the condition it is used in.

`tools/decode_path.py` runs the same model from the same starting context for 40 steps, twice, and only the input differs:

```
teacher-forced accuracy : 22.19%   (320 tokens)   <- fed the true next token, as in training
free-running accuracy   :  4.06%   (320 tokens)   <- fed its own argmax, as in generation
free-running repetition :  0.93 mean (1 - unique/40); 1.0 means the same token every step
```

## What those three numbers say together

**Accuracy collapses 22.19% -> 4.06%, a factor of 5.5, purely from consuming its own output.** And repetition goes to **0.93**, meaning across 40 steps it emits roughly **3 distinct tokens**.
That is the loop, quantified: one wrong token moves the recurrent state into a region it never saw in training, and the state then drives nearly the same prediction at every subsequent
step.

**Exposure bias is therefore the mechanism**, confirmed by the falsifier that was written into the script before it ran: *if free-running accuracy were not collapsing, the loop would have
another cause.* It collapsed.

## Why this reframes the whole scan line

Every scan run 1-9 was scored on **teacher-forced held-out accuracy**, which is the training condition. On that measure run 9 improved to 16.93%, and the improvements were real. But
**teacher-forced accuracy cannot see this defect at all** - the model is being graded in the one condition where it is fine.

So the honest standing of the scan line is:

- teacher-forced accuracy: **16.70-22.19%** depending on context length, real and improving
- free-running accuracy: **4.06%**, and repetition **0.93** - the model does not generate
- the gap between them is the thing that was never measured until now

## The lever, named but not yet tried

**Training on its own rollouts** (scheduled sampling, or an explicit free-running loss term) is the standard family for this, and it is the first mechanism found in this line that
addresses a defect measured in the generation condition rather than the training one.

It also has an obvious cheap gate, pre-registerable now: **free-running accuracy above 10% while teacher-forced accuracy stays at or above 16%**, with the repetition ratio below
**0.5**. Falsifier: if free-running accuracy stays under 6% after the treatment, exposure bias was not the binding constraint and the loop has to be sought elsewhere - most likely in
how the generation context is built compared with the training window.

## What this does not say

It does not say the architecture is wrong, or the readout, or the states. Those measured fine: accuracy above its bar, states decorrelated at cosine 0.1736, readout 47x sharper than
noise. **The failure is that the model has never been trained to survive its own mistakes**, and every previous measurement was taken in a condition where it never has to.
