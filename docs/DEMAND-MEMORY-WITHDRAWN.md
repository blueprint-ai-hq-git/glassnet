# WITHDRAWN: "demand training creates 16-step memory" is refuted by a frequency split (2026-11-18)

Three entries today claimed that delayed-target training gives the model measurable 16-step recall, with the memory bank worth +3.45 points. **That interpretation does not survive a
control that splits accuracy by the frequency of the target token, and it is withdrawn.**

## The test

If a model recalls *which* token occurred 16 steps back, it must get rare tokens right sometimes. If instead it has merely learned a better prior over frequent tokens, all of its gain will
sit in the most common tokens and it will score **nothing** on the tail. Frequency ranks are computed on the **training region only**, so the split cannot borrow held-out information.

```
delay 4    (250 steps)        top-20    ranks 21-500    tail (888)     tail n
  td4 arm                     35.16%       4.83%         0.00%          888
  base                        19.61%       4.75%         0.00%          888
  same-window constant        24.87%       5.56%         0.56%

delay 16   (1000 steps)       top-20    ranks 21-500    tail (892)     tail n
  td16 arm                    16.15%       1.60%         0.00%          892
  base                        10.09%       2.14%         0.00%          892
  same-window constant        23.65%       6.85%         0.56%
```

## What the numbers say

1. **All of the gain is in the frequent band.** At delay 4: +15.55 points on the top-20 tokens, **+0.08** at ranks 21-500, **0.00** on the tail. At delay 16: +6.06 on top-20,
   **-0.54** at ranks 21-500, **0.00** on the tail.
2. **Rare-token recall is exactly zero at both delays** - 0 correct out of 888 and 892 positions, against a constant predictor that gets 0.56%. One correct answer would have shown as
   0.11%, so this is not a rounding artefact; the model never once recalled a rare token from a delay.
3. **The arm is below a per-window constant predictor in every band** at delay 16 (16.15 vs 23.65, 1.60 vs 6.85, 0.00 vs 0.56), and below it in two of three bands at delay 4.

So what the delayed-target objective changed is **how the model distributes probability mass across frequent tokens** - a readout/prior shift. It did **not** create token-specific memory,
and the earlier "above a fair constant predictor" framing was measuring the prior shift, not recall.

## Why the earlier claims looked positive

They compared **pooled** accuracy against a **pooled** constant baseline. Pooled numbers are dominated by the top-20 tokens, which are 3325 of the 6552 scored positions. In that band the
arm does beat the pooled constant, so the pooled comparison turned positive - while the band where memory would have to show, the tail, sat at zero the whole time. **A pooled comparison
hides exactly the failure this project exists to catch.**

## What still stands from today

- The **memory module** itself holds a fact for 16 steps in isolation: 81.25% recall at 16 steps, control at chance, probe gradient certified at 6.789e-08, one bank per sample. That result
  is untouched by this correction - it is a different measurement with a different control.
- The **bank's contribution** to the delayed-task numbers (+3.45 points at delay 16) is real *as an accuracy difference*, but it is a difference in the same frequency band, so it is not
  evidence of memory either.
- The **infrastructure** stands: `tools/eval_delayed.py`, `eval_delayed_ci.py` (window-level bootstrap) and now `eval_delayed_split.py`.

## The lesson, stated as a rule

**A pooled accuracy gain is not evidence of the capability the task is named after.** Any claim about recall must be split by target frequency first, because the frequent band is large
enough to carry a pooled number on its own. This control should have been run before the first publication, not after three.
