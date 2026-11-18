# The one dishonest concept: a claim 96.9% of images satisfy cannot be read (2026-11-18)

The readable vision model's honesty audit reported "20 of 21 concepts passing". That number hides which one fails and why, so this breaks it down per concept
(`tools/concept_honesty_detail.py`, 4,000 held-out images).

```
k    concept                     AUROC     fires     truth   pass
0    is_bright                  0.9565     21.7%      5.2%    yes
1    is_dark                    0.9683     36.2%     29.9%    yes
2    has_high_contrast          0.7456     53.6%     61.5%    yes
7    centre_heavier             0.3196     39.2%     96.8%     NO   <- the failure
...
18   has_top_edge_ink           0.9601     27.8%      8.8%    yes   ADDED TODAY
19   has_bottom_edge_ink        0.9651     32.9%     21.3%    yes   ADDED TODAY
20   has_diagonal_structure     0.8003     42.8%     15.6%    yes   ADDED TODAY
```

## The failure is an inversion, not noise

`centre_heavier` scores **AUROC 0.3196 - below chance.** The neuron fires *more often* on the 3.2% of images where its claim is **false** than on the 96.8% where it is
true. The cause is in the formula: at threshold 0.05 the concept fires on **96.9%** of training images, so there is almost nothing to discriminate. A claim nearly every
image satisfies is not a property of an image, and a neuron asked to match it can only learn the prior.

This is the same defect class the project already recorded on the text side: a **near-constant** concept can pass an audit for free. Here it fails instead of passing, which
is luckier but the same root cause - a concept with no negatives to rank against.

## The three added today are all honest

`has_top_edge_ink` **0.9601**, `has_bottom_edge_ink` **0.9651**, `has_diagonal_structure` **0.8003** - all well above the 0.55 bar, and two of them are among the most
load-bearing units in the model. The widening that took the layer from 13 to 21 units bought genuinely used, label-honest neurons.

## The fix, and its predictions

The formula keeps its meaning but sharpens its claim: `centre_heavier` goes from threshold **0.05 to 0.40**, roughly one standard deviation above the training mean
(mean 0.2993, sd 0.1364). Measured before launch: the concept now fires on **23.45%** of training images instead of 96.9%, and the layer still keeps all 21 concepts at
the 0.002 support floor.

`lens/scripts/run_fix_centre.sh` (waits for the orthogonal-recurrence arms so the machine is never oversubscribed), parity config, seed 0:

- **P1** `centre_heavier` AUROC rises above **0.80** (from 0.3196)
- **P2** its support lands between **10% and 40%** - informative, not near-constant
- **P3** overall accuracy stays within **0.5 points** of 84.62% and honesty reaches **21 of 21**
- **FALSIFIER** AUROC still below **0.60** with the sharper threshold: the neuron is not learning this concept at all, and project doctrine says **drop it** rather than keep
  a named neuron that cannot mean what it says.
