

---

# THE CLIFF IS AT RANK 50, NOT 500 - AND THE 8-WINDOW FIGURES ARE CORRECTED BY THE FULL SAMPLE (2026-11-19 20:00)

Two things came out of running the mid-band diagnostic at the full **104 windows** and splitting the band. Both matter, and one of them corrects numbers I
published an hour ago from 8 windows.

## 1. The published 8-window figures are superseded

```
                           8 windows (published)    104 windows (authoritative)
r21-500 accuracy            14.58%                   8.13%
predicts a top-20 token     74.48%                   84.12%
median rank of the correct token   34                63
top-100 hit rate            66.67%                   59.22%
```

Every number moved in the same direction, and the **conclusion is unchanged and stronger**: the failure is frequency bias, not ignorance. But the magnitudes
I published at 8 windows were optimistic - 74.48% becomes **84.12%**, and median rank 34 becomes **63**. The earlier note said the magnitude needed the full
104; this is that measurement, and it replaces those digits.

## 2. The band is NOT homogeneous - the cliff is at rank 50

```
sub-band   n      accuracy   predicts top-20
20-50      786    24.05%     71.88%
50-100     451     0.22%     95.12%
100-250    690     0.00%     91.45%
250-500    410     0.00%     83.17%

weighted check: (786 x 24.05 + 451 x 0.22) / 2337 = 8.13%   <- reproduces the band figure exactly
```

- **Ranks 21-50 sit at 24.05%** - nearly two-thirds of the top-20 rate. The model is genuinely a **top-50** predictor.
- **Ranks 50-100 collapse to 0.22%**, and 100-500 are **exactly zero** across 1,100 positions. That is not a gradient, it is a cliff.
- So the "36.08% of positions in the r21-500 band worth +4.3 points" framing was too coarse. The headroom is concentrated in **ranks 21-50** (786 positions in this
  draw), where the model already works, and in the **cliff at 50+** (1,551 positions, 66% of the band), where it scores nothing at all.
- **The top-20 default rate is highest exactly where accuracy is lowest** (95.12% at ranks 50-100), which is the signature of a model that answers "a frequent
  word" whenever it does not know.

## Why this kills the calibration approach more completely than the failed replication did

The prior correction moved the mid band by +1.5 points, but 66% of that band is at **zero** accuracy. A global re-ranking cannot lift a sub-population that sits at
0.00% - it can only shuffle which frequent token wins. The earlier result is now explained rather than merely observed.

## What the next experiment should target

**Ranks 50+ specifically.** The model has all the information for many of them (top-100 hit rate 59.22%) but produces them essentially never - which is a readout
that has never been asked to distinguish 11,950 tokens, only to rank the head. The concrete next step is a **per-frequency-band loss re-weighting that excludes the
top-50** (unlike the rarity arms, which re-weighted the whole vocabulary and were refuted) or an **untied readout with frequency-tiered capacity**, and both need
their predictions written before launch.
