

---

# THE MID-BAND FAILURE IS FREQUENCY BIAS, NOT CAPACITY: THE CORRECT TOKEN IS USUALLY AT RANK 34 (2026-11-19 19:15)

The band budget showed the biggest headroom in this project sits in the r21-500 band: **36.08% of scored positions at only 8.13% accuracy**, worth about
**+4.3 points** overall if moved to 20%. Before designing a repair I measured WHICH failure it is, because the two candidates need opposite fixes.

```
MID-BAND DIAGNOSTIC  win6koff_K0_ep6000.npz
  windows 8, scored positions 483, of which r21-500: 192
  r21-500 accuracy 14.58%   top-20 accuracy 36.04%

  P1 prediction is a TOP-20 token when the target is r21-500 : 74.48%   -> MET (frequency bias)
  P2 of the wrong ones, prediction is also r21-500           : 12.80%   -> NOT met
  P3 median rank of the CORRECT token in the model's ordering : 34      -> MET
     top-100 hit rate on r21-500 targets: 66.67%
  spread of wrong predictions (sd of log10 predicted rank)    : 0.476
  FALSIFIER: did not fire
```

**SMALL SAMPLE, stated up front:** 8 windows and 192 mid-band positions, so treat the percentages as indicative and the direction as the finding. The
population figures (8.13% accuracy on 36.08% of positions) came from 104 windows; this diagnostic needs the same before it is quoted as a magnitude.

## What this establishes

1. **It is not a near-miss problem.** If the state could not separate mid-frequency tokens, wrong answers would be other mid-frequency tokens. Only
   **12.80%** are - the model is not confusing `rank 300` with `rank 350`.
2. **It is frequency bias.** On mid-band targets the model outputs a **top-20 token 74.48%** of the time. It defaults to the head of the distribution.
3. **The information is present.** The correct token's **median rank is 34**, and on **66.67%** of mid-band positions it is inside the model's own top-100.
   The model is not ignorant of the answer; it is choosing not to say it.

## Why this matters more than any architecture change

A capacity repair (wider embeddings, untied readout, more layers) would need to move the correct token from rank ~34 to rank 1 by representing more. That is
expensive. A **calibration** repair only has to re-rank answers the state already contains - and it can be tested at inference, with no retraining at all.

The obvious first candidate is **logit adjustment**: subtract `tau * log(prior)` from the logits before the argmax, which is the standard correction for
exactly this failure. The tradeoff is explicit and measurable: the top-20 band will lose some hits while the mid band gains, and the question is whether the
total rises.

Pre-registered for that experiment:
- **P1** some `tau > 0` raises OVERALL accuracy above the shipped 0/tau baseline, on 104 windows, both instruments
- **P2** the mid band gains more than the top-20 band loses
- **P3** the effect is monotone in `tau` up to a peak and then reverses - a single lucky `tau` with no structure is a warning, not a result
- **FALSIFIER:** no `tau` beats the baseline by more than the paired resolution (0.009 points is the SE, so require >= 0.5 points) -> calibration is not the
  lever either, and the mid band needs a different explanation.

This is a cheap experiment (inference-time only, minutes not hours) aimed at a **+4.3-point** headroom, and it exists because the diagnostic ruled out the
expensive explanation first.
