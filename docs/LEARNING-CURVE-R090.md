# The returns-to-training curve at radius 0.9, measured on every checkpoint (2026-11-18)

"Training length is the lever" came from two points (1500 -> 3000 steps). Two points are not a curve, so this measures held-out accuracy on **all twelve** 250-step checkpoints of the
radius-0.9 arm, same protocol throughout (`tools/learning_curve.py`, 12 held-out chunks, scored positions 64-127).

```
checkpoint                  step   held-out       loss    change
scan11_r090_ep250.npz        250      5.29%     7.4510
scan11_r090_ep500.npz        500     12.70%     7.0637    +7.41
scan11_r090_ep750.npz        750     13.49%     7.0336    +0.79
scan11_r090_ep1000.npz      1000     16.80%     6.8275    +3.31
scan11_r090_ep1250.npz      1250     16.27%     6.8106    -0.53
scan11_r090_ep1500.npz      1500     17.20%     6.6977    +0.93
scan11_r090_ep1750.npz      1750     17.33%     6.6526    +0.13
scan11_r090_ep2000.npz      2000     19.31%     6.5897    +1.98
scan11_r090_ep2250.npz      2250     19.71%     6.5370    +0.40
scan11_r090_ep2500.npz      2500     20.24%     6.4954    +0.53
scan11_r090_ep2750.npz      2750     20.90%     6.4535    +0.66
scan11_r090_ep3000.npz      3000     21.03%     6.4417    +0.13
```

## What is solid, and what is noise

**The overall climb is real:** 5.29% at step 250 to 21.03% at step 3000, a **15.7-point** gain. Loss falls monotonically from 7.4510 to 6.4417 with one small wobble.

**But the standard error on a single point is 1.47 points** (p ~ 0.21, 768 scored positions). That number changes what may honestly be said about the fine structure:

- The four most recent increments are **+0.40, +0.53, +0.66, +0.13** points per 250 steps - mean **+0.43**, sd 0.20. **Every one of them is inside the noise.** So "the curve is
  decelerating" is **not established**; the appearance of flattening is what increments smaller than the error bar look like.
- What *is* resolvable is the coarse statement: from step 1500 (17.20%) to step 3000 (21.03%) the model gains **3.83 points, about 2.6 standard errors**. Doubling training helped,
  and that claim survives its error bar.
- The non-monotone step (`ep1000` 16.80% higher than `ep1250` 16.27%) is a 0.53-point wobble against a 1.47-point error - noise, not a regression.

## What this means for the 6000-step arm's prediction

Linear fits over the later half of the curve give slopes of **+0.0018 to +0.0027 points per step**, which extrapolate to **26.7% - 29.6%** at 6000 steps. That comfortably contains the
prediction of P1 (>= 24%). **But the fit is built on points whose individual error is 1.47 points**, so it is a weak extrapolation and is reported as one: the honest statement is
that 24% is **plausible and unproven**, and the 6000-step arm is the test rather than this arithmetic.

## Why the curve matters anyway

It converts "train longer" from a slogan into a measured shape: the model is still gaining at step 3000, the gains are of the same order as the measurement error, and any future
claim about a plateau at this scale will need either more scored positions per point or several seeds, because a single-curve argument cannot see a plateau that is smaller than
1.47 points per point.
