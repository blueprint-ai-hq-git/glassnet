

---

# THE READABLE WINDOW BANK AT 3000 STEPS: +0.50 POINTS, EXACTLY AT THE BAR, NOT YET PROVEN (2026-11-19 12:44)

The first fair asking of the memory question, at a budget where the model is no longer far below trivial. Predictions and falsifier were
written before the run.

```
arm            params      HELD-OUT              train ema   gap
winoff (K=0)   1,684,224   14.75%  (loss 5.8357)   6.4652    -0.6294
winon  (K=8)   1,848,320   15.25%  (loss 5.8194)   6.4538    -0.6344
                                                    difference  +0.50 points

window at the end: ` | class | for | a | library | management | ` | book   (8 distinct real words)
```

- **P1** both arms clear the 14.70% constant - **MET, barely**: 14.75% and 15.25%. The control clears it by 0.05 points.
- **P2** window at least 0.5 points above control - **MET exactly**, +0.50.
- **P3** the window holds K distinct real words - **MET**.
- **FALSIFIER** ("both clear trivial AND window below +0.5") - **did not fire**, by 0.00 points.

## Why this is not a win yet, stated plainly

A 0.50-point difference measured on **one seed** with a 104-window evaluation is **inside the noise of the instrument**: the standard error
of a single 104-window number for this model class is around 1.5 points. So the correct statement is:

**The window bank's point estimate lands exactly on the pre-registered threshold, and one seed cannot distinguish +0.50 from 0.**

This is the first time a memory mechanism has not been measurably negative, which is worth recording - and it is not evidence that memory
works. The previous two memory designs failed outright (slot bank +0.06 with collapsed slots; window bank at 1200 steps -0.03 while the model
was below trivial). This one is, for the first time, on the right side of zero.

## The only thing that would settle it

The same comparison at **3 seeds**, same budget, and scoring the paired per-seed difference rather than comparing arm means. If the mean
paired delta stays near +0.50 with a confidence interval that excludes zero, the window bank earns its place. If the interval straddles
zero, the honest conclusion is that a 1.68M model at 3000 steps cannot demonstrate a memory benefit and the line should stop.

Note on instrumentation: these two arms launched before the automatic SELF-CHECK was added, so their logs contain no independent re-score.
That is exactly the gap the self-check now closes for every future run.
