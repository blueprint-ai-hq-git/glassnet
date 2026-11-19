

---

# CALIBRATION IS NOT THE LEVER: LOGIT ADJUSTMENT GAINS +0.84 ON ONE CHECKPOINT AND NOTHING ON ANOTHER (2026-11-19 19:35)

The mid-band diagnostic said the failure is frequency bias - the model puts the correct token at median rank 34 and then emits a top-20 token 74.48% of the
time. The textbook correction is logit adjustment: `logits - tau*log(prior)`, argmax. Inference-time only, no retraining. Run on 104 windows it looked like a
clean win:

```
LOGIT ADJUSTMENT  win6koff_K0_ep6000.npz   104 windows
  tau    top20     r21-500   r501-2k   tail      OVERALL
  0.00   36.15%    8.13%     0.00%     0.00%     20.30%
  0.10   36.53%    8.34%     0.00%     0.00%     20.56%
  0.20   36.90%    8.81%     0.00%     0.00%     20.92%
  0.35   36.70%    9.67%     0.00%     0.00%     21.14%     <- peak, +0.84 points
  0.50   33.97%    9.76%     0.00%     0.00%     19.88%
  0.75   10.57%    1.71%     0.00%     0.00%      5.68%
  1.00    1.74%    0.00%     0.00%     0.00%      0.83%

P1 overall >= 0.5 : MET (+0.84)   P2 mid gains more than top-20 loses : MET (+1.54 vs +0.55)
P3 monotone to a peak then reverses : MET (peak at tau 0.35)   FALSIFIER: did not fire
```

All three pre-registered predictions met, with the curve shape I asked for. So I tested two more checkpoints before publishing it:

```
win6kon_K8_ep6000 (same family, K=8)   best tau 0.35 -> +0.49 overall (P1 NOT met)  FALSIFIER FIRED
                                     at that tau: top20 -0.31, r21-500 +1.67
scan12_r090_6k_ep6000 (UNTIED table)   best tau 0.00 -> +0.00 overall (P1 NOT met)  FALSIFIER FIRED
```

**The falsifier fired on 2 of the 3 checkpoints.** The +0.84 does not replicate, and on the untied model the optimum is `tau = 0` - calibration buys exactly
nothing there, which makes sense if an untied answer table already absorbs frequency in its row norms, so there is nothing left for the correction to undo.

## What survives, and it is worth keeping

1. **The DIAGNOSIS survives.** The mid band really is frequency-biased: the correct token sits at median rank 34 and is inside the model's own top-100 on
   66.67% of mid-band positions. That measurement is unaffected by the fix failing.
2. **The mid band does move**, on every checkpoint tested: **+1.54** and **+1.67** points on r21-500. The premise was right; the correction is just too blunt
   to convert that into overall gain without costing the top-20 band on two of three runs.
3. **It is cheap to be wrong here.** The whole test took minutes and no training, which is exactly why the diagnostic was worth running before spending hours
   on capacity.

## The rule this re-confirms

**A pre-registered falsifier is only worth writing if it is evaluated on more than the checkpoint that motivated it.** Had I stopped at the first table I
would have published "+0.84 free points" from a single file - the same error shape as this morning's single-seed +0.50 that had to be withdrawn.

## Next

The mid band needs something better aimed than a global prior correction. Since the mid band gains +1.5-1.7 points under tau but the top-20 band pays for it,
the obvious next question is whether the correction should be applied **only where the model's own top-100 contains a mid-frequency candidate**, or with a
per-band temperature - and both are measurable with the same instrument in minutes. What is now established is that **capacity is not the first thing to fix**.
