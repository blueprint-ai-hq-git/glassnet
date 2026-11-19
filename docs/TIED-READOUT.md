

---

# TIED READOUT vs SEPARATE ANSWER TABLE, measured (2026-11-19 10:03)

The audit said the H x V answer table was 62.3% of the model. This is the head-to-head. Same trainer, same seed, same data order,
same 600 steps, same batch; ONE thing differs - whether the answer comes from a dedicated H x V table or from the token
embeddings themselves through a 32,768-parameter bottleneck.

```
arm        params        ep600 HELD-OUT     ep500 overall   ep500 top-20   train time
untied     4,735,970     5.27% (loss 6.3339)     5.10%          10.04%      3599 s
tied       1,684,224     4.93% (loss 6.9018)     5.71%          11.24%      1848 s
```

## The honest reading: the difference is NOT resolvable

At ep600 the tied model is 0.34 points behind. At ep500 it is **0.61 points ahead**. The sign flips between two checkpoints of the
same run, at a budget where both models are far below the 14.70% trivial same-window constant. So the correct statement is:

**At 600 steps, a 64%-smaller tied model is statistically indistinguishable from the untied one.** The separate answer table bought
nothing measurable here, and cost 3,051,746 parameters and 1.95x the training time.

## Predictions, scored

- **P1** "tied reaches >= 90% of untied accuracy" - **MET**: 4.93/5.27 = **93.5%**.
- **P2** "the frequency bands are close" - **MET**: both are 0.00% on ranks 21-500, 501-2000 and the tail at this budget; the
  frequent band is where all the accuracy lives (10.04% vs 11.24%).
- **P3** "tied trains faster" - **MET**: 1848 s vs 3599 s for 600 steps, **1.95x**.
- **Falsifier** "tied below 80% of untied" - **did not fire**.

## What this does NOT establish

Both arms are under-trained (5% against a 14.70% trivial constant), so this tests the SHAPE at a small budget, not the final
quality of either model. The claim to carry forward is "the answer table is redundant at this scale and budget", not "tied is
better". A longer run is needed before any superiority claim, and the ep600 weights did not exist until the driver was fixed
(see below), so ep600 can only be compared through its logged held-out line, not re-scored from disk.

## Two defects found while measuring this

1. **No final checkpoint.** The driver saved on multiples of 250, and these arms ran to 600 - so the graded endpoint left **no
   weights on disk**. The reported 5.27% and 4.93% had nothing behind them to re-check. The driver now saves at the last step as
   well. (ep250 and ep500 for both arms did survive, which is what made the ep500 comparison possible.)
2. **The band evaluator only knew the untied readout** and raised `KeyError: 'Wa'` on a tied checkpoint - it would have silently
   excluded the very model the repair produced. It now understands both readouts.
