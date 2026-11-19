

---

# CALIBRATION LINE CLOSED: GATED ADJUSTMENT DOES NOT REPLICATE EITHER, AND THE MODEL HAS NO CONFIDENT SET (2026-11-19 19:50)

The global correction failed to replicate (+0.84 / +0.49 / +0.00 across three checkpoints), so the one refinement I allowed myself was to gate it on the model's
own uncertainty - apply the prior correction only when the top1-top2 margin is small, protecting confident answers. Pre-registered with replication as the bar:
a setting counts only if it clears **+0.5 points overall on ALL THREE checkpoints**.

```
margin   tau   |  win6koff_K0    win6kon_K8    scan12 (untied)
0.5      0.20  |  ov +0.62       ov +0.29      ov -0.11
0.5      0.35  |  ov +0.76       ov +0.41      ov -0.29
0.5      0.50  |  ov +0.05       ov -0.13      ov -1.17
1.0      0.35  |  ov +0.84       ov +0.49      ov -0.37
2.0      0.35  |  ov +0.84       ov +0.49      ov -0.37
4.0      0.35  |  ov +0.84       ov +0.49      ov -0.37
ungated  0.35  |  ov +0.84       ov +0.49      ov -0.37

P2 settings clearing +0.5 on ALL THREE: 0
FALSIFIER: FIRED - no gated setting replicates; calibration is not the lever and this line closes
```

## The finding inside the failure

Thresholds of **1.0, 2.0, 4.0 and ungated produce identical numbers to two decimal places.** That can only happen if **essentially every scored position has a
top1-top2 margin below 1.0** - gating at 1.0 or higher is the same as not gating at all. So there is no confident set to protect: **the model is uniformly
uncertain across positions**, which is why gating cannot recover the top-20 band.

Tightening to 0.5 does change the numbers - and makes them **worse** (+0.84 -> +0.76, +0.49 -> +0.41), because the positions it protects were the ones the
correction was helping least.

## What the mid band needs instead

Calibration is closed by pre-registration, not by opinion: the falsifier said "no gated setting clears +0.5 on all three -> calibration is not the lever and
this line closes rather than being refined a third time". Three refinements were allowed by the plan (global, cross-checkpoint, gated); all three are measured
and none replicates.

Two facts now constrain the next attempt:
1. The mid band **does** gain +0.47 to +1.67 points under a prior correction on every checkpoint - so the frequency bias is real and exploitable.
2. The **top-20 band pays more than the mid band gains** on the untied model, and the model has no confident subset to spare.

That points away from post-hoc re-ranking and toward something that changes the readout's behaviour **during** training - but the honest next step is a
different diagnostic first: which specific mid-frequency tokens does the model get right when it does get them (are they the more frequent half of the band?),
because that determines whether the band should be split rather than treated as one 480-token block.
