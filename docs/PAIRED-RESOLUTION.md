
## Paired over 200 windows the improvement is real, and 5x larger than the arm reported (2026-11-17)

The epoch-mode arm reported -0.0046 nats on 40 held-out windows. `tools/eval_resolution.py` scores the same models on **200 held-out windows, paired by window**,
which is the correct statistic because window difficulty dominates the variance.

```
shipped baseline, 200 windows: mean 6.0386   sd 3.3764   SE(40) 0.53386   SE(200) 0.23875

pass1  mean 6.0146
  paired diff -0.02394 nats | sd of per-window diff 0.11696
  bootstrap 95% interval [-0.03987, -0.00775]   RESOLVABLE (excludes 0)
  first-40-windows-only paired diff -0.00407    <- what the arm reported

pass2  mean 6.0123
  paired diff -0.02623 nats | sd of per-window diff 0.17257
  bootstrap 95% interval [-0.04997, -0.00305]   RESOLVABLE (excludes 0)
  first-40-windows-only paired diff -0.00460    <- what the arm reported
```

## Two things this settles

**1. The arm's arithmetic was right and its sample was unlucky.** Recomputing on the first 40 windows reproduces -0.00407 and -0.00460, matching the reported -0.0041
and -0.0046 to the digit. Nothing was miscalculated. Forty windows simply happened to be a subset where the effect looked five times smaller than it is.

**2. The effect is resolvable and it is larger.** On 200 paired windows the block arm improves held-out loss by **0.0240 nats** (pass 1) and **0.0262 nats** (pass 2),
and both bootstrap intervals exclude zero. So the direction I reported as "directionally positive but unresolvable" is **measurably positive**.

## Why pairing is the whole lesson

Per-window losses have a standard deviation of **3.3764 nats**. Per-window *differences* between the two models have a standard deviation of **0.11696 nats**.
Same windows, same positions, only the quantity differs.

> **Pairing removes a factor of 29 of the noise.** 3.3764 / 0.11696 = 28.9.

Most of a held-out window's loss is how hard that window is, and that is identical for both models. Comparing arm means throws that shared signal away and compares
against a scale dominated by it. Comparing paired differences cancels it, and the same 200 windows then resolve an effect 5x smaller than the unpaired SE at 40
windows (0.53386) would ever allow.

Every earlier arm in this project compared **arm means**, not paired differences. That is not a defect in those results - they were large effects that did not need the
extra resolution - but it is a defect in *method* wherever a small effect was called noise, and it is now the standing rule for this arm.

## What it still does not settle

**0.024 nats is 0.4% of the loss.** It is a real, resolvable improvement and it is small. It does not establish that depth is a win worth its 15.2M parameters: a block
that costs 51% of the parameter budget to move held-out loss by 0.4% is not yet earning its place. What it establishes is narrower and firmer:

> adding one certified block to the shipped model and training with distinct windows per pass improves held-out loss by **0.0240-0.0262 nats**, resolvable at 200 paired
> windows, where the same arm read as noise at 40.

The depth question is now a question about *magnitude*, not existence, which is a much better place to be. The next arm should ask whether that gain grows with the
training it gets, since the shipped model needed roughly 17,000 steps and this arm has had two passes.
