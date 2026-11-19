

---

# THE RANKS-50+ ARM: THE LOSS NOW POINTS AT THE MEASURED CLIFF (2026-11-19 21:00)

The band analysis said the model is a **top-50 predictor**: ranks 21-50 at **24.05%**, ranks 50-100 at **0.22%**, ranks 100-500 at **exactly 0.00%** over 1,100
positions. Calibration was tried and closed (global and gated both failed replication, and 66% of the mid band sits at zero, so re-ranking cannot lift it). So this
arm changes the **loss during training** instead, and only for the cliff.

## The change, and how it was certified before launch

`forward_backward` takes an optional per-position weight `pos_w` that multiplies the existing `<unk>` mask **before** normalisation; the backward already scales
`dlogits` by that same normalised weight, so the gradient stays exactly consistent with the weighted loss.

```
non-uniform pos_w: 84 params checked, worst abs error 3.366e-10   -> PASS
uniform pos_w:     reproduces the unweighted loss, |diff| 0.000e+00 -> PASS (verified no-op)
```

## How big is the intervention? Bigger than expected

Measured over 193,725 scored positions using the trainer's own draws (200 steps, batch 16, seed 0):

```
rank >= 20 : 63.91%
rank >= 50 : 50.06%   <- the population the 4x boost hits
<unk>      :  5.41%   (masked, never trained)

loss mass: the boosted positions move from 50.1% -> 80.0% of total weight
```

**Half of all training positions are up-weighted 4x, moving them from half the loss to four-fifths of it.** That makes the arm a powerful test of whether the long
tail responds to loss weight at all - and it puts P2 (overall must not fall more than 2 points) genuinely at risk, because the top-20 band, which supplies nearly
all the measured accuracy, keeps only 20% of the weight.

## Predictions, written before the run

- **P1** ranks 50-500 accuracy rises above the **0.22%** baseline (it can only move if the weighting bites at all)
- **P2** overall accuracy does not fall by more than **2 points** against `win6kon`
- **P3** the arm still beats the matched constant on the trainer's own line
- **FALSIFIER:** ranks 50-500 stay at or below 0.22% -> weighting the loss cannot teach the long tail, the limiter is the **readout's capacity**, and the next
  experiment must change the readout (untied, or frequency-tiered capacity) rather than any loss setting.

## Status: measured to 2125 of 6000 steps, then stopped for the night

8 checkpoints kept (`boost50_K8_ep250` .. `ep2000`), no held-out line yet, which is correct for a mid-run stop. It resumes tomorrow.

## Side observation worth its own measurement

**Rank >= 20 is 63.91% of TRAINING positions here, but the band budget measured top-20 at 45.31% of VALIDATION positions.** The validation region is more
top-heavy than training, which inflates validation accuracy relative to the distribution the model is actually trained on. Recorded, not acted on.
