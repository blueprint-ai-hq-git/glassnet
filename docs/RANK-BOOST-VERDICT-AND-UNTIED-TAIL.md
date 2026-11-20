

---

# RANK-BOOST VERDICT: THE FALSIFIER FIRED ON ALL THREE PREDICTIONS - AND THE UNTIED READOUT IS THE FIRST THING TO EVER MOVE THE TAIL (2026-11-20 09:52)

## Part 1 - the rank-boost arm failed, decisively

`boost50` (4x loss weight on targets ranked >= 50, everything else identical to `win6kon`) at its pre-registered 6000-step endpoint:

```
                          win6kon (control)   boost50      delta
ranks 20-50               25.19%              0.00%        -25.19
ranks 50-100               0.22%              1.33%        +1.11
ranks 100-250              0.00%              0.00%         0.00
ranks 250-500              0.00%              0.00%         0.00
r21-500 overall            8.52%              0.26%        -8.26
top-20 accuracy           35.81%             24.22%       -11.59
trainer line vs constant  22.88% (BEATS)     14.23% vs 14.47% = -0.24 -> BELOW
```

- **P1 FAILED**: the band did not rise. It fell 8.26 points, because ranks 20-50 - the only sub-band that ever worked - went to **exactly zero**.
- **P2 FAILED**: overall fell **11.59 points** against a 2-point allowance.
- **P3 FAILED**: on the trainer's own matched constant the arm scores **below a trivial predictor** (14.23% vs 14.47%).
- **The falsifier fired exactly as written**, and this is the outcome it was written to produce: weighting the loss cannot teach the long tail, so the
  limiter is the **readout**.

The mechanism is visible in the numbers: 4x weight on **half of all training positions** starved the head so completely that the model never became
better than a frequency table. It bought +1.11 points in the 50-100 band at the cost of everything else - and even that band only reached 1.33%.

## Part 2 - the untied readout moves the tail, which nothing else has

Prediction written **before** the measurement: *if the limiter is readout capacity, the untied 6000-step checkpoint must show ranks 50-500 above the
0.22% tied baseline.* Measured on `scan12_r090_6k_ep6000` (untied answer table, 6000 steps, 104 windows):

```
                          tied control   UNTIED      delta
ranks 20-50               25.19%         35.24%      +10.05
ranks 50-100               0.22%          2.66%      +2.44   (12x the tied rate)
ranks 100-250              0.00%          0.29%      +0.29   (from exactly zero)
ranks 250-500              0.00%          0.00%       0.00
r21-500 overall            8.52%         12.45%      +3.93
top-20 accuracy           35.81%         41.61%      +5.80
top-20 default rate        94.90%         70.73%     -24.17  (far less head-biased)
```

**Prediction MET.** This is the first configuration in the entire memory/readout line where the mid band moves up *and* the head improves at the same
time. The frequency bias itself drops by 24 points.

## The confound, stated before anyone quotes it

`scan12_r090_6k` is **not** a clean one-change control: its readout is untied AND it carries ~3.07M answer-table parameters against the tied arm's 32,768
bottleneck, so it has ~2.8x the parameters and a different (rarity-weighted) training setting. So this shows **capacity in the readout is sufficient to
move the tail**, not that tying alone was the cause.

## Launched: the clean one-change test

`untied6k` - `win6kon`'s exact config with **TIE=0 and nothing else changed** (K=8 window bank, seed 0, batch 16, 6000 steps, no rank boost). Predictions
written before launch:

- **P1** ranks 50-100 above the tied baseline **0.22%** - capacity reaches the first rare band
- **P2** top-20 accuracy does not fall below **35.81%** - the tail must not be bought with the head
- **P3** overall `<unk>`-excluded **at or above 21.88%** on the trainer line (no more than 1 point below the tied control's 22.88%)
- **FALSIFIER:** ranks 50-100 at or below 0.22% -> the scan12 result was its config and parameter count, not the readout, capacity is not the lever
  either, and this line closes rather than being refined again.
