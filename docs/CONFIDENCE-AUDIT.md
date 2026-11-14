# Is the model's confidence real? Measured, not assumed (item 6, first pass)

Question in his words: "why the hell is the model confident - check how it gets all of that power". Before explaining
WHY, the prior question is whether the confidence carries information at all. Measured on `exposure_d1024_e9_s0_best`
(d1024, 9 epochs), 20,000 held-out windows, 14,145 assistant-masked positions, computed from the checkpoint's own logits.
Tool: `tools/confidence_audit.py` (AUC and ECE implemented from scratch, no sklearn).

## The numbers

```
accuracy                    29.34%
mean confidence             43.02%
gap                        +13.68 points     (over-confident, not under)
ECE                          0.1368           (0.00 = perfect calibration)
AUC (confidence -> correct)  0.8069           (0.50 = confidence is noise)
wrong answers: n=9,995, mean confidence 33.49%
```

Confidence bins, which is where the shape shows:

```
confidence bin        n      mean conf   actual acc
  0.00 - 0.10       768        7.64%        4.95%
  0.10 - 0.20      2876       15.10%        8.94%
  0.20 - 0.30      2451       24.76%       13.99%
  0.30 - 0.40      1854       34.69%       20.93%
  0.40 - 0.50      1395       44.69%       25.38%
  0.50 - 0.60      1015       54.92%       31.82%
  0.60 - 0.70       808       64.83%        40.59%
  0.70 - 0.80       731       75.06%        48.70%
  0.80 - 0.90       755       85.16%        63.31%
  0.90 - 1.00      1492       96.86%        86.13%
```

## Predictions, scored honestly

- **P1 "ECE > 0.15: badly over-confident" - FAILED** at 0.1368. It IS over-confident (+13.68 points) but just under the
  threshold I set. I predicted a failure mode that is only half present.
- **P2 "AUC >= 0.65: confidence still ranks correct above wrong" - HOLDS** at 0.8069, well above the bar.
- **P3 "wrong answers carry > 50% confidence: confidently wrong" - FAILED** at 33.49%. Wrong answers are **quiet**, not
  loud. The model already signals its own uncertainty; what it does badly is report an accurate number.

## What this actually says about the question

The confidence is **real and ordered**: sorted by confidence, accuracy climbs monotonically from 4.95% to 86.13%, and a
confident prediction is far more likely to be correct (AUC 0.81). So "why is it confident" is NOT "because it always says
so" - the model knows when it is on thin ice. The defect is a **uniform inflation of about 14 points**: when it feels 43%
right it is 29% right. That is a calibration fault, which is fixable by post-hoc scaling or a temperature, not by
re-architecting the model.

Two things this does not yet answer, and they are the next steps for item 6: WHERE the confidence comes from inside the
network (which concepts/rows drive it), and whether it can be corrected without losing accuracy.
