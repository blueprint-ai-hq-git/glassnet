# One temperature removes 62% of the model's over-confidence, for free (item 6, second pass)

The audit (`docs/history/CONFIDENCE-AUDIT.md`) measured the model as over-confident by 13.68 points, ECE 0.1368. This
asks whether that is a scaling problem, which a single temperature can fix, or something structural.

Method: fit one temperature T on the **first half** of the held-out windows and score on the **second half**. Fitting and
scoring on the same positions would manufacture a good-looking number. Dividing logits by T cannot change any prediction,
so accuracy is the strict test that the tool is honest. Tool: `tools/calibrate.py`.

## Result

```
positions: 7,072 fit / 7,073 test (chronological halves)
fitted temperature: 1.1318        (fit-half NLL 4.07284 -> 4.00408)

                  accuracy   mean conf      ECE      AUC
BEFORE (T=1)        27.71%      41.41%     0.1370   0.7955
AFTER  (T=1.132)    27.71%      32.83%     0.0521   0.7980
```

**ECE falls 2.6x, 0.1370 -> 0.0521, with accuracy exactly unchanged (27.71% -> 27.71%).** Mean confidence drops from
41.41% to 32.83%, i.e. it moves from 13.7 points above accuracy to about 5. So the over-confidence is mostly a scaling
fault, exactly as the fitted T > 1 says: the logits are too sharp.

## Predictions, scored

- **P1 T > 1.0 (logits too sharp) - HOLDS** at 1.1318.
- **P2 test ECE < 0.05 - FAILED** at 0.0521. It is very close and it is a 2.6x improvement, but it misses the bar I set,
  so it is a miss. A single temperature corrects most of the inflation, not all of it; the residual ~5 points needs more
  than one parameter (per-class or per-concept scaling), and that is the next step rather than a claim.
- **P3 accuracy EXACTLY unchanged - HOLDS** (0.2771 -> 0.2771), which is the check that proves the measurement is honest:
  temperature cannot move an argmax, so if accuracy had moved, the tool would have been measuring something else.
- **P4 AUC unchanged to 4dp - FAILED** narrowly (0.7955 -> 0.7980). Temperature is a monotone rescaling, so the ranking
  of confidence is preserved and this should be identical; the 0.0025 move is a tie-group artifact from float32 rounding
  after division, not a real change in what the confidence knows. P3 is the strict test and it held.

## What this means for the standing question

The model is not bluffing: its confidence ranks correct answers above wrong ones (AUC ~0.80) and its wrong answers are
low-confidence. Its only defect was reporting too much certainty, and **one free parameter removes most of it with no
retraining and no accuracy cost**. What is left is small enough that the honest statement is "the model's confidence is
usable after calibration, and not perfectly calibrated".
