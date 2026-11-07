# 8M windows ungated - predictions written BEFORE the run (2026-11-07)

Measured context: ungated ladder 500k 22.05 -> 2M 23.12 -> 4M 23.74 (per doubling +0.54, +0.62, not decaying).
4M with 6 epochs reached 23.92, so doubling steps bought +0.18 while doubling data bought +0.62.

Setup: identical to the 4M ungated arm except --max-windows 4,000,000 -> 8,000,000. One knob.

Predictions:
- W1: best VAL strictly above 23.74 (the 3-epoch 4M arm). At or below means the ungated curve saturates at 4M.
- W2: lands 24.0-25.2 pp, point estimate 24.36 from the measured 0.62 pp/doubling.
- W3: ppl below 95.99.
- W4: zero gate lines in the log (the arm must actually be ungated).
- W5: concepts at least 36 of 36 kept.
Falsifiers are the same lines read backwards. No verdict until epoch 3.
