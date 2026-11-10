# Run 8 final verdict: E1 pass, E2 fail, E3 pass, E4 fail (2026-11-10)

Run 8 = direct path + constant decorrelation penalty (lambda 0.5), 4,735,714 params, batch 32, 1500 steps,
held-out loss **8.8631**, accuracy **5.60%**, train/held-out gap **+0.2994**.

All four criteria as written before the run:

```
E1 state cosine below 0.5     : 0.4643   PASS (predicted ~0.47 at 14:02, measured 0.4643)
E2 KL above 0.05 nats         : 0.0151   FAIL
E3 held-out at or above 5.34% : 5.60%    PASS (+0.26 pp over run 5)
E4 no degenerate repetition   : FAIL
G2 entropy                    : 9.373 (99.8% of uniform)
F4 trained vs random readout  : 0.0151 vs 0.0153 = 0.99x   FAIL
```

## What each verdict means
- **E1 PASS, and the drift prediction was right.** At 14:02 I extrapolated the cosine trend at 0.00030 per
  step and predicted ~0.47 at 1500 steps. Measured: **0.4643**. The decorrelation penalty damps the collapse
  but does not stop it; it is still climbing linearly and would fail a longer run. That is why run 10 replaces
  the constant pull with a quadratic barrier.
- **E2 FAIL.** KL 0.0151, unchanged from run 5's 0.0102 and from every intermediate checkpoint. Longer
  training, decorrelated states and a direct path all leave the output distribution nearly input-independent.
- **E3 PASS, marginally.** 5.60% against run 5's 5.34%, which is +0.26 pp: a small gain that should be read as
  inside the run-to-run spread until it is repeated on other seeds. Not a claim of improvement.
- **E4 FAIL, and worse than the smoke test.** Greedy generation collapses to
  `. . the , the the the the of the happen of the a the` for every prompt, while the same code at 12 steps had
  produced four distinct strings. Training longer produced a repetition loop at the same time as the state
  differences grew, which is what mode collapse toward the most frequent tokens looks like.
- **F4 FAIL.** The trained readout still scores the same as random numbers at the same scale (0.99x), so the
  answer layer continues to throw the state away.

## Honest summary
Run 8 moved one thing it was designed to move (state separation) and left the readout exactly where it was.
The accuracy tick is too small to call. Run 9 tests whether our own readout normalisation changes that; if it
does not, the next suspect is the 12,002-way target itself rather than any of the state levers, all of which
have now been measured.
