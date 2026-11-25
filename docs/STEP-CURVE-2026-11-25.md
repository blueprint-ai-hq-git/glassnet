# 2026-11-25 - step curve: the training collapse was MY sampler bug, and the curve still pays at 6,400 steps

Source: lens/runs/STEP-CURVE.out, lens/runs/STEP-CURVE.json, lens/runs/LR-STAB.out. Backend: verified cupy path.

## What was claimed first, and why it was wrong

First run (lr 0.05) appeared to show the model DIVERGING after 3,200 steps:

    200   5.99601 | 400 5.64017 | 800 5.41846 | 1600 5.34663 | 3200 5.27488 | 6400 7.23145

Sweeping the learning rate (0.05 / 0.02 / 0.01) showed the same collapse at every rate (6400: 8.65 / 5.50 / 5.43), which
comments suggested was optimiser instability. It was not. The bug was in the experiment:

    sel = np.arange((done % 1000) * B, (done % 1000) * B + B)

That cycles the SAME 1,000 windows forever. After ~3,200 steps each window had been seen about 25 times, so the model was
memorising them and held-out loss rose. The instrument was measuring overfitting on 1,000 windows, not step-count scaling.
This is the EIGHTH instrument bug in the project's history, and it was caught only because the numbers disagreed with three
independent learning rates.

## Fixed run (epoch shuffle over all 20,000 train windows, so nothing is over-sampled)

    steps     held-out    gain vs previous
      200     6.17211
      400     5.56575     +0.60636
      800     5.41231     +0.15344
     1600     5.46428     -0.05197   (noise blip)
     3200     5.25623     +0.20805
     6400     5.13029     +0.12594

No collapse at any length. Gains are shrinking (0.208 for 1600->3200, 0.126 for 3200->6400) but were still paying at 6,400,
so the curve is longer than this test.

## Verdict, stated honestly

  P1 monotonic fall             NOT MET (one noise blip at 1600)
  P2 flattens by half after 3200 NOT MET (still paying, 0.126)
  P3 <= 4.90 at 6400            NOT MET (5.13029)
The earlier "STEPS_ALONE_WILL_NOT_DO_IT" conclusion is WITHDRAWN: it was produced by the sampler bug. The correct statement
is that steps were still paying at 6,400 and the next test must extend to 25,600 before any budget claim is made.

## Cost, which is the real news

6,400 steps took 10.3 s on the verified cupy path (624 steps/s). Before the port the same work was minutes. At this price
the next curve point (25,600 steps) is about 40 s, so budget questions are now cheap to answer instead of guesswork.
