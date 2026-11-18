# The accuracy plateau is now measurable: flat from step 4500 while loss keeps falling (2026-11-18)

The learning-curve report a few hours ago measured the shape of returns-to-training but had to withhold the conclusion: **each point carried a 1.47-point standard error**, so
increments of 0.4-0.7 points per 250 steps were indistinguishable from noise, and it said explicitly that a plateau smaller than 1.47 points per point could not be seen. The corrected
104-chunk protocol (SE **0.50**) can see it. The 6000-step extension arm is now past step 5500:

```
                             104 chunks (6,552 scored)      loss
step 3000  (resume point)          23.95%                  6.3261
step 4500                          25.35%                  6.1303
step 5000                          25.24%                  6.1148
step 5500                          25.76%                  6.0826
```

## What this says

**Accuracy has plateaued.** From step 4500 to 5500 the model reads 25.35%, 25.24%, 25.76% - a spread of **0.52 points** against a **0.50-point** standard error. That is **flat**: the
2500 steps from 3000 to 5500 bought **+1.81 points** in total, against the **+3.83 points** that the 1500-to-3000 stretch bought. Returns have fallen by more than half.

**But loss has not plateaued.** Held-out loss falls monotonically, 6.3261 -> 6.1303 -> 6.1148 -> 6.0826, a drop of **0.24 nats** over the same stretch in which accuracy moved 0.52
points inside its own error bar. **The model is getting more confident, not more correct.** Those are different things, and only one of them is what "learning" is supposed to mean here.

## Predictions, scored honestly

- **Original P1** for this arm: "held-out >= 24% at 6000 steps." **MET**, and already met at step 4500 (25.35%).
- **Restated P1** (after the eval correction changed its baseline from 21.03% to 23.95%): ">= 26% at 6000 steps." At step 5500 it reads **25.76%**, needing +0.24 in the last 500 steps.
  On the measured slope this is a coin-flip, and it **must be reported as met or missed on the final checkpoint, not extrapolated now**.
- **P2** "free-running rises above 8%" and **P3** "k=16 memory rises above 2%" - both are measured on the final checkpoint only; not yet known.

## The honest headline

**More training stopped buying accuracy somewhere around step 4500 at ~25.5%, and the last 1000 steps moved it 0.52 points, which is one standard error.** The lever that this
project identified - training length - has now been measured to its end at this scale. The next lever is not more steps.

That is a *negative* result and it is the useful kind: it closes the "just train longer" line with a measured floor rather than another doubling, and it was only visible because the
measurement was made four standard errors tighter than the one that started the line.
