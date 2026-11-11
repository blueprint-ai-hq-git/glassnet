# First RL ladder: reward shaping does kill repetition, and then it kills the model (2026-11-11)

All three arms start from `matched_d512_s0_best.npz`, same harness, same 2,000 steps, batch 256, lr 0.05, one decision
per row. Predictions were written before the run (`RL-PREDICTIONS-2026-11-11.md`).

```
arm   objective                                        argmax acc      token-in-window      ECE
ce    ordinary supervised training (CONTROL)      19.42% -> 24.17%    36.8% -> 31.0%    0.294 -> 0.287
rl1   reward = 1 if sampled token correct         19.42% -> 20.17%    36.8% -> 57.4%    0.294 -> 0.535
rl2   rl1 minus a charge for re-using window words 19.42% -> 6.25%    36.8% ->  0.6%    0.294 -> 0.920
```

## What the numbers say
**P1 falsified.** I predicted rl1 within 2 pp of the control. The control gained **+4.75 pp** and rl1 gained **+0.75**,
so reward-only training is **4.0 pp behind ordinary training** at this budget, and it also made the model *more*
repetitive (36.8% -> 57.4%) and much less calibrated (0.294 -> 0.535). The honest reading: the score-function estimator
is high variance, and 2,000 steps of it does not buy what 2,000 steps of supervised training buys. The gradient
direction is right (cosine 1.0000 was measured), the sample efficiency is not.

**P2 passed, spectacularly.** The repetition charge cut window-echoing from 36.8% to **0.6%** against the control's
31.0%. Nothing in four supervised attempts ever moved that number like this.

**P3 falsified just as spectacularly.** The price was accuracy collapsing from 24.17% (control) to **6.25%**, an
18 pp loss rather than the 2 pp I allowed.

## Why it collapsed, and this is the useful part
At `lam_rep = 1.0` the reward is `correct - in_window`. The base rate of picking an in-window token was about 40%, and
the base rate of being correct 12%, so **the repetition term was three times larger than the correctness term**. The
model did exactly what it was paid to do: it stopped echoing, and it stopped being right. Its confidence also
detonated (ECE 0.920 out of a possible 1.0), which is the direct answer to the maintainer's "why is the model confident"
question - in this arm the confidence is worth nothing at all.

That is reward hacking, observed in our own model, in our own trainer, in one afternoon. It is also the first time any
lever in this project has moved repetition at all, so the mechanism works; the shape of the reward is what is wrong.

## Next, one change at a time
Dose response at `lam_rep = 0.1` and `0.3`, same everything else, predictions written first:
- P4 at 0.1: accuracy within 2 pp of the control and in-window below the control's 31%.
- P5 at 0.3: accuracy within 5 pp of control, in-window at or below 20%.
- If even 0.1 collapses the model, the dose is not the problem and the next move is a correctness-gated penalty or a KL
  anchor to the starting policy, not another dose.
