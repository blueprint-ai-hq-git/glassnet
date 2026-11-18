# 16-step demand on real text: the bank converts it, and the effect is small but real (2026-11-18)

Two arms, both resumed from the same `scan12_r090_6k_ep6000` checkpoint, both trained 1000 steps with the target at position t being GIVEN BY `tok[t+1-16]` instead of `tok[t+1]`.
They differ in exactly one thing: whether the 8-slot memory bank is on.

```
                          delayed acc   next-token   same-sample constant
td16   (demand + bank)       8.67%       16.54%        6.59%
td16nb (demand, no bank)     5.22%       14.32%        6.59%
mem8sym (no demand, bank)    7.02%       23.50%        6.59%
6552 scored positions, 104 held-out chunks
```

## Predictions, scored at the endpoint

- **Bank arm above a trivial predictor?** **YES**: **8.67%** against a **6.59%** same-sample constant baseline, **+2.08 points**, roughly **6 SE** -> resolvable.
- **Falsifier** ("k=16 memory at or below 2.214% means demand is not the limiter") - it **did not fire on the valid measure**. On the next-token probe it *appears* to fire (the probe reads ~1.1%), but that probe is the wrong instrument here and the reason is below.
- **Does the bank earn its keep?** **YES**: **8.67% vs 5.22%**, **+3.45 points** (~7 SE). The bankless arm is **below** the trivial baseline, the banked arm is **above** it.
- **Does demand itself add anything?** **YES, weakly**: the no-demand bank arm scores **7.02%** on the same delayed measure, so demand adds **+1.65 points** (~3 SE) on top of the bank.

## The instrument error that hid this, recorded so it is not repeated

For several wakes this arm was reported as **~1.1% k=16 memory**, apparently a failure. The cause was mine: `tools/eval_radius_arm.py` scores the model's **next-token** logits, and these arms were trained to emit
`tok[t+1-16]`. `td16` deliberately trades next-token accuracy away (16.54% next-token versus `mem8sym`'s 23.50%), so a next-token probe reads ~1.1% while the arm's own task reads 8.67%.

`tools/eval_delayed.py` now scores the offset the arm was actually trained on. It also compares against a **same-sample constant baseline** rather than chance (1/12002 = 0.008%, which would make 5.22% look
like success) or the corpus-wide 5.3406% (which is not the right pairing for a specific sample).

## What this is and is not

- **Is:** the first measurable, above-baseline 16-step recall on real text in this project, and evidence that the memory bank is what enables it.
- **Is not:** "memory solved". 8.67% is small, the gate is 10% **and that gate is scored by the next-token probe which these arms were not trained for**, so it does not apply unchanged to them. One seed, one arm
  each, 1000 steps.
- **Still open:** whether delayed recall climbs with more training, and whether a smaller delay gives a cleaner dose-response.


---

# CORRECTION AND ADDITION (2026-11-18, same day): the interval was overstated and one baseline was missing

The entry above quotes "+2.08 points, roughly 6 SE". That standard error treated all 6552 scored positions as independent. **They are not.** The 64 scored positions inside one window
share a recurrent state and a memory-bank state, so they are correlated, and the unit of resampling is the **window** - of which there are only 104. Re-measured with a window-level
bootstrap (10,000 resamples, `tools/eval_delayed_ci.py`):

```
vs the FAIR constant predictor (always emit the most frequent corpus token, 6.59%):
  td16   (demand + bank)   8.67%   +2.08 points   95% CI [+0.93, +3.27]   RESOLVABLE (above)
  td16nb (demand, no bank) 5.22%   -1.37 points   95% CI [-2.24, -0.44]   RESOLVABLE (below)

vs the WINDOW-ORACLE constant predictor (most frequent target inside each window, 14.41%):
  td16                     8.67%   -5.74 points   95% CI [-6.75, -4.81]   RESOLVABLE (below)
  td16nb                   5.22%   -9.19 points   95% CI [-10.30, -8.06]  RESOLVABLE (below)

bank vs no-bank:
  +3.45 points   95% CI [+2.43, +4.43]                                      RESOLVABLE
```

**What stands:** the bank arm is above the fair constant predictor, the bankless arm is below it, and the bank is worth **+3.45 points** with an interval that excludes zero. Those are the
claims this entry rests on, and they survive cluster-robust resampling.

**What was wrong:** "roughly 6 SE" - the true spread is about **3.5 SE**, because the effective sample size is 104 windows, not 6552 positions. A claim that is resolvable at 3.5 SE is
still resolvable, but the margin I published was inflated by a factor of about 1.7.

**What was missing, and it cuts against the result:** a **window-oracle** constant predictor scores **14.41%**. It sees the whole window including the future, so no causal model can
implement it and it is not a fair competitor - but it is the right reference for how much structure the targets contain. Against it, both arms are **far below**: the banked arm by 5.74
points. So "the bank gives measurable 16-step recall" is true against a fair baseline, and simultaneously "this recall is far weaker than the per-window target concentration allows" is
also true. Both belong on the page.

The honest one-line summary is now: **demand training with the memory bank lifts 16-step recall above a fair constant predictor (+2.08 points, CI [+0.93, +3.27]) and the bank is worth
+3.45 points over no bank (CI [+2.43, +4.43]), while both arms remain well below a window-informed constant predictor (14.41%).**
