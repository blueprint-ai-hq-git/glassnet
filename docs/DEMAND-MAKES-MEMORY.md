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
