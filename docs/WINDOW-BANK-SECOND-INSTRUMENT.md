

---

# THE WINDOW BANK EFFECT SHRINKS 84% ON A SECOND INSTRUMENT: +0.87 BECOMES +0.14 (2026-11-19 16:29)

The three-seed paired result said +0.87 points on the trainer's line. A delta that lives on only one instrument is not a delta, so all six checkpoints were re-scored with
`tools/eval_any.py` on the same draw. The `<unk>`-excluded model accuracy, which is the matched denominator:

```
              trainer line        independent instrument
seed 0        14.75 / 15.25       12.97 / 12.80        paired delta -0.17
seed 1        16.03 / 17.11       13.88 / 14.01        paired delta +0.13
seed 2        10.50 / 11.52        9.95 / 10.41        paired delta +0.46

mean paired delta   trainer +0.87        independent +0.14
independent sd 0.32, se 0.18, 95% CI [-0.65, +0.94], t = 0.74
```

- **P1 MET** (sign holds on 2 of 3 seeds - not 3 of 3).
- **P2 MET, barely** (mean +0.14).
- **Falsifier did NOT fire** (it required a negative mean or only 1 of 3 keeping the sign).
- **But the magnitude collapses by 84%**, and the independent 95% interval is **[-0.65, +0.94]**, which contains zero comfortably (t = 0.74, against a 4.303
  critical value).

## What the honest conclusion is

**The +0.87 was mostly instrument effect.** Two facts make that the only defensible reading:

1. The trainer's own line reads **~2 points higher than the independent tool on the same files** (14.75 vs 12.97, 16.03 vs 13.88, 10.50 vs 9.95), and that gap
   is a protocol difference that has still not been explained - I measured the draw noise at 0.41 points earlier today, so a 2-point gap is not sampling.
2. Whichever instrument is used, the effect is **far below that instrument's own resolution** once the protocols are matched.

So the sequence of this claim, recorded plainly: **+0.50 (one seed, arm means) -> withdrawn -> +0.87 (three paired seeds, one instrument, t 4.71) -> +0.14 (three
paired seeds, second instrument, t 0.74)**. The direction survives; the size does not. What can be said without overreach: **adding the readable window does not
measurably HURT the language objective at 3000 steps** - it is not the negative result the earlier arms suggested - and no positive claim is warranted.

## Also learned from the re-score

**One of three seeds clears the trivial constant at 3000 steps** (seed 1: 13.88 and 14.01 against 13.14), while seeds 0 and 2 do not (12.97/12.80 and
9.95/10.41). So the 3000-step budget is not uniformly "below trivial" as I wrote earlier - it is **seed-dependent**, which is a further reason this budget
cannot carry a mechanism claim.

## The unexplained protocol gap is now the top instrument problem

A 2-point difference between the trainer's held-out line and the independent evaluator, on the same checkpoint and the same window count, is larger than every
effect this project has measured today. Either the trainer is scoring something subtly different (a normalisation, the gain term, or the exact positions) or the
evaluator is. Until that is resolved, every delta computed on one instrument carries a ~2-point unknown.
