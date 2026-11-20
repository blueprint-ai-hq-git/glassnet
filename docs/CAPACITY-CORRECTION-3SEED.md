

---

# CORRECTION: THREE SEEDS REFUTE THE CAPACITY SIGNAL I REPORTED ONE HOUR AGO (2026-11-20 20:13)

One hour ago I reported that capacity had finally moved: a 30%-table config beating the 85%-table default while 29% smaller, with the
accuracy winner at +0.58. **Three paired seeds say that was a single-seed artefact, and I am correcting it in the same place it was
published.**

## The three seeds, paired

```
seed   A default   B (30% table)   C (accuracy winner)   C - A     B - A
 0      10.29%        10.74%           10.87%           +0.58     +0.45
 1      10.88%         7.59%           10.73%           -0.15     -3.29
 2      10.72%        10.73%           11.09%           +0.37     +0.01
mean                                                  +0.27     -0.94
sd                                                      0.31      1.67
```

- **P1 NOT MET:** mean C-A is **+0.27** against a **+0.5** bar, and its sd is **0.31** - the gain is inside one standard deviation.
- **P2 FAILED:** mean B-A is **-0.94**. The "29% smaller and still better" claim was **seed 0 being lucky**; on seed 1 the same config
  lands **3.29 points below** the default.
- **P3 MET:** every arm of every seed beat the 4.08% matched constant, so nothing failed to train.
- **Falsifier did NOT fire**, by **0.07** (it needed mean C-A below +0.2, and the mean is +0.27).

## What is actually true, stated plainly

1. **Capacity did not move.** The 30%-table config is not reliably better than the default and is **unstable** - its sd on the paired
   delta is **1.67**, five times the default's spread on the same comparison.
2. **The accuracy-winning config gives no reliable gain**: +0.27 +- 0.31 is indistinguishable from zero at this budget.
3. **It is not a training failure:** B on seed 1 fell from loss 9.41 to 6.26 and reached 7.59% - it simply ended worse. So this is
   genuine seed sensitivity of the rank-8 / 3-layer shape, not divergence.

My one-hour-ago message said "capacity finally moves". **That sentence was wrong and three seeds are what caught it** - which is
exactly why the project rule demands 3 seeds before a claim, and why the rule exists even when a single seed looks exciting.

## What this changes about the plan

The allocation line is not closed by its falsifier's letter, but the substance is clear: at 300 steps there is no reliable allocation
win to build on, and the smaller-table shape is too unstable to recommend. So the honest next move is **not** another allocation grid -
it is the third thing asked for, which is still untouched: **memorise versus generalise**, built with a trap dataset that separates the
two by construction.
