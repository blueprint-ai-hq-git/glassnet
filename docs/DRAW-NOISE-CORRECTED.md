

---

# CORRECTION: DRAW NOISE IS 0.93 POINTS, NOT 0.41 - AND IT EXPLAINS THE 2-POINT INSTRUMENT GAP (2026-11-19 17:04)

This morning I published a draw-noise figure of **0.41 points** from five draws and used it as the project's "resolution floor". Ten draws on a different
checkpoint now give **0.93**, and the ten-draw test also prints each seed's first window index so an unapplied seed would be visible rather than inferred - all
ten indices differ, so the seeds are definitely being applied and this is real variance.

```
seed   overall     first_window
4242   23.06%      293962348
7      22.28%      294197419
13     21.47%      294172950
21     22.01%      293875757
34     24.24%      293756555
55     21.57%      294206767
89     22.37%      294154201
144    22.51%      293881804
233    23.73%      293774664
377    23.50%      294126831

mean 22.67%   sd 0.93   range 2.77 points (21.47 to 24.24)
```

## Why the two measurements differ, and why the earlier one was wrong to publish as a constant

The 0.41 came from a **12%-accuracy** checkpoint; the 0.93 comes from a **23%-accuracy** one. Draw noise is not a property of the tool, it is a property of
**the checkpoint's position on the accuracy curve**: the same 104 windows sample a binomial, so the variance scales with `p(1-p)`, and low-accuracy models are
also more homogeneous in what they get right. Publishing one number as "the instrument's resolution floor" was a mistake - it is a function of what is being
measured.

## What this explains

**The 2-point trainer-vs-evaluator gap is now fully explained.** `tools/instrument_agreement.py` showed the two forward passes agree to **+0.06 points on
identical windows**, so the gap is window selection - and the range across ten draws is **2.77 points**, which comfortably contains a 2-point difference between
two different draws (the trainer uses `rng(999)`, the evaluator `rng(4242)`).

## The important distinction this clarifies: paired vs unpaired

- **Unpaired single-draw comparisons** carry the full 0.93-point sd and a 2.77-point range. Any claim built that way needs to clear ~2 points to be safe.
  The absolute model-vs-constant numbers in this project are of this kind, and their margins (7.16 and 7.48 points at 6000 steps) clear it.
- **Paired comparisons on identical windows** cancel the draw entirely. The relevant noise is the per-window paired difference: measured **sd 0.72 points**
  over one window, so over 104 windows x 63 positions the standard error of a paired delta is **0.04 points**. That is why the window bank's paired deltas
  (+0.06 trainer, +0.32 independent) are meaningfully measured even though they are far below the unpaired noise - and why the paired design was the right
  response to the earlier withdrawal.

## The corrected statement of the resolution floor

| comparison type | noise that applies | smallest resolvable effect |
|---|---|---|
| unpaired, single draw, 104 windows, ~23% accuracy | sd 0.93, range 2.77 | ~2 points |
| unpaired, single draw, 104 windows, ~12% accuracy | sd 0.41, range 0.96 | ~1 point |
| paired, identical windows, 104 windows | SE 0.04 | ~0.1 points |

## What this does not change

Every paired result today stands on its paired test, not on this floor: the window bank at 6000 steps (+0.06 trainer, +0.32 independent, both below a 0.5 bar -
so still "buys nothing"), the age-key probe's +65.01 at delay 8 with per-seed deltas +62.1 to +69.5 (a value 70x the unpaired noise, so it never depended on
a small margin), and the three-seed arm comparisons that were explicitly paired on seed.
