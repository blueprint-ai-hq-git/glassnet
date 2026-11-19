

---

# THE INSTRUMENT IS SHARPER THAN I SAID: THE 104-WINDOW DRAW NOISE IS 0.41 POINTS, NOT 1.5 (2026-11-19 13:35)

Twice today I wrote that a single 104-window number has a standard error "around 1.5 points", and I used that to dismiss small effects. **I never measured
it. I asserted it.** So I measured it: one fixed checkpoint, five different window draws, no training, nothing changes but which 104 windows are drawn.

```
checkpoint: win3kwinoff_K0_ep3000.npz        windows per draw: 104
draw_seed 4242 -> 12.19%
draw_seed 11   -> 12.19%
draw_seed 22   -> 11.57%
draw_seed 33   -> 11.23%
draw_seed 44   -> 11.81%

mean 11.80%   sd 0.41 points   full range 0.96 points
```

**The draw noise is 0.41 points, about 3.6x smaller than the 1.5 I had been quoting.** I was wrong, and in the pessimistic direction - which means I
have been discarding effects as "inside the noise" that were actually larger than the instrument's real resolution.

## What this changes, and what it does not

- **The withdrawal of the +0.50 window-bank delta STANDS.** It is not draw noise that killed it: the independent re-score put it at -0.16 (sign
  flipped) and the arm's own seed-to-seed spread is **4.25 points** (seed 0 control 14.75%, seed 2 control 10.50%). Seed variance is the dominant
  noise source, not the window draw - it is **10x larger**.
- **The 2.6-point gap between the trainer's line and the independent evaluator is NOT draw noise.** At 0.41 points of draw noise, a 2.6-point gap is a
  **protocol difference** between the two tools - different band definitions or a different window space. That is a real discrepancy to investigate,
  not sampling.
- **The resolution floor for future single-draw comparisons is now measured, not guessed: 0.41 points.** Anything smaller cannot be claimed at this
  protocol. Anything larger deserves a re-score on a second draw before it is believed.
- **The right noise hierarchy for this project, now measured:** seed variance **4.25** >> protocol difference **2.6** >> draw noise **0.41**. Every
  arm today was compared on one seed, so every arm has been dominated by the largest of these and the instrument was never the limiting factor.

## The lesson

An unmeasured error bar is a made-up error bar. I used "~1.5 points" as a rhetorical device to justify dismissing results, and the real number is 3.6x
smaller - which cuts both ways: it makes small effects more believable than I claimed, and it makes the seed variance (which I had not measured either
until twenty minutes ago) the thing I actually needed to control.
