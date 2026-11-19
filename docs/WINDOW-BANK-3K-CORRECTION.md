

---

# CORRECTION: THE WINDOW BANK'S +0.50 POINTS DOES NOT SURVIVE AN INDEPENDENT RE-SCORE (2026-11-19 13:20)

An hour ago I published the window bank beating its control by **+0.50 points**, exactly on the pre-registered bar. I labelled it "not yet
proven" because one seed cannot separate +0.50 from 0. **That caution was correct and the number itself does not stand up.** The two arms
launched before the automatic SELF-CHECK existed, so their numbers came from ONE instrument - the trainer's own held-out line. Re-scoring the
same two checkpoints with the independent evaluator (`tools/eval_any.py`) gives:

```
checkpoint                    trainer's own eval   independent re-score
win3kwinoff_K0_ep3000.npz     14.75%               12.19%   (top20 23.99%, r21-500 0.04%)
win3kwinon_K8_ep3000.npz      15.25%               12.03%   (top20 23.66%, r21-500 0.04%)

paired delta                  +0.50                -0.16        <- THE SIGN FLIPS
```

Three separate reasons the +0.50 cannot be quoted:

1. **It does not reproduce.** An independent instrument on the same two files reads **-0.16**, not +0.50. Both are 104-window evaluations with
   different window draws, so the difference between draws is itself larger than the effect.
2. **It is smaller than seed noise.** Seed 2's control just finished at **10.50%** against seed 0's **14.75%** - **4.25 points of seed-to-seed
   variance** in the same arm, same data, same settings. The claimed effect (0.50) is **8.5x smaller** than the arm's own seed spread.
3. **The trainer's number is systematically higher than the independent one** on these files: 14.75 vs 12.19 and 15.25 vs 12.03, both ~2.6 points.
   That is a protocol difference, not a window-bank difference, and it applies equally to both arms - which is why the DELTA is what matters and
   why a delta computed on one instrument must be confirmed on a second.

## What is still true, and what is withdrawn

- **WITHDRAWN:** "the readable window bank gives +0.50 points at 3000 steps". It is instrument-dependent and far below seed noise.
- **STILL TRUE:** the window is **readable** (8 distinct real tokens at the end of training), its gradient is exact (4.334e-10), its batched path
  equals the certified looped path (1e-16), it does not collapse (slot-to-slot cosine -0.081 against the old bank's 0.864), and it is not
  measurably NEGATIVE at 3000 steps the way the slot bank was.
- **STILL OPEN:** whether any memory mechanism helps this model. Three seeds are already running; the statistic that matters is the mean PAIRED
  delta with an interval, and given 4.25 points of seed variance this arm was always going to need more than one seed per cell.

## The process lesson, which is worth more than the result

The self-check I added two hours ago exists precisely to force this comparison at the end of every run. These two arms predate it, and they are
the first case where its absence hid a withdrawn number. **A delta measured on one instrument is a hypothesis about the architecture until a
second instrument agrees on the same file.**
