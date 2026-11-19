

---

# THE WINDOW BANK AT THREE SEEDS: +0.87 POINTS, ALL THREE SEEDS POSITIVE (2026-11-19 16:24)

The paired comparison that replaced this morning's withdrawn single-seed +0.50. Three seeds, both arms per seed, same data order, only the window differs.
Scored on the trainer's own 104-window line (`rng(999)` is fixed, so the window draws are IDENTICAL across arms - this is a genuinely paired design).

```
seed   control (K=0)   window (K=8)   paired delta
0      14.75%          15.25%         +0.50
1      16.03%          17.11%         +1.08
2      10.50%          11.52%         +1.02

mean paired delta +0.87   sd 0.32   se 0.19   t = 4.73 (crit 4.303 at 95%, df=2)   CI [0.07, 1.66]
```

- **P2 MET: all three seeds share the same sign.**
- **The paired test is significant at 95%** (t 4.73 against a critical value of 4.303) - three of three positive with a mean of +0.87 and a 95% interval
  that excludes zero, just barely.
- **This is not the same claim that was withdrawn this morning.** The withdrawn one was a single seed compared as arm means, with an independent re-score
  that flipped its sign. This one is three paired seeds, and its interval only just clears zero.

## The two caveats that keep this small

1. **It is measured on ONE instrument.** The independent evaluator gave 12.19% / 12.03% on the seed-0 pair - a **negative** delta - so the sign of this effect
   has not yet been confirmed on a second instrument for seeds 1 and 2. That is the immediate next measurement, because a delta that exists on only one
   instrument is not a delta.
2. **Both arms are at or below the trivial constant** on the corrected protocol at 3000 steps (constant 13.14% `<unk>`-excluded). A +0.87-point gain on a
   model that is still at the level of "guess the most frequent token in the window" is real but not useful yet.

## What this means in plain terms

At a budget where the model cannot beat a constant, adding the readable window changes its score by **+0.87 points consistently across three seeds** - small
but no longer explicable as one seed's luck. Whether that survives an independent instrument, and whether it grows at the qualifying 6000-step budget, are
both open. The age-key result (+65.01 at delay 8, three seeds) says the bank's ADDRESSING was the real defect; this run says that even the un-fixed bank is
mildly positive on the language objective rather than neutral.
