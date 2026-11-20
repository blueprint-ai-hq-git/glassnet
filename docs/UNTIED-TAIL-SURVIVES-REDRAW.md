

---

# THE UNTIED-READOUT FINDING SURVIVES A DIFFERENT DRAW OF WINDOWS - AND A SILENTLY IGNORED SEED WAS CAUGHT BY IDENTICAL NUMBERS (2026-11-20 10:17)

Two things happened here, and the smaller one is the more embarrassing.

## 1. The tool could not be re-drawn, and said so silently

`tools/midband_split.py` had `rr = np.random.default_rng(4242)` hardcoded. I set `DRAW_SEED=777` in the environment, the tool **ignored it**, and both
checkpoints returned numbers identical to the last decimal place. That identity is what exposed it - a genuine re-draw cannot produce the same digits.

My first patch attempt also silently failed: I wrote a regex looking for `rng = np.random.default_rng(` while the variable is `rr`. The patch reported
nothing, applied nothing, and the "re-draw" measured the same windows again.

Fixed properly: `DRAW_SEED` is read from the environment **and printed with every run**, so a reader can always see which draw produced a number:

```
DRAW_SEED=777  (windows are drawn with this seed; same seed = same windows, paired)
```

This is the fifth silently-ignored flag or variable this project has produced, and the rule that would have caught all of them is the one already written:
**a flag that does nothing must make the tool refuse to run, and every tool must print the settings that shaped its output.**

## 2. The untied-readout finding survives a different draw

Same two checkpoints, 104 windows, different windows (n per sub-band changed: 786/451/690/410 becomes 777/541/656/422):

```
draw 4242                         tied       UNTIED
  ranks 20-50                    25.19%      35.24%
  ranks 50-100                    0.22%       2.66%      (1 of 451 -> 12 of 451)
  ranks 100-250                   0.00%       0.29%
  r21-500 overall                 8.52%      12.45%
  top-20 accuracy                35.81%      41.61%
  top-20 default rate            94.90%      70.73%

draw 777                          tied       UNTIED
  ranks 20-50                    21.62%      31.02%
  ranks 50-100                    0.18%       2.77%      (1 of 541 -> 15 of 541)
  ranks 100-250                   0.00%       0.76%
  r21-500 overall                 7.05%      10.89%
  top-20 accuracy                34.78%      43.07%
  top-20 default rate            85.64%      71.83%
```

Both draws agree in sign and in rough magnitude on every band, including the one that matters most: the tail rate goes from ~1 position in ~500 to
12-15 positions in ~500 under an untied readout. The absolute numbers move by a point or two between draws, which is the draw noise already measured
(sd 0.93 overall); the **sign never flips**.

So the claim stands with its confound still attached: capacity in the readout is **sufficient** to move the tail, on two independent draws, and the
clean attribution test (`untied6k`, tying off and nothing else changed) is running.
