# Where the 29% actually comes from: the model is mostly a copier (LOOK INSIDE FIRST)

Tool `tools/failure_attribution.py`, checkpoint `exposure_d1024_e9_s0_best` (d1024, 9 epochs), 20,000 held-out windows,
14,145 assistant-masked positions. For every position it records whether the target token was present in the input
window, whether the model's prediction was present in the input window, and whether it was right.

## The numbers

```
overall accuracy                 29.34%

target IS in the input window    43.90%   (n=6,396, 45.2% of positions)
target is NOT in the window      17.32%   (n=7,749, 54.8% of positions)
ratio                             2.54x

predictions copied from window   60.23% of ALL predictions
                                 67.66% of CORRECT predictions

top-200 target tokens            70.0% of positions, accuracy 39.89% (vs 29.34% overall)
```

Most frequent targets and their accuracy: `<unk>` 56.41% (n=764), `,` 38.43%, `.` 48.51%, `the` 39.77%, `of` 72.28%,
`)` 77.19%, `$` 62.76%, `to` 44.87%, `=` 28.63%, `-` 30.40%, `(` 45.29%, `:` 67.31%.

## Predictions, scored
- **P1 in-window accuracy >= 2x out-of-window - HOLDS** at 2.54x.
- **P2 >= 60% of correct predictions are copies from the window - HOLDS** at 67.66%.
- **P3 top-200 targets cover >= 50% of positions and beat overall accuracy - HOLDS** (70.0%, 39.89% vs 29.34%).

All three held, which is unusual for this project and means the attribution is not ambiguous.

## What this establishes, and what it changes

**Two thirds of the model's successes are retrieval, not generation.** When the answer is already somewhere in the
window it reaches 43.90%; when it is not, it drops to **17.32%**. So the thing the model is actually good at is finding a
token and repeating it, and that is also the thing this project already has a separate, stronger mechanism for (the
pointer/copy path, measured at 71.0% held-out copy from its own base).

Two consequences for the standings goals:
1. **Width is not the lever** (settled separately: +0.53 paired at matched training). Neither is training length (both
   sizes plateau by epoch 6). The lever is the **generative path** - the 54.8% of positions where the answer must be
   produced rather than found - and that is where 17.32% lives.
2. **`<unk>` is the single most frequent target (n=764) and the model scores 56.41% on it.** A model rewarded for
   emitting `<unk>` is being paid for giving up. This is a data/mix question, and it is measurable: how much of the 29%
   is `<unk>` predictions that are "correct" only because the corpus itself contains `<unk>`?

Next measurement this implies (not yet run): accuracy recomputed with `<unk>` targets excluded, which separates "real
token prediction" from "correctly guessing that a token is unknown".
