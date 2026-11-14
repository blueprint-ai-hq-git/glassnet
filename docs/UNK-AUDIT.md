# The headline is propped up by `<unk>`, and the model over-predicts it badly (follow-up to the attribution)

The failure attribution (`docs/history/FAILURE-ATTRIBUTION.md`) flagged `<unk>` as the most frequent target with a
56.41% score. This measures whether that distorts the headline number. Tool `tools/unk_audit.py`, checkpoint
`exposure_d1024_e9_s0_best`, 20,000 held-out windows, 14,145 assistant-masked positions.

## The numbers

```
overall accuracy                     29.34%
accuracy, <unk> targets excluded     27.79%     (5.40% of positions ARE <unk> targets)
model PREDICTS <unk> on              21.73%  (n=3,074)
  of those, correct                  14.02%
accuracy on <unk> targets            56.41%  (n=764)
```

## Predictions, scored
- **P1 excluding `<unk>` targets lowers accuracy by >= 0.5 pt - HOLDS** (-1.55).
- **P2 the model predicts `<unk>` on >= 10% of positions - HOLDS** (21.73%).
- **P3 fewer than 60% of `<unk>` predictions are right - HOLDS** (14.02%).

All three held again.

## What it establishes

Three facts, and they point the same way:

1. **The headline overstates real token prediction by 1.55 points.** 29.34% becomes 27.79% once positions whose target
   is `<unk>` are removed. Small, but it should be quoted from now on alongside the raw number.
2. **The model over-predicts `<unk>` by 4x.** It guesses `<unk>` on 21.73% of positions when only 5.40% of targets are
   `<unk>`. `<unk>` has become the place it puts uncertainty.
3. **And that dumping ground is wrong 86% of the time** (14.02% of `<unk>` predictions are correct). So the model is not
   even hedging usefully - it is spending a fifth of its predictions on a token that costs it accuracy.

This is the most actionable defect found today, and it is a **data/mix and vocabulary problem, not a width or scale
problem**: id 0 is `<unk>`, vocabulary is a fixed 12,002 entries, and 2.1% of corpus tokens are unknown. A model whose
best response to uncertainty is a token that is almost always wrong is being trained to give up.

Next lever, in order: (a) reduce `<unk>` in the training targets (drop or replace unknown tokens rather than train the
model to emit them), (b) re-measure the 27.79% baseline without them, (c) only then look at the generative path again.
