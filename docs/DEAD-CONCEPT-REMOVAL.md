# Deleting the causally dead concepts makes the model worse, not cleaner (2026-11-12)

`explain()` measured three of twelve concepts as carrying no decision at all: `top_heavy` changes 0.01% of predictions,
`ring_ink` 0.00%, `edge_balance` 0.00%, while passing the label-honesty gate. The obvious cleanup is to drop them and retrain on
nine concepts, so that every concept in the vocabulary earns its place. That was tested.

**Prediction, written before the run:** accuracy within 1.0 point of the twelve-concept pruned arm (76.10% +/- 2.40), mean
causal influence per concept up from 20.90%, honesty unchanged near 0.70. **Half failed, badly.**

```
arm                       accuracy (3 seeds)      mean causal influence per concept
12 concepts, pruned       76.10 +/- 2.40              20.90%   (3 concepts below 1.5%)
 9 concepts (3 dropped)   67.48 +/- 10.01             31.24%   (lowest single concept 0.92-13.83%)
change                    -8.62 points                +10.34 points
```

Individual seeds of the nine-concept arm: **55.59 / 74.30 / 72.55**. Seed 0 collapsed, and its own connection pruning cost
10.74 points (66.33 pre-prune -> 55.59 post-prune), so the arm is not merely worse, it is unstable.

## What this shows
- **The influence half of the prediction held**: with three decorative concepts gone, mean per-concept influence rises from
  **20.90% to 31.24%**. Every remaining concept now changes 1% to 80% of predictions.
- **The accuracy half failed by 8.62 points** against a 1.0-point tolerance, and the seed spread grew from +/- 2.40 to +/- 10.01.
- **So the decorative concepts were not decorative in combination.** The concept layer is the only path from pixels to answers;
  a concept that never wins an argument on its own is still a channel the rules can combine with others. Removing it removes a
  channel, and nine channels are markedly worse than twelve - not "cleaner at the same accuracy".
- **The lesson for readability is the uncomfortable one**: influence per concept and accuracy trade in the direction opposite to
  what a readability-first design wants. You can have a layer where a third of the concepts never matter (76.10%) or a layer
  where every one matters (67.48%), and the first is 8.62 points better.

## What the right fix looks like instead
Deletion is the wrong lever; the measurement suggests **coverage** - require every concept to be used by at least one rule that
is itself load-bearing, rather than deleting concepts that are not. That keeps all twelve channels (and the accuracy) while
making the claim "every named concept matters" true by construction. That is the next structural experiment, and it is a change
to the training constraint, not to the concept set.

## Reproduction
`data/concepts_sel9.npz` / `.json` (the twelve-concept set minus the three measured-dead ones, provenance in the JSON note),
trainer with `GLASSNET_CONCEPTS=$PWD/data/concepts_sel9.npz --K 9 --R 32 --alpha 3.0 --tag K9_s*`, causal influence from
`tools/explain_image.py <ckpt> 0` with `GLASSNET_CONCEPTS_META=$PWD/data/concepts_sel9.json`. Three checkpoints kept under
`runs/phase2_sweep/K9_s*/`; the twelve-concept caches are untouched.
