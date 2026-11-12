# Re-timing the concept layer instead of weighting it: tested, rejected (2026-11-12)

The best accuracy arm on the original task is 12 de-duplicated concepts at alpha 1: **79.88% +/- 0.32**, but 7 of its 12
concept neurons do not track their names. The honest fix so far is raising the concept-loss weight (alpha), which reaches
0.7 of 12 misnamed but costs 9.88 accuracy points. That cost invites a different hypothesis:

**Hypothesis:** the concept layer is not under-*weighted*, it is under-*trained*. A single learning rate serves every
parameter, so the concept head may simply be moving too slowly to fit its labels while the class head fits the classes. Give
the concept parameters their own learning rate (`--lmul`, added to `glassnet/optim.py` and `gates/phase2_train.py`) and honesty
might arrive without paying for it in accuracy.

**Prediction, written before the run:** concept-group lr x3 at alpha 1 reaches mean rank honesty >= 0.65 (against 0.478) with
accuracy within 1 point of 79.88% and <= 4 of 12 misnamed. **It failed.**

```
| concept lr x | accuracy (3 seeds) | mean |rho| | misnamed | median per-rule | dead | interaction |
|---|---|---|---|---|---|---|
| 1 | 79.88 +/- 0.31 | 0.478 | 7.0 of 12 | +0.35 | 1.0 | 9.5% |
| 3 | 68.85 +/- 7.79 | 0.499 | 6.0 of 12 | +0.74 | 1.7 | 4.9% |
| 10 | 65.47 +/- 6.53 | 0.591 | 3.7 of 12 | +0.95 | 0.7 | 1.2% |
```

The patch is verified to have taken effect, not assumed: the trainer prints `[opt] concept-layer lr = 0.3000 (global lr
0.1000 x 3.00)`, and `concept.W` differs between the x1 and x3 checkpoints by up to 0.337.

## Result
- **Honesty barely moves**: 0.478 -> 0.499 -> 0.591, against 0.698 for alpha 3 on the same concept set.
- **Accuracy collapses anyway**: 79.88 -> 68.85 -> 65.47, with the x3 seed spread exploding to **+/- 7.85** (individual seeds
  72.88 / 73.81 / 59.87).
- So re-timing is strictly worse than re-weighting: it buys less honesty for more accuracy.

## Efficiency of the honesty levers, measured on the same concept set
```
lever                     accuracy cost   honesty gain (rho)   rho per accuracy point lost
alpha 1 -> 3                  -9.88            0.478 -> 0.698            0.0220
concept lr x10               -14.41            0.478 -> 0.591            0.0078
concept lr x3                -11.03            0.478 -> 0.499            0.0019
```
**The concept-loss weight is about 3x more efficient than the concept learning rate.** That ranking is the useful output of
this run: when a readable concept layer refuses to be honest, weighting its objective is the lever to pull, not re-timing it.

## One thing the lr lever does buy
Concept interaction falls **9.5% -> 4.9% -> 1.2%** and every rule becomes load-bearing at x10 (median per-rule +0.95, dead
rules 0.7). If a later arm needs maximally additive explanations rather than accuracy, that is the knob - recorded as a lead,
not a recommendation, since it costs 14.4 points of accuracy.

## Two bugs in my own audit tool, caught by these runs
1. **Metadata path assumed a filename.** `_ALT_META` replaced the literal string `concepts_selected.npz`, so pointing the
   audit at `concepts_sel12.npz` made it try to JSON-load an npz. All nine models reported "parse-fail" and the first read
   looked like nine broken checkpoints. Now the suffix is derived (`.npz` -> `.json`).
2. **The audit chose the flattering rule mode.** With two candidate forwards it picked whichever scored higher, which was
   linear at 73.10% over the sigmoid 72.88% the trainer actually used - a 0.22-point flattery and, worse, an audit of a model
   the trainer never built. It now picks the mode that **reproduces the recorded accuracy** and prints `MATCH`/`MISMATCH`.

## What stands
The 12-concept set with **alpha 3** remains the honest configuration (70.00% +/- 3.54, 0.7 of 12 misnamed) and **alpha 1** the
accuracy configuration (79.88% +/- 0.32). Re-timing the concept layer is closed as a lever.

## Reproduction
`--lmul {1,3,10}` with `GLASSNET_CONCEPTS=$PWD/data/concepts_sel12.npz --K 12 --R 16 --alpha 1.0`, audited by
`tools/phase2_reaudit.py` with the same environment variable. Nine checkpoints under `runs/phase2_sweep/LM*_s*/`.
