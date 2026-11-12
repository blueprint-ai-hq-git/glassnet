# Selecting concepts by dispersion does not remove dead concepts either (2026-11-12)

Third attempt at the same defect. `explain()` measures concepts that change 0.00% of predictions. Deleting them cost 8.62 points
(`docs/DEAD-CONCEPT-REMOVAL.md`); penalising the rule weight they attract did nothing (`docs/CONCEPT-COVERAGE.md`). That
measurement showed influence tracks activation **variance** (dead concepts mean std 0.101 versus 0.160 for live ones), so the
next lever was to exclude low-variance candidates **before** training instead of fixing them after.

**Prediction, written before the run:** a dispersion-filtered 12-concept set (min activation std 0.12) keeps accuracy within 1.5
points of 76.10% +/- 2.40 while raising the minimum concept influence above 2% and leaving at most one concept under 1%.
**Both halves failed.**

## The filter, and the first warning sign
`tools/select_concepts.py <K> <diversity limit> <min_std>` now scores candidates by probe learnability, rejects any whose
normalised activation std is below `min_std`, then greedily picks mutually distinguishable concepts.

```
diversity 0.75 + min_std 0.12  ->  6 concepts survive
diversity 0.85 + min_std 0.12  -> 10 concepts survive
```
The diversity constraint and the dispersion constraint fight each other: the well-spread, distinguishable features in this pool
are the low-variance ink summaries. And the filter excludes `symmetry_ud` (std **0.115**, just under the bar) which measured
**15.79%** causal influence - so the filter already discards a live concept while keeping `corner_ink` (std 0.138) which carries
1.49%.

## Result: 10 concepts, three seeds
```
arm                             accuracy (3 seeds)   min influence   mean influence   concepts <1%
12 concepts, unpruned a3 R32    72.01 +/- 4.35         0.00%            20.90%          3 of 12
10 dispersion-selected a3 R32   66.04 +/- 2.38         0.00%            31.90%          1-2 of 10
change                          -5.97 points           unchanged        +11.00          -
```
Per seed: 63.88 / 65.64 / 68.59 accuracy; minimum influence **0.00%** in all three; 1, 2 and 2 concepts under 1%.

**Honest confound:** this arm changed two things at once - the concept set AND the count (12 -> 10). The earlier K=9 experiment
gave 67.48% +/- 10.01, so most of the 5.97-point drop is probably the count rather than the dispersion filter. The arm cannot
separate them, and it should have been run at matched K.

## The finding that survives the confound
**Even when every selected concept varies (std >= 0.12 by construction), training still produces concepts with 0.00%
influence.** The deadness is not inherited from the feature set - it is created by training. Three independent attempts now agree:

```
lever                                      what it changed              what happened to dead concepts
delete dead concepts after training        concept set (12 -> 9)        fewer channels, accuracy -8.62, dead ones gone but model worse
penalise rule weight per concept (cov)     how much weight they attract  none - minimum influence still 0.00%
exclude low-variance candidates up front   the candidate pool            none - minimum influence still 0.00%, accuracy -5.97
```
Through all three, mean influence per concept rises whenever the layer gets narrower (20.9 -> 31.9 here) while the minimum stays
at zero: narrowing redistributes attention but the optimiser always leaves at least one channel unused.

## What that implies for the next lever
The answer path is free: 32 rules with full answer weights can simply ignore a concept, and nothing in the objective notices.
The lever that removes that freedom is **concept dropout** - randomly zero a concept's activation during training so the model
cannot lean on the others - which makes each concept *necessary* rather than merely *readable*. That is the next experiment,
and unlike the previous three it changes what the model can rely on rather than what it is scored on or which features exist.

## Reproduction
`tools/select_concepts.py 12 0.85 0.12` (writes `data/concepts_disp10.npz` / `.json`, preserved), trainer with
`GLASSNET_CONCEPTS=$PWD/data/concepts_disp10.npz --K 10 --R 32 --alpha 3.0 --tag D10_s*`, influence from
`tools/explain_image.py <ckpt> 0` with `GLASSNET_CONCEPTS_META=$PWD/data/concepts_disp10.json`. Three checkpoints kept under
`runs/phase2_sweep/D10_s*/`; every earlier cache untouched.
