# The pruned ladder: every arm of the concept study, before and after rule pruning (2026-11-12)

Pruning deletes a rule's incoming weights AND its answer row, judged on a held-out slice of the train split (a rule goes if it
costs less than 0.05 points there), greedily with re-measurement after each removal. Applied to every arm of the concept-set
study, three seeds each, with the same tool and tolerance.

## Class-prior census first, because it predicts where pruning pays
A rule whose output is constant across every test image but whose answer row still pushes a class is a class prior wearing a
rule's name (see `docs/CLASS-PRIOR-RULES.md`). Counting them per arm, on seed 0:

```
arm      rules   constant-output rules   share   median input strength
a3 R8       8           1                 12%         0.383
a3 R16     16           2                 12%         0.282
a3 R32     32          11                 34%         0.066
a3 R48     48          15                 31%         0.052
a1 R16     16           2                 12%         0.276
```
The degeneracy appears once the layer is wide enough to afford it: **12% of rules at 8-16 rules, ~31-34% at 32-48.** That is
why the pruning gain is largest exactly where the layer is widest.

## Before and after pruning, three seeds each
```
| arm | rules | unpruned accuracy | pruned accuracy | change | removed (mean) | left to read | class-prior share |
|---|---|---|---|---|---|---|---|
| a1 R16 | 16 | 79.88 +/- 0.31 | 79.78 +/- 0.48 | -0.10 | 0.3 | 16 | 12% |
| a3 R16 | 16 | 70.00 +/- 3.41 | 70.39 +/- 3.40 | +0.39 | 1.0 | 15 | 12% |
| a3 R32 | 32 | 72.01 +/- 4.36 | 76.10 +/- 2.40 | +4.09 | 3.3 | 29 | 34% |
| a3 R48 | 48 | 74.73 +/- 1.73 | 76.20 +/- 1.71 | +1.47 | 5.0 | 43 | 31% |
| a3 R8 | 8 | 69.48 +/- 5.35 | 69.48 +/- 5.35 | +0.00 | 0.0 | 8 | 12% |
```

## What this changes
- **The best honest arm is now `a3 / 32 rules / pruned`: 76.10% +/- 2.40 with 28.7 rules to read** and 1.3 of 12 concepts
  misnamed. Pruning is worth **+4.09 points** there - the largest single gain measured on this layer - because a third of that
  rule layer was class priors.
- **It matches the 48-rule arm's accuracy (76.20% +/- 1.71) with 14 fewer rules to read.** For a readable model that is the
  better deal: same accuracy, 29 sentences instead of 43.
- **Pruning is not universally positive.** On the 16-rule alpha-1 arm it cost 0.30 points on seed 0 and removed nothing on two
  other seeds; on the 8-rule arm nothing was removable at all. The gain tracks the class-prior share, as the census predicts.
- **The accuracy champion is unchanged**: `a1 / 16 rules / pruned` = 79.78% +/- 0.48, but it still carries 7 of 12 concept
  names that do not describe their neurons. Pruning cannot fix concept honesty - a rule that reads nothing is not a concept
  problem.

## The honest scoreboard after pruning
```
configuration                            accuracy          misnamed    rules to read
a1 / 16 rules / pruned                79.78 +/- 0.48       7.0 of 12        15.7
a3 / 32 rules / PRUNED                76.10 +/- 2.40       1.3 of 12        28.7
a3 / 48 rules / pruned                76.20 +/- 1.71       1.0 of 12        43.0
a3 / 16 rules / pruned                70.39 +/- 3.40       0.7 of 12        15.0
```
**The gap between the accuracy champion and the best readable honest arm is 3.68 points**, and the readable one has 5.4x fewer
misnamed concepts for it.

## Reproduction
`tools/prune_all.py` (drives `tools/prune_rules.py` over every arm, writes `runs/phase2_sweep/prune_all.csv`). Pruning JSON
written beside each checkpoint; no checkpoint modified or deleted.
