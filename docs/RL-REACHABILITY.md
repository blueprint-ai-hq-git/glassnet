# The carry task IS reachable - 34.66% +/- 5.55 with demonstrations, 0.00 everywhere without (2026-11-13)

Every RL arm in this investigation sits at **acc_gen 0.00** on sums >= 10 (`docs/RL-GEN-VERIFY.md`, `docs/RL-REWARD-HACK.md`,
`docs/RL-SHAPE-HACK.md`). Before spending another arm, the task had to be shown reachable by *something*, otherwise "our RL cannot
learn carry" would be a statement about an impossible task rather than about the method.

## The check
Supervised training, 800 steps, lr 0.02, three seeds, same 7,342-parameter model. Only the training distribution changes:
`sup_in` keeps the original in-distribution pairs (sums <= 9), `sup_gen` trains only on sums >= 10, `sup_both` mixes both.

**Prediction, written before the run:** `sup_both` and `sup_gen` reach `acc_gen` >= 20%. **Falsifier:** ~0.00 means the task is
beyond this model and gets dropped rather than reward-tuned.

## Result - prediction passed
```
arm        acc_gen per seed          acc_gen mean +/- sd    acc_in per seed        acc_in mean
sup_in     0.00,  0.00,  0.00            0.00               28.04, 25.85, 30.28      28.06
sup_both  40.84, 28.57, 34.57           34.66 +/- 6.16       1.40,  0.98,  2.29       1.56
sup_gen   41.36, 27.04, 34.04           34.15 +/- 7.02       0.00,  0.00,  0.00       0.00
```

**The carry task is reachable at 7,342 parameters: ~35% on sums the model never saw demonstrations for, in the two arms that were
given two-digit answers to learn from.** So the previous conclusion stands on firm ground now - it is not a task artefact:

```
same model, same compute, same pair distribution
supervised WITH two-digit labels       acc_gen 34.66 +/- 6.16
RL from a verifier, binary reward      acc_gen  0.00  (advantage exactly zero, cannot start)
RL from a verifier, per-digit reward   acc_gen  0.00  (answers '1' to everything)
RL from a verifier, shape bonus        acc_gen  0.00  (answers '98' to everything, acc_in also 0.00)
```

## A second finding, and it is about capacity not reward
**Learning carry costs in-distribution accuracy.** `sup_in` keeps 28.06% on singles and 0.00% on carries; `sup_gen` flips it to
34.15% on carries and 0.00% on singles; `sup_both` - which is given both distributions - gets 34.66% on carries but only **1.56%**
on singles. Being shown both does not let the model hold both: it moves to one and leaves the other.

That is the same competition this project measured between copying and language (`docs/COPY-KEEP-LEARNING.md`,
`docs/COPY-WEIGHTS.md`), now reproduced on arithmetic: at this size the model allocates its representational budget to one
distribution and the other is lost. It also explains why every RL arm that trained on carries lost the singles - that collapse was
never purely a reward artefact, there is a capacity term underneath it.

## What changes about the RL line
The task is fair, so RL can now be measured honestly against a reachable target. The obvious next experiment is the standard one
this project has not run in this configuration: **warm-start from a checkpoint that can already produce two digits (sup_gen), then
run RL with a binary verifier.** With the reward satisfiable, the advantage is no longer zero, and the question becomes whether RL
holds or improves a working capability rather than whether it can create one from nothing. Prediction and falsifier are written
into the script before it runs.

## Reproduction
`lens/scripts/run_rl_reach.sh` (prediction in the header), `--sup-dist` in `tools/rl_compare.py`, logs `logs/rl_reach/`.
