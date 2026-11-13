# RL: more samples per prompt did not help either - the comparison is what is wrong (2026-11-13)

Third and last cheap lever on the RL line, after the implementation (`docs/RL-V1-METHOD.md`) and reward density
(`docs/RL-DENSITY.md`) were both tested and rejected. The group-relative advantage used only **8 samples per prompt**, so its mean
and standard deviation are noisy, and that noise goes straight into every token weight. 32 samples per prompt gives a ~4x less
noisy baseline with the reward untouched.

## Design
`lens/scripts/run_rl_group.sh`, three seeds, identical to the dense arm in `docs/RL-DENSITY.md` except `--group 32`. Equal compute
(400 supervised warm start + 400 RL steps). Pinned to one core with `taskset -c 2`, `nice 19`, `ionice c3` because a game was
running, and started through `lens/scripts/launch_detached.sh` so it could not die with the tool session (that is what killed the
first attempt, `logs/rl_group/g32_s0.log`).

**Prediction, written before the run:** group 32 beats group 8 (19.95 +/- 1.11) by >= 2 points on `acc_in`, same seeds.
**Falsifier:** within 1 point of 19.95 means advantage variance is not the constraint either.

## Result
```
seed-matched acc_in        group 8    group 32     difference
seed 0                      21.03      21.50        +0.47
seed 1                      20.00      18.54        -1.46
seed 2                      18.81      18.81         0.00
mean +/- sd            19.95 +/- 1.11   19.62 +/- 1.64   -0.33
supervised reference (equal compute)   28.06 +/- 2.21
```

**Prediction FAILED and the falsifier TRIGGERED.** Group 32 is **0.33 points worse**, not 2 points better, and every seed-matched
difference is inside 1 point (or negative). Quadrupling the samples per prompt bought nothing.

## What the RL line now knows, after three measured levers
```
lever                                    result                                    verdict
implementation corrected                 still loses (docs/RL-V1-METHOD.md)        not the cause
reward density (per-digit, per-token)    +1.11 points, ~1.3 SE (RL-DENSITY.md)      not the cause
samples per prompt 8 -> 32               -0.33 points (this doc)                    not the cause
supervised at equal compute              28.06 +/- 2.21 vs RL best 19.95 +/- 1.11   an 8.1-point gap
```

The method is not being starved of signal in any of the ways the project guessed. **The comparison itself is the problem**: every
arm so far asks RL to beat supervised learning on a task where supervised has every token labelled. That is the one setting in
which RL is least likely to win, and it has never been tested where RL is actually for - an objective that is **checkable but not
demonstrable**.

That is the next experiment and it is already written with its prediction: consistency between `a+b` and `b+a`. There are no labels
for "give the same answer to the same sum written the other way round" - only a check - so supervised has nothing to imitate, and
the measured accuracy (which the reward never sees) tells us whether the model learned a real capability or merely learned to be
**consistently wrong**. The second outcome is a genuine risk and is written into the script as the falsifier.

## Reproduction
`lens/scripts/run_rl_group.sh` (prediction in the header), logs `logs/rl_group/g32_s{0,1,2}.log`, per-run JSON in `runs/rl/`.
Timing: ~300 s per seed on one core. The group-8 and supervised numbers it is compared against come from
`lens/scripts/run_rl_dense.sh` (same seeds, same compute, same script lineage).
