# RL from a verifier on the carry problem: nothing learns it, and partial credit destroys what was known (2026-11-13)

Three levers failed to close an 8.1-point gap against supervised learning on a fully-labelled task (`docs/RL-V1-METHOD.md`,
`docs/RL-DENSITY.md`, `docs/RL-GROUP-SIZE.md`). The remaining honest test is the setting RL exists for: an objective that is
**checkable but not demonstrable**. The model's supervised warm start only ever sees sums <= 9, so it cannot imitate an answer to
8+7 - no such demonstration exists in its data - and every arm ever measured here scores **acc_gen 0.00%**. RL is handed sums
>= 10 plus a checker, and no demonstrations.

## Arms (3 seeds each, 400 supervised warm start + 400 RL steps, one nice-19 core)
```
arm          reward on sums >= 10                  purpose
gen_sparse   binary: 1 if the whole answer is right  can the verifier teach carry on its own?
gen_dense    per digit: +1 / -1 / -0.5 per digit    partial credit, so a right first digit is already signal
sup_gen      800 supervised steps, in-distribution  reference under identical compute
```

Predictions, written before the run: `P1` dense beats sparse on `acc_gen` by >= 3 points, sparse stays within 1 point of 0.00;
`P2` dense does not reach the labelled reference but closes >= 20% of that gap; `P3` `acc_in` falls by no more than 3 points.

## Result
```
arm          acc_gen per seed      acc_gen mean    acc_in per seed              acc_in mean
gen_sparse   0.00, 0.00, 0.00          0.00       18.22, 22.93, 21.10            20.75
gen_dense    0.00, 0.00, 0.00          0.00        2.34,  2.93,  4.13             3.13
sup_gen      0.00, 0.00, 0.00          0.00       28.04, 25.85, 30.28            28.06
```

- **`acc_gen` is exactly 0.00 in all nine runs.** Binary verification cannot start (see below), and partial credit, which does
  produce gradient, still cannot produce a single correct two-digit answer.
- **P3 FAILED catastrophically: per-digit credit collapsed in-distribution accuracy from ~22% to 3.13%.** The RL phase did not
  trade in-distribution ability for carry - it simply destroyed it. That is worse than not helping, and it is the single most
  important number in this table.
- **My own framing of the reference arm was wrong and is corrected here.** `sup_gen` was described in the script as "the ceiling
  when labels DO exist", but the tool trains the supervised arm on the in-distribution split by construction, so it never sees a
  sum >= 10 either. Its 0.00 tells us the un-generalising model cannot do carry with 800 labelled in-distribution steps; it does
  **not** price the label advantage on the carry problem, and P2 was therefore never testable as written.

## Why binary verification cannot start - the mechanism, measured
The smoke run (seed 0, 150 steps) recorded **mean reward -0.006** with `acc_gen` 0.00. A two-digit answer sampled from a policy
that has never produced one is almost never exactly right, so every sample in a group receives the same reward, the
group-relative advantage is **exactly zero**, and `logp_selected` receives zero weight on every token. The method is not failing to
learn slowly; it is receiving no learning signal at all. This is the standard sparse-reward cold start, reproduced here rather
than asserted.

## What this settles and what it leaves
Settled: **on this task and this model, RL from verification does not produce carry**, and the per-digit variant damages what the
warm start already knew. Reporting partial credit as the fix for RL - which `docs/RL-DENSITY.md` left open - is now closed for
this task.

Left open, in order of how cheaply they can be separated:
1. **Is the collapse a learning-rate artefact or a reward artefact?** A 19-point loss from a 400-step RL phase at lr 0.02 looks
   like overshoot rather than a property of per-digit credit. Same arm at a lower lr separates those, and it is running next.
2. **Does mixing in-distribution pairs into the RL batch prevent the collapse?** If yes, the reward is usable and the damage came
   from training on one distribution only.
3. **Can the carry task be reached at all at 7,342 parameters?** Every method - supervised on in-distribution data, binary
   verification, partial credit - sits at 0.00. Before blaming the RL method, a curriculum (single-digit results first, then
   two-digit) or a larger model has to show the task is reachable. Until something reaches **any** accuracy above zero on sums
   >= 10, "RL cannot learn carry" is not a claim about RL.

## Reproduction
`lens/scripts/run_rl_gen.sh` (predictions in the header), logs `logs/rl_gen/{sparse,dense,sup}_s{0,1,2}.log`, per-run JSON in
`runs/rl/`. Code: `tools/rl_compare.py --task gen_verify`, `glassnet/rl.py: token_rewards`.
