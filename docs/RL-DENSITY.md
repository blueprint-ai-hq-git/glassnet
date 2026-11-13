# Reward density is not what was holding our RL back either (2026-11-13)

Our RL method lost to supervised learning twice with a correct implementation. `docs/RL-V1-METHOD.md` states the reason and names
the first fix to try: a supervised step labels **every** answer token, while the RL reward was **one bit per sampled sequence**, so
400 RL steps buy less than 400 supervised steps. That doc's own recommendation is reward density. This tests it.

## What was built
The addition task's answer is checkable digit by digit, so the reward does not have to be one bit per sequence.
- `glassnet/rl.py: token_rewards()` - **+1** for a digit in the right place, **-1** for a wrong one, **-0.5** for a digit past the
  true answer's length, and **+1** on the final token when the whole answer is exactly right.
- `tools/rl_compare.py --dense-reward` - builds a **per-token** advantage against the same digit position of the other samples of
  the same prompt, instead of one sequence-level advantage per sample. A sequence-level advantage cannot distinguish "first digit
  right, second wrong" from "both wrong"; this can.

Equal compute in every arm: 800 supervised-equivalent steps (supervised gets all 800; the RL arms get 400 warm start + 400 RL).
Three seeds each.

## Predictions, written before the run
`P1` dense beats sparse on `acc_in` by >= 2 points. `P2` dense closes at least half the sparse-to-supervised gap. `P3` `acc_gen`
stays near zero; if dense moves it, that is the more interesting result. Falsifier: dense within 1 point of sparse means reward
density was not the binding constraint.

## Result
```
arm      seeds  acc_in mean +/- sd      acc_gen   ece_in
sup      3      28.06 +/- 2.21          0.00      0.087
sparse   3      18.84 +/- 1.00          0.00      0.278
dense    3      19.95 +/- 1.11          0.00      0.267
        dense per-seed acc_in: 21.03, 20.00, 18.81
        sparse per-seed acc_in: 17.76, 19.02, 19.72
```

- **P1 FAILED**: dense improves on sparse by **+1.11 points**, against a bar of 2.0.
- **P2 FAILED badly**: dense closes **12%** of the sparse-to-supervised gap (**18.84 -> 19.95** against supervised **28.06**),
  where half was required.
- **P3 passed trivially**: `acc_gen` is 0.00 in every arm, so dense reward did not accidentally change the generalising split.

## What the number actually says
Dense credit is a real but small effect: **+1.11 points** with per-arm spreads of 1.11 and 1.00, a combined standard error of
about **0.86** - roughly 1.3 standard errors on three seeds. So it is **not distinguishable from noise**, and it is nowhere near
enough to close an **8.1-point** deficit to supervised learning at equal compute.

The falsifier written before the run said "within 1 point of sparse -> density was not the binding constraint". At +1.11 the letter
of that bar is narrowly not met, but the intent is: **density is not the binding constraint either.** Both of the project's
diagnoses for why RL loses - implementation, then reward density - have now been tested and rejected by measurement.

## What this rules out, and what is left
Ruled out as the explanation: a wrong policy-gradient implementation (`docs/RL-V1-METHOD.md` records the corrected version), and
too-coarse reward (this doc). What has NOT been tried, in the order this result makes them attractive:
1. **Sample efficiency of the advantage estimate** - group 8 is small; 32 samples per prompt cuts advantage variance and buys
   reward information per step without touching the task. This is the direct successor and is queued with its prediction.
2. **A task with no demonstrations.** Every comparison so far is RL against supervised on a task where supervised has full labels,
   which is the case RL is least likely to win. RL is for objectives that are checkable but not demonstrable; that comparison has
   never been run and would be the honest test of the method rather than of the task.
3. **On-policy advantage for a specific failure** (looping, over-confidence) rather than for raw accuracy.

## Reproduction
`lens/scripts/run_rl_dense.sh` (predictions in the header, 9 runs, CPU so the GPU stayed free), logs in `logs/rl_dense/`,
per-run JSON in `runs/rl/`. Code: `glassnet/rl.py: token_rewards`, `tools/rl_compare.py --dense-reward`.
