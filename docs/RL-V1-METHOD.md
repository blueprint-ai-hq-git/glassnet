# Our own RL trainer, built properly this time: still does not beat supervised (2026-11-13)

Yahya's instruction: the custom RL training method comes first, done properly, and the memory/judge work follows. This is the
first honest measurement of a from-scratch RL method with a **verifiable** reward.

## What was built (all our own code, `glassnet/rl.py`)
- **Task with an exactly checkable reward:** `a+b=` -> the answer is right or wrong, no reward model to hack. Trained on pairs
  whose sum is a single digit; tested twice, in-distribution and on **unseen two-digit sums** (the generalisation split).
- **Group-relative advantage** (GRPO shape): each prompt samples a group of 8 answers from the model's own distribution, and every
  sampled token is weighted by how much better its sequence did than its own group mean. This is what makes repetition and echo
  costly instead of invisible.
- **Repetition penalty inside the reward**, so looping on a digit is not a free way to answer.
- **Entropy floor** (a hinge, not a bonus): pushes entropy up only when the policy has collapsed.
- Two new ops with hand-written backward passes, both **numerically gradient checked before use**:
  `logp_selected` (weighted log-probability) and `entropy_floor`.

## Three of my own bugs caught by measurement, not by reading
1. **Sign error in `logp_selected`.** `loss = mean(-w log p)` needs `d/dz = +w(p - onehot)`; I wrote the negative. The numerical
   check caught it at 3.2e-01 against a 1e-8 bar, before any training run used it.
2. **Float32 quantisation in the loss scalar.** A Python float or a `np.float64` scalar is not an ndarray, so the engine routed it
   through `xp.asarray(..., _DEFAULT_DTYPE)` and the loss value became float32 - quantised to ~2.4e-7 steps, which made the
   finite-difference check read a **staircase** and look like a broken gradient. A 0-d float64 ndarray fixed it: gradient check now
   passes at **4.9e-11**.
3. **Entropy as a constant bonus.** The first fix added `-beta * H`, worth 0.10-0.42 nats against a policy loss of 0.2-0.3, so it
   dominated and dragged the policy toward uniform. Replaced with a hinge that only acts below a floor.

## Results, three seeds each, 400 RL steps on top of a 400-step supervised warm start
```
arm                          in-distribution accuracy   seen-sum generalisation   calibration (ECE)
supervised, 800 steps         51.12 +/- 3.30 %                 0.00%                0.131
supervised, 400 steps (warm)  20.93  (17.89 / 21.96 / 22.93)    0.00%                -
RL, entropy bonus 0.05        20.56 +/- 3.69 %                 0.00%                0.219
RL, entropy bonus 0.20        22.31 +/- 0.69 %                 0.00%                0.109
RL, entropy hinge 0.5         18.84 +/- 1.00 %                 0.00%                0.278
RL, entropy hinge 2.0         18.84 +/- 1.00 %                 0.00%                0.281
```

## The correction I have to make against myself
I reported mid-session that RL "destroyed 29 accuracy points" (51.12% -> 20.56%). **That comparison was wrong.** The RL arms do
400 supervised steps and then 400 RL steps; 51.12% is what 800 supervised steps achieve. Against the fair baseline - the 400-step
warm start at **20.93%** - RL at 400 steps scores **18.84-22.31%**, i.e. **roughly neutral, slightly negative**. It never cost 29
points; it never gained anything either. The same mistake in reverse of the one I caught earlier today: comparing two arms that
differ in two ways.

## Why the entropy betas gave identical numbers
`entropy hinge 0.5` and `2.0` produced identical per-seed accuracies, which normally means the patch did not apply. I instrumented
the term instead of assuming: after 400 supervised steps the policy entropy is **2.46 nats** against a floor of **1.20** and a
maximum of 2.64, so the hinge is legitimately **zero** and both betas do nothing. That is a real fact about the model, not a bug.

## Verdict, stated the way the project requires
**Our RL method still does not beat supervised learning, for the second time, now with a correct implementation.** The mechanism
is measurable: the supervised signal is dense (every answer token is labelled) and the RL reward is sparse (one bit per sampled
sequence), so 400 RL steps buy less than 400 extra supervised steps. RL is roughly neutral on top of a warm start.

## What would have to change for RL to win here
1. **Reward density.** One bit per sequence is not enough on a 13-way vocabulary. Dense credit assignment (per-token rewards from a
   verifier, or a curriculum where answers are single digits first) is the standard fix.
2. **A converged warm start.** RL is normally applied on top of a model that has already converged, not one trained for a third of
   the steps.
3. **A task where RL has an advantage at all** - i.e. where the correct answer is not simply imitable token by token. On addition
   supervised learning already has the exact right answer as a label, so RL can only lose.

That third point is the honest one to keep: for a task with dense labels, RL has no structural advantage, and the RL argument
belongs to tasks where a checker exists but labels do not.

## Reproduction
`tools/rl_compare.py --arm {supervised,rl} [--sft_steps N] [--entropy_beta B] --steps N --seed S`; arms launched in parallel by
`lens/scripts/run_rl_compare.sh`, `run_rl_v2.sh`, `run_rl_v3.sh`; logs in `logs/rl/`, results in `runs/rl/*.jsonl`.
Gradient checks: `tools/gc_rl.py`, `tools/gc_entropy.py`, `tools/gc_entropy_floor.py`.
