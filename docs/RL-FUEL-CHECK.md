# The fuel check: zero correct samples in 12,800 draws, so no self-training method could have started

Date: 2026-11-14. Tool: `tools/fuel_check.py`. Follows `docs/history/RL-METHOD-V2.md`, where three LENS-PG v2 attempts plus
nine earlier arms all measured `acc_gen = 0.00` - **13 consecutive zeros** on the generalising split (a+b >= 10).

## The question underneath the 13 zeros

Every one of those arms trained a policy on a verifiable reward. If the model NEVER samples a correct answer, then no
self-training method can start - not policy gradient, not rejection-sampling fine-tuning, not STaR - because there is
nothing correct to reinforce. A zero gradient would then be a consequence of an always-wrong reward, and no estimator can
manufacture fuel that is not there.

So this run trains nothing. It measures whether the fuel exists.

## Protocol (mirrors the harness, so the numbers are comparable)

- warm start: 400 supervised steps, lr 0.05, batch 16, on sums <= 9 **only** - the same warm start the RL arms used, which
  is exactly why `acc_gen` is 0.00 after it: the model has never been shown an answer of 10 or more;
- then 200 gen prompts (a+b >= 10) x **G = 64** samples at temperature 1.0 = **12,800 drawn answers**;
- a draw counts as correct when `rl.reward_of()` returns 1.0 (exact integer match, repetition punished).

## Result

```
answers drawn              12,800
correct answers                 0   (0.0000%)
prompts with >=1 correct    0 of 200   (0.00%)
prompts inside the 0.10-0.90 band      0
mean per-prompt success     0.0000%    max 0.00%
```

**P1 (at least 1% of prompts produce a correct sample) - FAILED.**
**P2 (mean rate above 0.1% but below the band) - FAILED.**
**FALSIFIER TRIGGERED: zero correct samples in 12,800 draws.**

## What this establishes, and what it does not

**Established:** with the warm start this project actually uses (400 steps, a 7,342-parameter model taught only sums <= 9),
the generalising split produces **no correct sample at all** in 12,800 tries. The 13 zeros are therefore not evidence about
our advantage estimator, our curriculum, our repetition term or our KL anchor. There was nothing to reinforce. Every one of
those arms was measuring an empty reward.

**Not established:** that the task is unreachable in principle. It is not - supervised training on both digits reaches
**34.66% +/- 6.16** on this split (`docs/history/RL-DEGRADES.md`). So the gap is specifically
**reachable-when-taught, unreachable-when-sampled at this capacity**. A 7,342-parameter model that has never seen a
two-digit answer does not stumble onto `8+7=15` by chance.

**What that means for the plan:** the next question for the RL line is not a better objective, it is **how much capacity
the fuel needs**. That is measurable and cheap: sweep model size (and warm-start length) on CPU and find the smallest
setting where at least one correct sample appears at G = 64. Below that size no verifier-only method can be tested at all,
and reporting RL numbers on it is reporting noise.
