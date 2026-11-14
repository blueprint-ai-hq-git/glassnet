# Our RL is not losing to supervised - it is training on an empty support

Date: 2026-11-14. Three measurements that replace the sentence "our RL is worse" with a cause. Tools: `tools/fuel_check.py`,
`tools/fuel_by_sum.py`. Data: `lens/runs/RL-FUEL-CAPACITY.txt`, `lens/runs/RL-FUEL-VS-CAPACITY.txt`.

## The starting point

Thirteen consecutive arms measured `acc_gen = 0.00` on the generalising split (a+b >= 10): nine previous RL variants plus
three LENS-PG v2 attempts. Every one of them reported a normal-looking loss curve while doing it.

## Step 1: is there anything correct to reinforce at all?

Warm start 400 supervised steps on sums <= 9 (the shipped setting), then 200 generalising prompts x **G = 64** samples =
**12,800 drawn answers**, each scored by exact integer match:

```
answers drawn              12,800
correct answers                 0   (0.0000%)
prompts with >=1 correct    0 of 200
```

## Step 2: is it just too little training?

Same measurement with the warm start lengthened:

```
warm start   400 steps  -> 0 correct of 12,800
warm start  2000 steps  -> 0 correct of 12,800
warm start  8000 steps  -> 0 correct of 12,800
```

Training length is not the limit.

## Step 3: is it just too little capacity?

`TinyAdd(d, hidden)` is parameterised, so capacity was swept directly:

```
d 16 hidden  64    7,342 parameters  -> 0 correct of 12,800
d 32 hidden 128   26,958 parameters  -> 0 correct of 12,800
d 64 hidden 256  103,054 parameters  -> 0 correct of 12,800      (14x the shipped model)
```

Capacity in this range is not the limit either.

## Step 4: is the fuel merely thin, or absent?

The last possibility was that correct answers exist for the EASIEST two-digit sums, where a curriculum could start and
expand. Fuel was measured per target sum, 60 prompts each at G = 64 (3,840 draws per sum):

```
sum   10    11    12    13    14    15    16    17    18
rate  0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%
```

**The support is empty, not thin.** 34,560 draws across the whole generalising split, zero correct answers, including the
smallest two-digit sum.

## What this means, and what it does not

**It means the 13 zeros are not a property of our estimator.** Group-relative advantage, dense per-token credit, group-32
sampling, KL anchors, variance gates, curricula and repetition penalties were all irrelevant on this task: a reward the model
never attains carries no gradient, and no estimator can manufacture fuel. This is the standard exploration-failure result,
measured here rather than assumed.

**It does not mean the task is unreachable in principle.** Supervised training on both digits reaches
**34.66% +/- 6.16** on the same split. The honest statement is: *reachable when taught, unreachable by sampling at every
capacity and schedule we tried.*

**It does mean our RL needs a different task, not more tuning.** Verifier-only RL is only testable where random behaviour
sometimes succeeds. Addition taught only sums <= 9 does not satisfy that, at any size tried. A testable replacement needs a
task where a wrong-but-plausible sample has a real chance of passing the checker - for example multiple choice (chance is
1-in-10), a format rule (satisfiable by luck), or copying a token from the prompt. Until then, RL results on this task are
measurements of an empty reward.
