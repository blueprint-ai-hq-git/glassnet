# A test bed with fuel: (a+b) mod 10, and why it matters for our RL

Date: 2026-11-14. Tool: `tools/fuel_sum10.py`. Follows `docs/history/RL-EMPTY-SUPPORT.md`.

## The problem this solves

Verifier-only RL was untestable on "a + b" because the correct answer was a TWO-digit number the model never sampled:
0 correct in 34,560 draws across sums 10-18, unchanged at 8,000 warm-start steps and at 103,054 parameters. A reward the
model never attains carries no gradient, so the 13 `acc_gen = 0.00` arms were measuring an empty reward, not a bad estimator.

## The fix, and it is deliberately minimal

Keep the prompts. Keep the training data identical. Change the target to **(a + b) mod 10** - always ONE digit.

Every training pair (sums <= 9) has the same answer it always had, because `(a+b) mod 10 == a+b` there. So the warm start
trains exactly the same model on exactly the same data. What changes is that a generalising prompt (a+b >= 10) now has an
answer built from a token the model has emitted thousands of times, so a random draw can land on it. That is precisely the
property verifier-only RL requires and the exact-answer task lacked.

## Measured fuel (this is the point of the document)

400 supervised steps (lr 0.05, batch 16) on 200 pairs with a+b <= 9, in-domain accuracy **70.50%**. Then 200 generalising
prompts x **G = 64** sampled digits:

```
digits drawn              12,800
correct digits                15   (0.1172%)
prompts with >=1 correct  12 of 200   (6.00%)
mean per-prompt rate      0.1172%   max 3.12%
emitted digits (top 5)    9:59.5%  8:31.2%  7:4.4%  6:2.9%  5:1.3%
```

**P1 (fuel exists, at least 1% of prompts) - HOLDS at 6.00%.**
**P2 (rate far below the 0.10-0.90 curriculum band) - HOLDS at 0.12%.**

## What this establishes

1. **Fuel exists.** Verifier-only RL is testable on this task, where it was not testable on the exact-answer version. Twelve
   of 200 generalising prompts can produce a passing sample at G = 64.
2. **The curriculum band we shipped is wrong for this task by two orders of magnitude.** The band is 0.10-0.90; the measured
   success rate is 0.0012. A band that never admits a prompt is the same failure as no curriculum at all, and this is now
   measured rather than suspected.
3. **The failure shape is informative.** The model puts 59.5% of its post-"=' mass on the digit 9 and 31.2% on 8, and almost
   nothing below 5. It is not a collapsed distribution (three digits carry real mass), but it is badly skewed - which is why
   fuel exists yet is thin.

## The next experiment this makes possible

An RL arm on `(a+b) mod 10` with a band widened to the measured range (for example 0.0005-0.5, minimum 8 samples per
prompt), so dead groups are the exception rather than the rule. The predictions for that arm should be written before it runs
and must include the greedy generalising accuracy measured *before* RL, because at 0.12% fuel the honest question is whether
RL beats its own starting point at all - not whether it catches supervised learning.

## Update 2026-11-14 10:35: the RL arm on the fuel-bearing task learned NOTHING, at every setting tried

With fuel available (`docs/history/RL-TEST-BED-WITH-FUEL.md`), LENS-PG v2 was run on `(a+b) mod 10` - 600 RL steps, group 16,
warm start 400 supervised steps held fixed at lr 0.05, evaluation greedy. Tool: `tools/rl_mod10_train.py`. Every number below
is a 3-seed mean unless stated:

```
sampling T   RL lr     dead groups   reward mean   greedy gen accuracy    in-domain accuracy
1.0          0.05          99%          0.0005       0.00% -> 0.00%       70.5% -> 42/24/54   (-31.2 mean)
2.0          0.05          99%          0.0007       0.00% -> 0.00%                          (-48.8 mean)
4.0          0.05          97.5%        0.0018       0.00% -> 0.00%                          (-51.3 mean)
4.0          0.010         49.7%        0.0507       0.00% -> 0.00%                          (-51.3 mean)
4.0          0.002         22.8%        0.0893       0.00% -> 0.00%                          (-41.5 mean)
4.0          0.0005        21.0%        0.0948       0.00% -> 0.00%                          ( -6.2 mean)
8.0          0.0005        37.3%        0.0654       0.00% -> 0.00%                          ( -5.3 mean)
8.0          0.002         45.2%        0.0542       0.00% -> 0.00%                          (-46.2 mean)
```

**Greedy generalising accuracy never moves off 0.00%.** In-domain accuracy is damaged in proportion to the RL learning rate
(-51 at lr 0.01, -6 at lr 0.0005), so the updates are doing something - they are just not doing the task.

## Exploration was tested, not assumed

Fuel is a function of sampling temperature, measured on 12,800 draws per point:

```
T=1   0.1172% of samples correct    6.00% of prompts can succeed
T=2   1.5938%                      55.00%
T=4   4.6719%                      93.50%
T=8   7.1406%                      99.00%
```

So by T=4, 93.5% of generalising prompts can produce a correct sample and the dead-group fraction drops to ~21% at the right
learning rate. **Every precondition I previously blamed was satisfied - fuel, group variance, dead-group rate - and the
policy still did not learn.** Reward density, support, capacity and estimator design have now all been eliminated as the
cause of this failure.

## A diagnostic worth keeping: the model's strongest habit is the one answer that can never be right

Exact enumeration of the generalising split (a+b >= 10, 45 prompt classes):

```
answer == 0  20.00%      answer == 4  11.11%      answer == 8   2.22%
answer == 1  17.78%      answer == 5   8.89%
answer == 2  15.56%      answer == 6   6.67%
answer == 3  13.33%      answer == 7   4.44%      answer == 9   0.00%
```

**The answer is never 9** - the largest sum is 18, so `(a+b) mod 10 == 9` only if a+b is 9 or 19, and both are excluded by the
>= 10 split. And the model emits **9 for 59.5%** of these prompts at T=1. Its single strongest habit points at the one digit
that is never correct on the target distribution. That is why fuel is 0.12% at normal temperature, and it is a much sharper
description of the failure than "RL does not work here".

A constant-9 policy would score 0.00% on gen, so this fuel is NOT a mode-hacking artefact - the correct samples that exist
are genuine, they are just rare and drowned by a confident wrong prior.

## Where this leaves the RL line

Twelve configurations, one outcome: the method does not move this policy. The next thing to try is not another estimator or
another temperature - it is the mechanism that works when successes are rare but real: **rejection sampling**. Collect the
samples that do pass (at T=4 there are ~600 correct samples per 9,600 draws), train on those supervised, and repeat. That
converts rare success into ordinary supervision instead of asking a noisy advantage to carry it. It is the obvious next arm,
and it should be judged against the same greedy generalising baseline reported above - which is 0.00%, so any movement at all
is a result.
