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
