# 2026-11-25 - the fair test: at EQUAL EPOCHS the 1M pool wins by 0.485

Source: lens/runs/EPOCH-FAIR.out (exit 0), lens/runs/EPOCH-FAIR.json, per-arm lens/runs/EPOCH-FAIR_200000.json and
EPOCH-FAIR_1000000.json. Built on lens/harness.py, so both arms share ONE fixed held set and the numbers ARE comparable.

    arm                  steps     epochs    held-out    time    params
    200,000 pool        50,000       2.00     4.23065     67s   5,476,816
  1,000,000 pool       250,000       2.00     3.74519    348s   5,476,816

Difference -0.48546 in favour of the bigger pool, at matched exposure.

## Why this test had to exist

Every earlier pool comparison was at equal STEPS, which penalises the larger pool because it has barely started: the 1M arm
was at 0.41 epochs when it "won" by 0.0607 in DATA-SCALE.out. That was suggestive, not fair. Giving each pool the same
number of passes over its own data removes the bias, and the effect got EIGHT TIMES LARGER (0.0607 -> 0.48546).

## What it does and does not say

It says: distinct data was still the strongest lever available, and 200k was not a ceiling.
It does not say anything about retention - this is the same weights learning more of the same distribution, which is exactly
the regime where catastrophic forgetting does NOT show up. That is why the next experiment is the forgetting test, which
measures the LEARNING METHOD rather than the data.
