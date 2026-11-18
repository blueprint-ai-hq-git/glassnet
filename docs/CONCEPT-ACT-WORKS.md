# The activation penalty works: 30/36 -> 34/36 concepts honest, and it names what is left (2026-11-18)

The audit said the readable chat model's neurons fire **47.4%** of the time while their concepts have supports of 0.6-14%, so I raised `--lam-act`, a dial that had been **0.0 in every
run this project has ever done**. Three arms, readable d512, same recipe, seed 0:

```
lam-act   pass     mean AUROC   mean fire rate   failing concepts
0.0      30/36      0.6886         47.4%         is_a_list, mentions_a_name, mentions_a_quantity, mentions_body, mentions_family, mentions_sport
0.3      32/36      0.7038         36.2%         mentions_a_name, mentions_a_quantity, mentions_body, mentions_sport
1.0      34/36      0.7370         20.1%         mentions_body, mentions_sport
```

## Predictions, scored

- **P1** "mean fire rate falls below 25% at lam-act 1.0" - **HOLDS**: 20.1%, less than half the starting rate.
- **P2** "mean AUROC rises above 0.80" - **FAILED**: 0.7370. The dial helps, and it does not reach the target.
- **P3** "at least 33/36 concepts pass" - **HOLDS**: 34/36.
- **FALSIFIER** "mean AUROC below 0.72 at lam-act 1.0" - **did not fire**, but only just: 0.7370 is 0.017 above the line.

## What the remaining two failures have in common

`mentions_body` (support 1.32%) and `mentions_sport` (1.43%) are among the **rarest** concepts in the set. But rarity alone is not decisive: **`mentions_a_name` has the lowest support
of all (0.82%) and now passes** at lam-act 1.0, having failed at 0.0. So the last two are not simply "too rare to learn" - something about these two formulas is harder to detect from a
flattened 32-token window.

Honest status: **34 of 36 concepts (94.4%) are label-honest**, up from 30 of 36 (83.3%), against 36 of 36 for a matched model with a direct path. The gap to the copy-path model has
narrowed from 6 failures to 2.

## The next test, and why it is the last one on this dial

`lens/scripts/run_concept_act2.sh` runs lam-act 1.5 and 2.0:

- **P1** mean AUROC keeps climbing: at least 0.76 at 2.0 (trend so far 0.6886 -> 0.7038 -> 0.7370)
- **P2** no new failures appear - the failing set stays inside {mentions_body, mentions_sport}
- **P3** at least 35/36 pass
- **FALSIFIER** AUROC at 2.0 not higher than 0.7370: the dial has saturated, and the remaining two concepts need **per-concept positive weighting** in the concept loss (which the vision
  model already has via `pos_weight`) rather than more uniform pressure.

The reason this is the last test on this dial: a uniform pressure that keeps rising eventually stops helping the rare concepts and starts suppressing the common ones, so if 2.0 does not
show progress the right move is a **per-concept** mechanism, not a bigger number.
