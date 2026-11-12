# Linear-rule ladder: the price of honest concept names falls from 3.68 points to 1.18 (2026-11-12)

Every earlier calibration on this task - the rule-count sweep and the concept-weight dial - was done with **sigmoid** rules.
Linear rules changed the landscape (81.19% against 76.10% at the same settings), so the grid was retrained from scratch:
R in {16, 32, 48} x alpha in {1, 3} x 3 seeds, each arm pruned, with accuracy, concept honesty and concept influence measured in
one pass. 18 arms, all numbers below from `runs/phase2_sweep/linear_ladder.csv`.

```
alpha  rules  pruned accuracy   mean |rho|   misnamed   rules to read   min influence   mean influence
1.0     16    81.20 +/- 1.26       0.451      6.7 of 12       13.7            0.00%           23.4%
1.0     32    82.37 +/- 0.76       0.422      7.3 of 12       20.3            0.00%           22.8%
1.0     48    81.62 +/- 2.31       0.463      6.0 of 12       27.3            0.00%           22.4%
3.0     16    72.10 +/- 6.85       0.603      2.3 of 12       14.0            0.17%           27.8%
3.0     32    81.19 +/- 0.98       0.640      2.3 of 12       20.0            0.00%           23.6%
3.0     48    78.79 +/- 5.32       0.632      1.7 of 12       25.3            0.00%           23.2%
```

## The prediction and what happened
**Predicted:** with linear rules the alpha-3 penalty largely disappears (rules cannot saturate), so alpha 3 stays within 2
points of alpha 1 at the same rule count, and the ladder's best arm beats 81.19%.
**At R=32 it held precisely (cost 1.18 points). At R=16 and R=48 it did not (9.10 and 2.83).** The best arm (82.37 +/- 0.76)
does beat 81.19. So the saturation explanation is right about the mechanism and wrong about universality: the honesty price is
small only where the rule layer has enough capacity to absorb the constraint, and R=32 is that place.

## Price of honest concept names, measured three ways on this task
```
configuration                          accuracy cost   misnaming improvement
sigmoid rules, R=16 -> concept weight   -9.88 points   4.7 -> 2.3 of 12
sigmoid rules, best honest vs best     -3.68 points   7.0 -> 1.3 of 12
LINEAR rules, R=32                     -1.18 points   7.3 -> 2.3 of 12
LINEAR rules, R=48                     -2.83 points   6.0 -> 1.7 of 12
```
At R=32 with linear rules, honest concept names cost **1.18 points** - a third of the sigmoid price and an eighth of the original
concept-weight dial price, for the same halving of misnaming.

## What to ship from this grid
**`alpha 3 / 32 rules / linear / pruned` = 81.19% +/- 0.98, 2.3 of 12 concepts misnamed, 20 rules a human must read.**
- Within **1.18 points** of the absolute best accuracy in the grid (82.37 +/- 0.76, which has 7.3 of 12 names wrong).
- Beats the original Phase 2 gate (79.04% +/- 2.38, 30 rules) by **2.15 points with ten fewer rules to read**.
- **R=32 is the only stable honest setting**: R=16 gives 72.10 +/- 6.85 and R=48 gives 78.79 +/- 5.32, both unusable as a
  ship. Stability, not just accuracy, points at 32.

## What is still broken
**Dead concepts survive every lever so far.** Minimum concept influence is 0.00% in four of these six configurations, and the
best any setting manages is 0.17%. Linear rules removed saturation - one of the three sub-causes identified in
`docs/CONCEPT-DROPOUT-AND-MECHANISM.md` - and the other two (a concept's last edge being pruned away, and rules that are
inert) are untouched. Coverage-aware pruning is the next lever and it is now the only one of the three not yet tested.

## Reproduction
`tools/linear_ladder.py` (drives train -> prune -> audit -> influence for all 18 arms, writes `linear_ladder.csv`);
`lens/runs/LINEAR-LADDER.txt` for the run log. Eighteen checkpoints kept under `runs/phase2_sweep/L_A*_R*_s*/`.
