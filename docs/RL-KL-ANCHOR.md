# The KL anchor stops the damage: RL is no longer destructive, but it still is not competitive (2026-11-13)

`docs/RL-DEGRADES.md` measured the standard recipe failing: from a warm start that could answer (acc_gen 27.51 average), 400
unanchored policy-gradient steps cost **-5.94 points**, worse in 3 of 3 seeds. Nothing kept the update near the policy that worked,
and the entropy term was the only thing holding the distribution. The standard remedy is an anchor, and this is that test.

## Design
```
loss = policy_gradient(group-relative, group 8) + entropy_floor(0.5) + beta * KL(policy || frozen warm-start policy)
```
The KL term is `glassnet/rl.py: kl_to_fixed`, and its gradient was **checked numerically before use** (the project's own rule):
max abs **6.096e-11**, rel **1.557e-09**, PASS. Reference model is a frozen copy of the parameters taken before step 1.

**Prediction, written before the run:** at beta 5 and beta 20, `acc_gen` stays within 2 points of its own warm start in every seed,
higher beta holds closer than lower, and any arm improving by >= 3 points is doing real work.
**Falsifier:** both betas losing more than 5 points on average means an anchor does not save this update at this model size.

## Result
```
arm          seed  warm start  after RL   delta
beta 5        0      32.98      30.37     -2.61
              1      22.96      29.08     +6.12
              2      26.60      25.00     -1.60
              mean delta +0.64 +/- 4.78    mean after 28.15

beta 20       0      32.98      32.98     +0.00
              1      22.96      29.08     +6.12
              2      26.60      26.60     -0.00
              mean delta +2.04 +/- 3.53    mean after 29.55

unanchored    mean delta -5.94   (docs/RL-DEGRADES.md)
supervised    acc_gen 34.66 +/- 6.16   (docs/RL-REACHABILITY.md)
```

**The falsifier did NOT trigger: the anchor removes the damage.** From -5.94 unanchored to **+0.64** (beta 5) and **+2.04** (beta 20)
mean delta. That is the first non-negative RL result in this entire line.

**The prediction holds only partly and is reported that way.** Beta 5 exceeds the 2-point bar on seed 0 (-2.61) while seed 1 gains
+6.12; beta 20 sits at 0.00 on two seeds and +6.12 on the third. The honest reading of the beta-20 zeros is that a strong anchor
mostly means **the policy does not move** - "not degrading" is being bought by "not learning", and a 0.00 delta is that, not a win.

## Verification done before believing the numbers
Seed 1 shows **+6.12 at both beta values**, and seeds 0 and 2 at beta 20 are unchanged to the decimal - the exact pattern that has
meant "the flag did not apply" three times in this project. Checked explicitly:
- every log prints `KL anchor active: beta 5.000 / 20.000` - the flag reached all six runs;
- the two seed-1 training traces **differ** (mean reward 0.070 vs 0.057 at step 400), so they are genuinely different runs;
- the identical delta is a **coarse-metric coincidence**: `acc_gen` has 214 items, so 22.96 -> 29.08 is 13 items, and two different
  policies can land on the same count.

## Where the RL line now stands, all measured
```
starting policy        reward/update                    result
cannot answer          binary, unanchored               advantage exactly zero, no learning
cannot answer          per-digit, unanchored            answers '1' to everything, acc_in 22 -> 2.34
cannot answer          shape bonus, unanchored          answers '98', acc_in -> 0.00
CAN answer             binary, unanchored               -5.94 points, worse in 3 of 3 seeds
CAN answer             binary, KL anchor (5 / 20)       +0.64 / +2.04, i.e. no damage
supervised, same data  -                                34.66 +/- 6.16
```

**Conclusion, stated plainly:** the anchored update is **safe but not useful** on this task. It stops RL from destroying a working
capability; it does not beat supervised learning, and on two of three seeds at the strong setting it does not change the model at
all. RL has not beaten supervised here in any of eight configurations, and the two failures that are not reward hacks are "cannot
start" and "does not improve".

Per the falsifier committed to before this run: the line is **not** abandoned as broken - the anchor works - but it is recorded as
**not competitive with supervised learning at this model size and this task**, and it stops being tuned. The next work goes to the
data judge, which still fails its own gate (false alarm 0.0730 against a 5% bar, one-token loops 69.5% against 80%), and that is an
engineering problem with a measurable pass condition rather than another reward shape.

## Reproduction
`lens/scripts/run_rl_kl.sh` (prediction in the header), logs `logs/rl_kl/b{5.0,20.0}_s{0,1,2}.log`, `--kl-beta` in
`tools/rl_compare.py`, `glassnet/rl.py: kl_to_fixed`.
