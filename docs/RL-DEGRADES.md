# RL does not hold a working capability either: it degrades it in all three seeds (2026-11-13)

`docs/RL-REACHABILITY.md` removed the last excuse: the carry task is reachable (~35% supervised on sums >= 10), so RL can be
measured against a real target instead of an impossible one. Every previous RL arm had started from a model that could not emit a
two-digit answer, which made the binary verifier unsatisfiable and the advantage exactly zero. This arm removes that too: **warm
start on sums >= 10 first, so the model can answer, then binary-verifier RL on the same distribution.** The reward is satisfiable
and is actually being collected.

## Prediction, written before the run
`acc_gen` stays >= 25% and improves by >= 3 points over the warm-start reference in at least two of three seeds.
**Falsifier:** below 20% means RL destroys a working capability even when its reward is satisfiable - a harder negative than
anything measured so far, because it would rule out the standard recipe on this task too.

## Result - it degraded, in every seed
```
seed   acc_gen after warm start   acc_gen after 400 RL steps   delta     acc_in after RL
0              32.98                       19.37              -13.61          0.00
1              22.96                       21.94               -1.02          0.00
2              26.60                       23.40               -3.20          0.00
mean           27.51                       21.57               -5.94
```

**Zero of three seeds improved. All three got worse.** Seed 0 lost 13.61 points - more than a third of the capability - and the
prediction's "stays >= 25%" bar was missed in two of three seeds (19.37 and 21.94 below it; seed 2's 23.40 also below).
`acc_in` remains 0.00, consistent with the capacity finding in `docs/RL-REACHABILITY.md`: this model holds one distribution at a
time.

So the RL line now stands like this, every row measured:
```
starting point                        reward                result
cannot emit two-digit answers         binary                advantage exactly zero, no learning
cannot emit two-digit answers         per-digit              answers '1' to everything, acc_in 22 -> 2.34
cannot emit two-digit answers         shape bonus           answers '98' / '83' / '81', acc_in -> 0.00
CAN emit two-digit answers (33%)      binary                acc_gen -5.94 on average, worse in 3 of 3 seeds
supervised on the same data, same compute                    acc_gen 34.66 +/- 6.16
```

**RL has not beaten supervised on this task in any configuration, and the last configuration is the standard one.** The failure is
not "cannot start" any more, and it is not a reward hack: with a satisfiable reward and a capable starting policy, 400 policy-gradient
steps still cost accuracy. That points at the update itself - sampling noise in a group of 8, no constraint keeping the policy near
the one that was working, and an entropy term added on top.

## The next arm, and why it is this one
The standard remedy for exactly this failure is an **anchor**: add a KL penalty against the warm-start policy, so the update cannot
wander far from a policy that works, and the entropy floor is no longer the only thing holding the distribution. That is the next
arm and its prediction is written before it runs. If a KL-anchored update still cannot improve on its own starting point, then the
honest conclusion is that this RL implementation, at this model size and this reward, is not competitive with supervised learning
and the line should stop rather than be tuned further.

## Reproduction
`lens/scripts/run_rl_warm.sh` (prediction in the header), `--warm-dist gen` in `tools/rl_compare.py`, logs
`logs/rl_warm/genwarm_s{0,1,2}.log`, warm-start reference `logs/rl_reach/sup_gen_s{0,1,2}.log`.
