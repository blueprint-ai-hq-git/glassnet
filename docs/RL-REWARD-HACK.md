# The collapse was reward hacking: partial credit taught the model to always answer '1' (2026-11-13)

`docs/RL-GEN-VERIFY.md` reported that per-digit credit on sums >= 10 took in-distribution accuracy from a ~22% warm start to 3.13%
and said "the RL phase destroyed what the warm start knew" without naming the mechanism. This names it, and rules out three
variables on the way.

## Ruling out the obvious causes first
Every configuration below was run on the same task and the same seeds, and produced **bit-identical** acc_in:
```
configuration                                               acc_in seed 0   acc_in seed 1
per-digit reward, lr 0.02                                    2.34            2.93
per-digit reward, lr 0.005                                   2.34            2.93
per-digit reward, lr 0.02, RL batch = in-dist + >=10 (609)    2.34            2.93
per-digit reward, lr 0.02, entropy_beta 0.0                   2.34            2.93
```
When changing a variable leaves the output identical to two decimals, that variable is not the cause. So:
- **not the learning rate** (a 4x lower lr changes nothing; the flag was confirmed present in the result JSON as `lr 0.005`),
- **not single-distribution training** (mixing in-distribution pairs into the RL batch changes nothing),
- **not the entropy floor** (beta 0.0 changes nothing, even though it was the only term with a non-zero gradient once the reward
  went to zero - the hypothesis that looked most promising from the reward trace).

## What it actually is, seen rather than guessed
Rather than reason further, the run now prints the model's greedy answers on a fixed probe set after training:
```
3+4 = '1' (true 7)  conf 0.83        8+7 = '1' (true 15) conf 0.86
5+2 = '1' (true 7)  conf 0.84        6+6 = '1' (true 12) conf 0.89
9+0 = '1' (true 9)  conf 0.88        4+2 = '1' (true 6)  conf 0.87
1+1 = '1' (true 2)  conf 0.87
```

**The model answers `'1'` to every question at 83-89% confidence.** That is the mechanism:

- On sums >= 10 the correct first digit is **always 1** (10 through 18). So the cheapest way to collect the position-0 part of a
  per-digit reward is to emit `'1'` unconditionally, and that is exactly what the policy found.
- The position-1 part of the reward is unreachable, because the model never emits a second token at all.
- The shortcut is worth more than everything the warm start knew, so in-distribution accuracy falls to the rate of pairs whose sum
  is 1 - which is what 2.34% is.

This is **reward hacking of a partial-credit reward**, not a bug in the optimiser, and not the entropy term. It is the same failure
mode the project already knows from other arms: a proxy that can be satisfied without doing the task gets satisfied.

## What this means for the RL line
- Binary verification gives **zero advantage** (all group members equal on a task nobody can yet solve), so it cannot start.
- Partial credit gives signal, and the signal is **gameable**, so it produces a shortcut rather than the ability.
- Neither failure is about learning rate, sample count, entropy or reward resolution - all four have now been measured.

The honest requirement for the next reward: **it must not pay for a constant guess.** The obvious candidate is a frequency-corrected
per-digit reward - subtract, per digit position, what always guessing the most common digit would collect, so `'1'` is worth nothing
there - and that is the next arm, with its prediction written before it runs. The stronger alternative is to stop rewarding digits
altogether and find a reward that only pays for the whole correct answer while making the second token reachable at all, because at
present the model never emits one.

## Reproduction
`lens/scripts/run_rl_gen3.sh` (entropy test), `lens/scripts/run_rl_gen2.sh` (lr and mixing), `--diag` in `tools/rl_compare.py`
(the probe printout). Logs: `logs/rl_gen3/noent_s{0,1}.log`, `logs/rl_gen2/{lr005,mix}_s{0,1}.log`, `logs/rl_gen/dense_s*`.
