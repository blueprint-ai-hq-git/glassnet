# Reward the shape and the shape gets hacked: a constant two-token nonce (2026-11-13)

Last arm of the carry investigation. `docs/RL-REWARD-HACK.md` ended with a structural blocker: every answer the model was ever
trained on is one token, so a two-digit answer needs a token pattern it has never emitted, and no digit credit can be collected
until that pattern appears. So this arm stops rewarding digits and rewards the **structure**: sparse verifier reward plus **+0.2 for
emitting any two-token answer**, correctness ignored. Unlike a per-digit correction, this is not constant within a group, so a
group-relative advantage can actually act on it.

## Prediction, written before the run
Two-token share above 20% (from ~0), and `acc_gen` off exactly 0.00 in at least one seed. Falsifier: if two-token answers appear
but `acc_gen` stays 0.00 in every seed, verification can bootstrap the SHAPE but not the ABILITY.

## Result - the falsifier triggered, and the reward was hacked again
```
seed   acc_in   acc_gen   probe answers after training (7 probes each)      confidence
0       0.00     0.00     '98' on all 7 probes                              0.31 - 0.67
1       0.00     0.00     '83' on 6 probes, '89' on 9+0                     0.28 - 0.54
2       0.00     0.00     '81' / '84'                                      0.35 - 0.68
mean reward during training, seed 0: 0.168 at step 400 against a 0.2 bonus
```

- **The prediction's first half PASSED**: two-token answers appear on every probe, in every seed, where none existed before.
- **The second half FAILED**: `acc_gen` is 0.00 in all three seeds, and `acc_in` is **0.00** - worse than the 'answer 1' collapse,
  which at least kept 2.34%.
- **Mean reward 0.168 against a 0.2 bonus** says exactly what is happening: the model is collecting the shape bonus and nothing
  else. It found the cheapest two-token answer (a constant, different per seed) and stopped. Rewarding the structure produced a
  structural nonce, not the beginning of an ability.

This is the third distinct reward hack in this investigation, each one shaped by the reward it was given:
```
reward                      what it paid for                  what the model did                     cost
binary verifier             nothing (advantage 0)             could not start                        -
per-digit partial credit    a right first digit               answer '1' to everything (conf .83-.89) acc_in 22 -> 2.34
per-digit + frequency fix   (cancels in group normalisation)  unchanged                              none
+ 0.2 shape bonus           any two tokens                    answer '98' / '83' / '81' to everything acc_in -> 0.00
```

## The conclusion this closes
**On this task and this model, RL from verification cannot produce carry.** Not because of the implementation, the reward
resolution, the sample count, the learning rate, the entropy term, the data mix, the reward's frequency bias, or the reachability
of the output structure - every one of those has now been measured and excluded. What RL did produce, twice, was the cheapest
thing that satisfied the reward.

Two facts force the next question, and they are the reason this line stops here:
1. **`acc_gen` is 0.00 for every method ever run on this task** - binary verification, partial credit, shape bonus, and 800
   supervised steps at `docs/RL-GEN-VERIFY.md`.
2. Nothing has yet shown the task is **reachable at all** at 7,342 parameters.

Until something reaches non-zero accuracy on sums >= 10, any statement of the form "our RL cannot learn carry" is a statement
about an unsolvable task. So the next arm is not another reward: it is a **reachability check** - supervised training with
two-digit answers in the data, which is the only method here that has ever been given the structure it needs. If that reaches a
real number, the task is fair and RL can be measured against it honestly. If it does not, the task is beyond this model and must be
dropped before another RL arm is spent on it.

## Reproduction
`lens/scripts/run_rl_len.sh` (prediction in the header), logs `logs/rl_len/b02_s{0,1,2}.log`, probes printed by `--diag` in
`tools/rl_compare.py`, `--len-bonus` in the same file.
