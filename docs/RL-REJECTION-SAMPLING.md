# Rejection sampling is the first method here that creates generalising ability from nothing

Date: 2026-11-14. Tool: `tools/rl_reject.py`. Task: `(a+b) mod 10`, the fuel-bearing test bed from
`docs/history/RL-TEST-BED-WITH-FUEL.md`. Warm start: 400 supervised steps on sums <= 9 (in-domain 70.5-72.5%, generalising
accuracy **0.00%** - the model has never been taught a sum of 10 or more).

## What was tried, and why this and not more policy gradient

Twelve LENS-PG v2 configurations left greedy generalising accuracy at exactly 0.00% while damaging in-domain accuracy in
proportion to the learning rate. Fuel was measured to exist (4.67% of samples correct at T=4, 93.5% of prompts able to
succeed), so sparsity of reward - not support, capacity or estimator design - was what a per-sample advantage could not cope
with.

Rejection sampling attacks that directly: **draw many samples at T=4, keep the prompts that produced a correct answer, train
on those as ordinary supervised rows, repeat.** Rare success becomes ordinary supervision instead of a noisy advantage.

## Result: 0.00% -> 78-88% generalising, at a measurable cost

Four rounds, 60 generalising prompts x 16 draws per round, 100 supervised steps per round, 3 seeds:

```
seed   gen before   round1   round2   round3   round4     in-domain after
0        0.00%       40.75%   60.00%   77.50%   88.00%       2.00%
1        0.00%       39.25%   54.75%   66.25%   82.75%       2.00%
2        0.00%       38.25%   51.00%   70.50%   78.25%       3.00%
```

**Generalising accuracy climbs every round and reaches 78-88%.** Nothing else tried in this project has moved it off zero: not
twelve policy-gradient configurations, not nine earlier RL variants, not a KL anchor, not dense per-token reward.

**And it destroys in-domain accuracy: 70.5-72.5% -> 2-3%.** That is textbook catastrophic forgetting, and it is reported here
rather than buried, because a model that gains the new task by losing the old one is not a solution.

## The replay knob traces the whole trade-off

Mixing the in-domain rows into every supervised batch (`--replay`, the standard remedy for forgetting):

```
replay   generalising (3 seeds)          in-domain (3 seeds)
0.0      88.00 / 82.75 / 78.25           2.00 /  2.00 /  3.00
0.5      40.25 / 34.00 / 35.75          72.50 / 69.00 / 62.50
0.8      11.50 / 10.50 / 14.25          83.00 / 85.50 / 81.00
```

Three things follow from that table:
1. **The trade-off is smooth and controllable** - one number moves the model along it, with no retraining from scratch.
2. **At replay 0.8, in-domain accuracy ends ABOVE where it started** (83.0 / 85.5 / 81.0 against a 70.5-72.5% baseline), because
   the replay rows keep training the arithmetic the model already had. The new task is learned at a cost that is not a loss
   for the old one.
3. **There is no free lunch at either end**: maximum new-task ability costs the old task, maximum retention costs most of the
   new task. A middle setting (0.5) keeps 34-40% generalising while holding in-domain near baseline.

## What this means for the standing goals

The honest answer to "make our own RL method that actually learns" is now a measurement rather than an opinion: **for this
setting, verified self-training beats policy gradient, decisively.** Policy gradient moved nothing in twelve configurations;
rejection sampling moved +78 to +88 points in four rounds with the same model, the same task, the same warm start and less
machinery.

The predictor for which one will work is the one this document's predecessor measured: whether a reward is
**attainable-but-rare** (self-training works: keep the rare successes) or **empty** (nothing works) or **dense enough for a
usable gradient** (policy gradient can work, and did not here).

## What is not claimed

- This is `(a+b) mod 10`, not the language model. Nothing here is evidence about the 47M chat line.
- The generalising numbers come from 4 rounds of 60 prompts; a wider evaluation and more rounds are the obvious next checks,
  and the 88% figure is a 400-prompt greedy measurement, not a benchmark.
- Replay 0.8's in-domain gain over baseline is measured on the training rows the warm start used, which flatters it slightly;
  the held-out in-domain number is the one to quote next time.
