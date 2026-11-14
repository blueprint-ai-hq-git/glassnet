# Where the readable rule layer's cost actually comes from: not from the number of rules, and not from sparsity

Date: 2026-11-14. Runner `gates/phase2_train.py`, K=14 named concepts, alpha 3.0, lam 0.005, 10 epochs, batch 128, lr 0.1,
3 seeds. Predictions written before each sweep: `lens/runs/RULE-WIDTH-PREDICTIONS.txt`, `lens/runs/KEEP-SWEEP-PREDICTIONS.txt`.
Reference points from `docs/history/PHASE2-ABLATION-CLOSED.md`: **A3 no rule layer 83.13% +/- 0.42**, matched plain control
**84.37% +/- 1.04**.

## A landmine found before the sweep could run

`gates/phase2_train.py --K 12` (the value in the published VF0.10 `config.json`) **cannot work**: the model builds `--K` concept
neurons, but `glassnet.concepts.load()` always returns all **14** named concepts, so the joint loss dies with

```
ValueError: operands could not be broadcast together with shapes (128,14) (128,12)
```

The trainer now **refuses** that configuration by name instead of failing deep in the loss:

```
REFUSING TO RUN: --K 12 does not match the 14 concept labels that glassnet.concepts.load() returns.
```

That also means the published VF0.10 arm (K=12) was not produced by this trainer as it stands - the label set must have been
different when it ran. Another reason its 81.73 is a historical number rather than a reproducible one.

## Sweep 1: number of rules (R), connections fixed at 3, K=14

```
R=16   76.41% +/- 6.35
R=32   78.57% +/- 3.87
R=64   79.42% +/- 1.53
```

- **P1 (R=64 beats R=32 by >= +1.0) FAILED** at **+0.85**. Rule count is not the binding constraint.
- **P2 (R=16 worse than R=32 by >= 1.0) HOLDS** at **-2.16**.
- **P3 (R=64 still below A3's 83.13) HOLDS** at 79.42.

The falsifier fired, and it pointed at the connection limit. But the sweep found something I did not predict: **rule count barely
moves the mean (+2.85 from 16 to 64) while collapsing the spread 4x, from +/-6.35 to +/-1.53.** Seed 2 alone goes 67.64 ->
73.49 -> 77.62. More rules buy **reliability**, not average accuracy - a different and more useful currency than the mean.

## Sweep 2: connections per rule (--keep), R=32, K=14

```
keep=1    65.44% +/- 3.06    (1.00 conns/rule)
keep=3    77.76% +/- 4.20    (3.00 conns/rule)   <- the shipped setting
keep=6    78.60% +/- 4.34    (6.00 conns/rule)
keep=12   78.45% +/- 4.55   (11.97 conns/rule)   <- essentially no sparsity at all
```

- **P1 (monotonic rise) PARTLY FAILED**: 1 -> 3 -> 6 rises (+12.32 then +0.84) but 6 -> 12 does not (-0.15, inside noise).
- **P2 (keep=12 within 1.5 points of A3) FAILED** at **4.68 points below** A3.
- **P3 (readability stays countable) HOLDS**: every arm reports its real average connections per rule, 1.00 through 11.97.

**The written falsifier triggered: no amount of connectivity makes the rule layer cheap.** At ~12 connections per rule - which is
no sparsity at all, every rule reading every concept - the readable model is still **4.68 points** below removing the rule layer
entirely. So the ~4 points of `explain()` cost is **structural**, not a sparsity artefact.

## Two things this changes

1. **The 3-connection limit is not the villain.** Going 3 -> 6 -> 12 connections gains +0.84 then -0.15, i.e. nothing. The limit
   that matters is between 1 and 3 (+12.32): a rule needs about three inputs to be worth anything, and beyond that the extra
   connections are decoration. **The shipped keep=3 is already at the sweet spot**, which is a genuinely good outcome for a
   readable architecture - we did not pay accuracy for the sparsity we advertise.
2. **The rule layer itself is the price.** Since connectivity and rule count both fail to recover it, the cost lives in having a
   small linear bottleneck between concepts and classes at all.

## The confound I am not going to paper over

The earlier ablation's A2 arm ("no sparsity": no L1, no prune) scored **81.71%**, while this sweep's `keep=12` arm - also
effectively unpruned - scored **78.45%**. Those differ in TWO ways: A2 used **alpha 1.0 and lam 0.0**, this sweep used **alpha 3.0
and lam 0.005**. So **the L1 penalty and the concept-supervision weight are not separated yet**, and it is entirely possible that
the expensive ingredient is the `lam` L1 pressure rather than the rule layer's existence. That is the next experiment, and it is
written to isolate `lam` at fixed alpha and fixed pruning:

```
lam 0.0 vs lam 0.005 at alpha 3.0, R=32, keep=3, K=14, 3 seeds
```

Until that runs, this document's claim is the narrow one it can support: **at alpha 3.0 / lam 0.005, neither the number of rules
nor the number of connections per rule recovers the readable model's loss to removing the rule layer.**
