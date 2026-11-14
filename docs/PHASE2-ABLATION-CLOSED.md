# Phase-2 gate item 5, closed: removing the rule layer is worth +4.09 points, and the concept names depend entirely on the supervisor

Date: 2026-11-14. Runner: `gates/phase2_ablations.py --seeds 0 1 2 --epochs 10`, output `gates/runs/phase2/ablations.md`.
Predictions written before the run in `lens/runs/PHASE2-ABLATION-PREDICTIONS.txt`. This was the one open gate item - the
Phase-2 document said it "needs a training run for each seed and was deliberately not launched" while a game was running.

## Results, 3 seeds, 10 epochs, same schedule everywhere

```
variant                        test acc (post-prune)     concept AUROC      conns/rule
A0 full (supervisor+L1+prune)      79.04% +/- 1.97        79.75 +/- 1.61        3.00
A1 no concept supervision          77.80% +/- 1.41        48.61 +/- 3.63        3.00
A2 no sparsity (no L1, no prune)   81.71% +/- 0.76        77.36 +/- 1.01       14.00
A3 no rule layer                   83.13% +/- 0.42        72.26 +/- 1.89          -
```

## What the middle layer costs, measured against the pieces

```
rules removed        A3 - A0 = +4.09 points     (83.13 vs 79.04)
sparsity removed     A2 - A0 = +2.67 points     (81.71 vs 79.04)
supervision removed  A1 - A0 = -1.24 points     (77.80 vs 79.04), AUROC 79.75 -> 48.61
pruning (this cfg)   A0 pre -> post: +0.02, -0.36, -1.02  = -0.45 mean: pruning HURTS slightly here
```

**The single biggest cost in the readable architecture is the rule layer: 4.09 points.** The sparse 3-connection constraints
cost a further 2.67. Together they are the price of `explain()` - and it is nearly 7 points, not the ~1 point the matched-gap
comparison suggested, because A0 here (alpha=1.0, lam=0.005) is a stricter configuration than the published VF0.10 arm
(alpha=3.0), which scored 81.73.

**And the concept supervision is doing exactly what it was built to do.** Removing it costs **31.14 AUROC points** of concept
honesty (79.75 -> 48.61): without the supervisor the neurons still fire, but they stop meaning their names. That is the check his
gate demanded, answered with a number - the names are taught, not decorative.

## Predictions, scored honestly (2 of 4 held)

- **P1 A3 beats A0 by at least +1.0 - HOLDS**, at **+4.09**. The rule layer is expensive, and more expensive than the matched-gap
  arithmetic suggested.
- **P2 A2 within 1.0 point of A0 - FAILED**, at **+2.67**. Sparsity is *not* free in accuracy; L1 plus pruning to 3
  connections costs 2.67 points. The "sparsity is free, it only buys readability" premise is **wrong as stated**.
- **P3 A1 loses concept honesty - HOLDS**, by a large margin (AUROC 79.75 -> 48.61, and accuracy also falls 1.24).
- **P4 pruning worth about +0.5 - FAILED, wrong sign**: in this configuration pruning costs **-0.45** points (it helped **+0.64**
  in the published VF0.10 pipeline). Pruning's value is configuration-dependent and cannot be assumed.

## The picture this completes

```
plain 10,345 params (matched control)                 84.37% +/- 1.04
readable, no rule layer (A3)                          83.13% +/- 0.42   -> -1.24 vs matched plain
readable, no sparsity (A2)                            81.71% +/- 0.76   -> -2.66
readable full (A0, strict config)                     79.04% +/- 1.97   -> -5.33
published readable (VF0.10, alpha 3.0)                81.73%
```

The readable architecture's price is now attributed piece by piece rather than quoted as one number: **about 1.2 points for the
concept layer itself, 2.7 more for forcing 3 connections per rule, and 4.1 more for having a rule layer at all in the strict
configuration.** If accuracy is the priority the rules are the expensive part; if `explain()` is the priority, 4.09 points is
its measured price at this size. Both are now known instead of assumed.
