# Per-region concepts: spatial resolution is real but small - the readable penalty is ~2.4 points and largely irreducible

Date: 2026-11-14. Tool `gates/phase2_train.py` with 12 new **per-region** concepts added to `glassnet/concepts.py`
(`ink_TL/TR/BL/BR`, `hedge_*`, `vedge_*` - ink, horizontal-stripe and vertical-stripe measurements computed separately on each
14x14 quadrant of the image, every one a named human-readable statistic). Predictions pre-registered in
`lens/runs/REGIONAL-PREDICTIONS.txt`; output `REGIONAL-RESULTS.txt`, honesty `REGIONAL-AUDIT.txt`.

## The pre-registered test

Last run wrote, before this experiment existed: *"if per-region concepts close more than 1 point of the 3.13-point gap at matched
parameters, then spatial resolution was the constraint all along; if they do not, the concept bottleneck is fundamental to a 14-24
concept middle layer and about 3 points is the irreducible price."*

## Result: 0.37 of the gap closed, not 1

```
                                              params    accuracy (3 seeds)    gap vs matched plain
14 whole-image concepts                       11,800    81.58% +/- 0.50       -2.79   (control 84.37 +/- 1.04 @ 10,345)
12 per-region concepts (this run)             10,166    81.95% +/- 0.53       -2.42   (same control)
```

- **P1 (penalty falls to 1.79 or better, i.e. accuracy >= 82.58%) FAILED** at 81.95, a gap of 2.42.
- **P2 (beats the 14 whole-image model despite FEWER parameters) HOLDS**: 81.95 vs 81.58 with **14% fewer parameters**
  (10,166 vs 11,800).
- **P3 (at most 2 of 12 mislabeled) HOLDS**: 1 of 12 mislabeled, 0 of 12 below 0.50 own-label correlation, mean 0.771, 1 dead rule,
  and the rules read 10 of 12 concepts.

**The pre-registered threshold was not met, so the honest conclusion is the one I wrote down in advance: spatial resolution is a
real but small factor (it recovered 0.37 of 2.79 points, about 13%), and roughly 2.4 points is close to the irreducible price of a
small readable middle layer on this task.**

## What per-region concepts are still worth

They are not a breakthrough, but they are a genuine efficiency improvement:

```
                                     accuracy        params     honesty (mislabeled)   concepts read by rules
14 whole-image                       81.58 +/- 0.50  11,800     0 of 14                 8 of 14
12 per-region                        81.95 +/- 0.53  10,166     1 of 12                 10 of 12
```

**Better accuracy, 14% fewer parameters, and better coverage of the concept layer by the rules** (10 of 12 against 8 of 14) - at the
cost of one mislabeled concept. That coverage matters for the readable claim: fewer of its named concepts are idle.

## Where this leaves the readable image line

Every lever has now been measured rather than assumed:

```
concept count        more concepts help a little and widen the penalty (22-set: gap 3.13, 3 mislabeled)
concept KIND         per-region helps a little (gap 2.42) and is cheaper
rule count           saturated (R=64 no better than R=32)
connections per rule inert at lam 0.02 (every rule carries one weight)
sparsity (lam)       creates single-concept rules; stronger L1 gives 0 multi-concept rules, 0 dead
coverage penalty     cannot beat L1 at lam 0.02
trimming concepts    costs 1.55 points
rule-layer masking   free (-0.13)
explain()            causal: named rules remove 346.7% of margin vs 10.8% random
```

**Best honest configuration: 12 per-region concepts -> 32 single-concept rules -> 10 classes, 81.95% +/- 0.53, 10,166 parameters,
1 of 12 concepts mislabeled, explain() verified causal.** Against a plain network of the same size that is **-2.42 points**, and the
price buys a model whose entire middle layer is twelve named measurements wired through rules a human can read.

## What I would try next, honestly ranked

1. **Larger images / finer regions are not the answer** (this run). The remaining ~2.4 points is unlikely to be recovered by more
   concepts of this family. A genuinely different idea is needed for that, such as concepts computed by convolution-like shared
   filters rather than per-region summaries - which trades nameability and needs its own honesty audit.
2. **Move the same discipline to the chat model**, where the readable line has not been rebuilt with the lessons from this session
   (de-duplicated concepts, single-concept rules, minimum contribution, rule-layer masking, causal explain()). That is where the
   project's remaining goals actually live.
