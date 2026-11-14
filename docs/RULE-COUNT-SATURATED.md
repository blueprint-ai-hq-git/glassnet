# Rule count is saturated: 64 rules do not beat 32, so the remaining 2.79-point gap is the concept bottleneck

Date: 2026-11-14. Tool `gates/phase2_train.py` at the final readable configuration (14 de-duplicated concepts, alpha 1.0, lam 0.02,
keep 3, `--rule-idle` for the 5 never-used concepts), changing ONLY R. Predictions in `lens/runs/R64-PREDICTIONS.txt`, output
`lens/runs/R64-RESULTS.txt`. Also closes the honesty gap left open last run (`lens/runs/RULEMASK-AUDIT.txt`).

## First: the honesty gap from last run is closed

Last document stated explicitly that concept honesty at the final configuration was inherited and not re-verified. Now audited:

```
seed 0   mean |rank corr| with own label 0.768   1 of 14 below 0.50   0 of 14 mislabeled   0 dead rules
seed 1   mean |rank corr| with own label 0.773   0 of 14 below 0.50   0 of 14 mislabeled   1 dead rule  of 32
```

**0 of 14 concepts mislabeled at the shipped configuration, in both seeds audited.** The claim now rests on measurement at that
exact setting.

## Second: doubling the rule count does nothing

```
R=32   81.58% +/- 0.50   32 single-concept rules   8 of 14 concepts read
R=64   81.27% +/- 0.65   60-64 single-concept rules, 0-4 dead   9 of 14 concepts read
```

- **P1 (R=64 beats R=32 by >= +0.5) FAILED** at **-0.31**.
- **P2 (rules stay single-concept) HOLDS** - 0 multi-concept rules in all three seeds; dead rules 3/0/4 of 64.
- **P3 (more distinct concepts used) HOLDS narrowly** - 9 of 14 against 8.
- **FALSIFIER TRIGGERED** (-0.31 is within my +/- 0.3 saturation band, and the sign is negative): **rule count is saturated. The
  remaining gap is not rule capacity.**

## Third: `keep` is now a dead knob, and that is a fact about the architecture, not a bug

The "each rule connects to at most 3 concepts" setting has nothing left to limit. At lam 0.02 every rule carries exactly **one**
weight above 0.50, so `--keep 3`, `--keep 6` and `--keep 12` all describe a structure the training already produces on its own. The
3-connection span limit was a real constraint when rules were multi-concept (start of this session); it is now inert. Anyone reading
the code should be told, or they will spend a run tuning it - as I did for the keep sweep before the L1 result landed.

## Where the 2.79-point gap actually lives

The two components have now been measured separately:

```
concept layer only (concepts -> classes, no rule layer)  83.13% +/- 0.42     -1.24 vs matched plain
full readable stack (concepts -> 32 rules -> classes)    81.58% +/- 0.50     -2.79 vs matched plain
```

So of the 2.79 points: **about 1.24 is the 14-statistic concept bottleneck** and **about 1.55 is the rule layer**. Rule count cannot
buy back its share (this run), and the rule layer's own cost was already shown to be structural rather than a sparsity artefact.
**The largest remaining lever is therefore the concept layer: 14 hand-computed pixel statistics cannot represent Fashion-MNIST as
well as 784 raw pixels can.** That is the honest price of a middle layer a human can read, and it is now quantified rather than
assumed.

## What the next run should try, in order of expected value

1. **More informative concepts, not more rules.** The library defines 36 features; the decorrelated selection used 14. A larger
   decorrelated set (say 20-24, still |corr| <= 0.60) would test whether the bottleneck is the COUNT of statistics or their KIND.
   Prediction to write before it runs: if accuracy rises above 82.3%, the concept count was the limit; if it stays at ~81.6, 14-24
   hand-computed statistics are simply insufficient and the next idea must be higher-resolution concepts (features computed per
   quadrant or per patch rather than per image).
2. **Higher-resolution concepts.** A `quad_UL_ink`-style statistic computed on a 4x4 grid of image regions would give 16x more
   spatial detail while keeping each concept nameable ("ink in the upper-left corner of the lower-right quadrant") - the cost is that
   the number of concepts grows, which the previous item tests first.

## Honest summary of where the readable image model stands

```
matched plain MLP (10,345 params)          84.37% +/- 1.04
readable, 14 honest concepts + 32 rules    81.58% +/- 0.50   (-2.79)
  of which: concept bottleneck             -1.24
            rule layer                     -1.55
concept honesty                            0 of 14 mislabeled (2 seeds audited at this setting)
explain()                                  causal: named rules remove 346.7% of margin vs 10.8% random
what it buys                               14 named concepts, single-concept rules, purely additive
                                           explanations, idle concepts physically barred from the rules
what it costs                              2.79 accuracy points, of which 1.24 is the concept bottleneck
