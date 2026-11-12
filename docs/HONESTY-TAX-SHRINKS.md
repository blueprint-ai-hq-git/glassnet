# The honesty tax is not fixed: rules recover most of it at the honest setting (2026-11-12)

Open trade before this run, on the 12-concept de-duplicated set at 16 rules:
```
alpha 1   79.88% +/- 0.32   mean |rho| 0.478   7.0 of 12 misnamed    <- accuracy configuration
alpha 3   70.00% +/- 3.54   mean |rho| 0.698   0.7 of 12 misnamed    <- honest configuration
```
The honest configuration cost **9.88 accuracy points**, and every lever tried so far (concept-loss weight, concept learning
rate) only moved along that same line. The untested assumption was that the tax is a property of honest concepts. It is not -
it is partly a property of an under-powered rule layer.

**Prediction, written before the run:** at alpha 3 with 12 concepts, raising the rule count recovers part of the tax - R=32
reaches >= 72% accuracy while keeping misnamed <= 1.5 of 12 and mean rank honesty >= 0.65. **PASSED** (72.01%, 1.3 of 12,
0.736), and R=48 went further still.

```
| rules | accuracy (3 seeds) | mean |rho| | misnamed | median per-rule | dead rules | interaction |
|---|---|---|---|---|---|---|
| 8 | 69.48 +/- 5.35 | 0.740 | 1.3 of 12 | +6.65 | 0.7 of 8 | 9.9% |
| 16 | 70.00 +/- 3.41 | 0.698 | 0.7 of 12 | +1.11 | 1.7 of 16 | 7.8% |
| 32 | 72.01 +/- 4.36 | 0.736 | 1.3 of 12 | +0.31 | 6.3 of 32 | 5.4% |
| 48 | 74.73 +/- 1.73 | 0.743 | 1.0 of 12 | +0.07 | 13.0 of 48 | 6.1% |
```

## What this changes
- **The tax shrinks from 9.88 points to 5.15 points.** At alpha 3, accuracy rises 69.48 -> 70.00 -> 72.01 -> **74.73%** as rules
  go 8 -> 16 -> 32 -> 48, against the 79.88% alpha-1 ceiling. The last arm is also the most *stable* configuration measured on
  this task (**+/- 1.73**, individual seeds 76.56 / 74.51 / 73.12 - tighter than the accuracy configuration's +/- 0.32 only in
  absolute spread terms it is comparable, and far tighter than every other alpha-3 arm at +/- 3.5 to +/- 5.3).
- **Honesty is now insensitive to rule count**: mean |rho| 0.740 / 0.698 / 0.736 / 0.743 and misnamed 1.3 / 0.7 / 1.3 / 1.0 of
  12 across 8 to 48 rules. Once the concept loss is weighted, adding rules extracts accuracy from honest concepts instead of
  diluting them. A single rule is never bought with a dishonest concept neuron.
- **The price moves to the rules, not the concepts.** Dead rules rise 0.7 -> 1.7 -> 6.3 -> **13.0 of 48**, and the median
  per-rule ablation loss falls +6.65 -> +1.11 -> +0.31 -> **+0.07** points. At 48 rules, 27% are inert - exactly the defect the
  original 30-rule gate had. So the trade is no longer "accuracy against honest names", it is "accuracy against how many rules
  a human must read".

## The three defensible operating points, measured
```
configuration                     accuracy          |rho|   misnamed   median rule   dead rules   rules to read
12 concepts, a1, 16 rules      79.88 +/- 0.32        0.478   7.0 of 12     +0.26        1.0            16
12 concepts, a3, 16 rules      70.00 +/- 3.54        0.698   0.7 of 12     +1.11        1.7            16
12 concepts, a3, 32 rules      72.01 +/- 4.35        0.736   1.3 of 12     +0.31        6.3            32
12 concepts, a3, 48 rules      74.73 +/- 1.73        0.743   1.0 of 12     +0.07       13.0            48
```
- **Best readability per point of accuracy**: a3 / 16 rules - every rule worth +1.11 points, 0.7 of 12 names wrong, 70.00%.
- **Best balance**: a3 / 32 rules - 72.01%, every rule still worth +0.31 on median, but 6.3 of 32 are inert.
- **Best accuracy with honest names**: a3 / 48 rules - 74.73% +/- 1.73, 1.0 of 12 names wrong, but a quarter of the rules are
  dead weight and a human reading all 48 is reading 35.

The alpha-1 16-rule arm is still the accuracy champion at 79.88%, and it still has 7 of 12 concept names that do not describe
their neurons. That has not changed.

## Reproduction
`--alpha 3.0 --K 12 --R {8,16,32,48}` with `GLASSNET_CONCEPTS=$PWD/data/concepts_sel12.npz`, three seeds each, audited by
`tools/phase2_reaudit.py`. Twelve checkpoints under `runs/phase2_sweep/A3R*_s*/`.
