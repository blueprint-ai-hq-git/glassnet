# "Duplicate" rules on the same concepts are load-bearing thresholds: a claim retracted (2026-11-12)

## The claim that was published and is now wrong
`docs/EXPLAIN-IMAGE.md` reported, from `explain()` output, that the 29 kept rules contain only 20 distinct 3-concept wirings and
concluded: *"A human must read 20 rules, not 29 ... the reported rule count overstates the reading work by ~31%."*

That followed from calling them duplicates. They are not. Grouped by wiring, the rules have **different input weights and
different biases**:

```
concepts (0, 2, 10)  rules 1, 13, 18, 21   biases  +0.2925 / -0.0201 / -1.1766 / -0.6214
concepts (4, 6, 7)   rules 4, 17, 27, 31   biases  -0.4389 / -0.3121 / +0.0185 / +0.5425
concepts (7, 8, 9)   rules 5, 6            biases  +0.8987 / -0.8875
concepts (2, 9, 10)  rules 8, 26           biases  +0.0333 / -0.2312
concepts (2, 4, 9)   rules 11, 12          biases  +0.0101 / -0.6791
```
Four rules on the same three concepts with biases from -1.18 to +0.29 are four THRESHOLDS - a piecewise response in those
concepts - not one rule written four times. The right test is therefore not "are they the same?" but "what does it cost to
collapse them?", which is what `tools/consolidate_rules.py` now measures: keep the strongest member of each wiring group, zero
the others' answer rows, re-measure accuracy per group.

## The measurement
Three seeds of the best honest arm (`a3 / 32 rules / pruned`):

```
seed 0: 76.60% -> 29.84%   (-46.76 points)   9 of 29 rules removed, 20 left to read
seed 1: 73.48% -> 47.18%   (-26.30 points)   5 of 23 rules removed
seed 2: 78.21% -> 41.26%   (-36.95 points)   6 of 23 rules removed
```

Individual groups, seed 0:
```
concepts (2, 4, 9)    rules [11, 12]        keep #12  -> -1.86 points
concepts (2, 9, 10)   rules [8, 26]         keep #26  -> -1.49 points
concepts (4, 6, 7)    rules [4, 17, 27, 31] keep #4   -> +0.78 points
concepts (7, 8, 9)    rules [5, 6]          keep #5   -> -12.22 points
```
One group of two rules - biases +0.90 and -0.89 on the same three concepts - carries **12.22 points**. Seed 2 has a group of
four rules on concepts (0,6,9) carrying **29.99 points**.

**Prediction written before the run: "collapsing shared wirings costs under 1.0 point on average". Wrong by 26 to 47 points.**

## What this means, and what it does not
- **The corrected claim: 29 rules is the honest reading count.** Nothing is overstated. A reader must read all 29, because the
  threshold variants change answers enormously.
- **It does not rescue the rule layer's other defect** - 15 of those rules never read a concept at all and are class priors
  (`docs/CLASS-PRIOR-RULES.md`), and pruning those did improve accuracy. Reading nothing and reading-the-same-concepts-with-a-
  different-threshold are opposite failures: one is free to remove, the other is expensive.
- **The useful product fact**: a rule group on the same concept triple is a piecewise function, so `explain()` should present it
  that way - "these 4 rules are one shape: when ink_amount and left_heavy and symmetry_ud are high, medium, low, very low" -
  rather than as four independent reasons. That is a presentation fix in `explain()`, not a model change, and it is the next
  thing to build.

## Method note
`tools/consolidate_rules.py` reports the cost group by group so a single dominant group is visible (seed 0's -46.76 aggregate
is mostly one 12.22-point group and one 29.99-point group on another seed). An aggregate number alone would have hidden that.
