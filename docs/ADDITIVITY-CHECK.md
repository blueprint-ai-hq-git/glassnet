# Additivity check: concept-level explanation is complete in EVERY configuration (2026-11-12)

Last wake concluded that the architecture "trades one kind of faithfulness for another" - rule necessity under no tie, or
provable absence of hidden freedom under a tie - and that neither configuration is fully faithful. That conclusion was
incomplete, and this check corrects it.

`tools/explain_measured.py` decomposes a decision into per-concept effects and compares their sum against switching every
concept off at once. If the two agree, a per-concept explanation is quantitatively complete. Measured on 2,000 held-out
windows per model:

```
model                        sample acc   sum of individual effects   all concepts off   interaction
TIED-23, distinct pairs        15.00%            -4.661                  -4.592          -0.069  (1.5%)
TIED-23, repeated pairs        12.40%            -3.880                  -3.897          +0.017  (0.4%)
UNTIED-11                      15.05%            -2.674                  -2.732          +0.058  (2.1%)
```

**Prediction on record: "additivity holds better under TIED than untied". FALSIFIED - all three are additive within 2.1%.**
Concept interaction is negligible in every configuration tested, tied or not.

## Why this matters: the untied model gets BOTH properties
```
config        accuracy          rule necessity          concept explanation   no hidden freedom
UNTIED-11     15.08 +/- 0.46    YES (median 0.36)       additive within 2.1%  no
TIED-23       15.44 +/- 0.22    no (median 0.000)       additive within 1.5%  YES (rank 22 of 23)
```

The untied 24-rule model **has every rule load-bearing AND an additive, quantitative concept-level explanation**. What it
lacks is the rank proof that the answer layer holds no hidden degrees of freedom - its rules own arbitrary push directions,
but those directions are *measurable* by intervention, and their effects add up.

So the honest summary is not "one property or the other". It is:

- **Untied** gives a model where every named part is necessary and where a measured per-concept decomposition accounts for
  97.9% of the decision's effect. The rule's push direction is learned rather than derived, so it cannot be predicted from
  the concept names alone - but it can be measured, and it is reported as measured.
- **Tied** additionally proves the push direction is a *function* of the concepts (rank 22 of 23), at the price of rules
  becoming redundant, and at the cost of a more expensive layer (3.46 vs 1.57 accuracy points).

## The distinct-pairs arm, complete
```
tied + distinct rule pairs   16.05 / 14.30 / 14.78   mean 15.04 +/- 0.90 +/- 0.90
tied + repeated rule pairs   15.40 / 15.25 / 15.68   mean 15.44 +/- 0.22
```
The means are inside each other's spread and the distinct arm is less stable (spread 0.90 vs 0.22), so no accuracy
claim is made for de-duplicating pairs - the reason to do it was to restore rule necessity, and that failed (median per-rule
ablation loss stayed 0.000).

## Caveats
- Additivity is measured on one model per configuration; the three measurements agree, but per-seed replication would be
  needed before quoting 2.1% as a hard bound.
- Attribution measures each concept's total effect on the predicted word's score, not a unique credit split between concepts
  that push the same word.
- All numbers are the small rule-layer testbed (11 or 23 concepts, 24 rules), not the 15.5M flagship.
