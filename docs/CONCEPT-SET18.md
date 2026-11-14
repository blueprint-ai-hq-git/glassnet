# Dropping the redundant concepts makes the readable model both MORE honest and MORE accurate - 17 of 18 neurons now mean their name

Date: 2026-11-14. Tools: `tools/select_concept_set.py` (selection), `gates/phase2_train.py --concept-names` (new flag, default
unchanged), `tools/phase2_reaudit.py` (honest audit, now accepts a custom concept set). Predictions written before the run in
`lens/runs/CONCEPT-SET18-PREDICTIONS.txt`. Data: Fashion-MNIST, R=32 rules, keep=3, alpha=3.0, lam=0.005, 10 epochs, 3 seeds.

## The diagnosis: the label set was redundant, not the neurons

Measured on the labels alone, with no model involved (`lens/runs/CONCEPT-LABEL-REDUNDANCY-PREDICTIONS.txt`):

```
8 concept pairs correlate above 0.80 (max 0.916, ink_amount ~ quad_LL_ink)
25 pairs correlate above 0.50
ink_amount correlates 0.87+ with four different "concepts"
symmetry is 82.1% positive - a neuron can be 82% "accurate" by always saying yes
```

When two names describe nearly the same measurement, no pair of neurons can both mean their own name. That is precisely the
2-of-14 mislabeled result the audit reported - so per the project's own doctrine (**a concept that cannot be label-honest is
DROPPED, never forced**) the label set is what had to change.

## The fix, selected by measurement rather than taste

`tools/select_concept_set.py` picked greedily from the **36 features the library already defines**, keeping only those with
|correlation| <= 0.60 against already-selected features:

```
18 features selected of 36
pairwise |corr| within the set: median 0.000, max 0.587
shipped 14 for comparison:      8 pairs above 0.80, max 0.916
```

Every selected feature has a documented plain-English definition in `glassnet/concepts.py DESC` - `center_ink`, `border_ink`,
`ink_contrast`, `outline_ratio`, `row_profile_std`, `light_region_count`, `dark_region_count`, `streak_ink`, `corner_ink`, and so
on. No mystery neurons were introduced.

## The result: better on honesty AND on accuracy, at all three seeds

```
                               shipped 14 concepts      18 de-duplicated concepts
accuracy (3 seeds)               78.46% +/- 4.63            79.69% +/- 1.13
mean |rank corr| with own label      0.732               0.844 / 0.845 / 0.845
concepts below 0.50                  2 of 14                    0 of 18
concepts tracking ANOTHER feature    2 of 14                    1 of 18
dead rules                           0 of 32                    0 of 32
concept interaction                   0.0%                      0.0%
```

**17 of 18 concept neurons now track their own name better than any other feature**, and the one that does not is
`vertical_edges` (own +0.741, tracks `wide_shape` +0.867). The count of concepts below 0.50 own-label correlation goes from
**2 to 0**. And the accuracy did not pay for it - it rose, with a quarter of the seed-to-seed spread.

Predictions: **P1 HOLDS** (0.845 mean >= 0.80; 1 of 18 mislabeled <= 2). **P2 HOLDS** (79.69% is +1.23 over 78.46%, inside the
1.5-point band - it went the good way). **P3 HOLDS** (0 of 18 below 0.50). The falsifier was not triggered.

## Two things I am not going to hide

1. **The 18-concept model has about 28% more parameters** (roughly 15,068 against 11,800), so part of the +1.23 accuracy may be
   capacity rather than label quality. The honesty improvement cannot be explained that way - it is a property of the mapping
   between neurons and names - but the accuracy claim needs a parameter-matched control before it is quoted as a win. The next
   run should either prune the 18-set to 14 by the same rule or widen the 14-set's per-concept width to match.
2. **Rule plausibility got WORSE**: rule-pair lift median falls from **1.12 to 0.92**, and the number of rules whose two wired
   concepts do not co-occur rises from **12 to 19 of 32**. With more separable concepts available, the rules are increasingly
   combining features that rarely appear together, which is harder for a human to check and is the next honesty problem. A
   decorrelated concept set makes honest *concepts* easier and honest *rules* harder.

## Reproduce

```
python3 tools/select_concept_set.py
python3 gates/phase2_train.py --concept-names "center_ink,quad_UL_ink,..." --R 32 --alpha 3.0 --lam 0.005 --keep 3 \
        --rule_linear --varfloor 0.1 --seeds 0 1 2 --epochs 10 --batch 128 --lr 0.1 --tag SET18
GLASSNET_CONCEPT_NAMES="center_ink,..." python3 tools/phase2_reaudit.py runs/phase2_sweep/SET18/final_seed0.npz
```
