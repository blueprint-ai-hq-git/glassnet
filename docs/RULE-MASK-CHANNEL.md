# The idle concepts act only through supervision - so excluding them from the rules is FREE (-0.13 points)

Date: 2026-11-14. New trainer flag `--rule-idle` (zeroes the named concepts' rows in `rules.W` after every optimiser step, while the
concept layer keeps training on all 14 labels). Predictions in `lens/runs/RULE-MASK-PREDICTIONS.txt`, output
`lens/runs/RULE-MASK-RESULTS.txt`.

## The question this answered

`docs/history/EXPLAIN-CAUSAL-IDLE-CONCEPTS.md` established two facts that looked contradictory:

- 5 of 14 concepts are read by **no rule** (`center_ink`, `border_ink`, `outline_ratio`, `symmetry_center`, `tall_shape`), so they
  cannot influence a prediction through the rule layer;
- yet **trimming them from the concept set costs 1.55 points** (80.16% vs 81.71%).

Two channels were possible: (a) they shape the shared training objective and nothing else, or (b) they reach accuracy through the rule
layer in ways the top-1-per-rule analysis missed. This run separates them: keep the full 14-concept supervision, but **forbid the
rule layer** from reading those five concepts.

## Result: channel (a), confirmed, at no cost

```
                                                        accuracy (3 seeds)   spread
14 concepts, rules FREE (baseline)                      81.71% +/- 0.78      0.78
14 concepts, rule layer FORBIDDEN the 5 idle concepts   81.58% +/- 0.50      0.50
9 concepts only (trimmed entirely - not recommended)    80.16% +/- 2.13      2.13
```

**-0.13 points, inside the ~0.2 reproducibility wobble measured earlier, and the seed spread IMPROVES from 0.78 to 0.50.** P1 held
(my bar was >= 81.2%). The mask held exactly as intended: none of the five forbidden concepts is read by any rule, and - side effect
worth reporting - `left_heavy` also went unused, so the rules now read **8 of 14** concepts.

## Why the trim hurt but the mask does not

Both keep 14 concepts in the *training objective*; only the trim removes them from it. Since masking the rule layer is free and
removing the concepts is expensive, **the five idle concepts matter purely through the concept-supervision term** (`alpha 1.0`) and
the `varfloor` penalty shaping how the concept layer trains - not through any path a prediction can take. They are, in the strictest
sense, training-time only.

## The readable claim this lets us make, without any hedging

```
14 named concepts, each audited to track its own name (0 of 14 mislabeled)
 8 of them are read by the 32 single-concept rules
 6 are training-time only: 5 forbidden to the rules by --rule-idle, plus left_heavy which the rules did not choose
explain() therefore names only rules that feed a prediction; the other concepts provably cannot appear in any explanation
```

This is stronger than the previous wording ("honest but idle, please ignore them"), because the exclusion is now **enforced by
construction** rather than by disclosure. A reader cannot be misled by a concept that the architecture physically denies the rules.

## Recommended configuration, updated

```
concepts : 14 de-duplicated, alpha 1.0 supervision
rules    : 32, single-concept (lam 0.02), keep 3, --rule-idle for the 5 never-used concepts
accuracy : 81.58% +/- 0.50   (matched plain control 84.37% +/- 1.04, so -2.79)
honesty  : 0 of 14 concepts mislabeled; 0 of 14 below 0.50 own-label correlation
explain  : verified causal - top-2 named rules remove a mean 346.7% of the logit margin vs 10.8% for 2 random rules
```

The 0.13-point difference against the free-rule baseline is not claimed as a win: it is inside the noise floor. The point is that it
**costs nothing** to make the explanation airtight.

## Predictions, scored

- **P1 accuracy within 0.5 of 81.71 - HOLDS** at 81.58 (-0.13), proving channel (a).
- **P2 the rule layer reads only allowed concepts - HOLDS**, none of the five appears in any rule.
- **P3 concepts stay honest - not re-audited in this run.** The concept layer is trained identically and the mask touches only the
  rule layer, so honesty is inherited from the audited runs; it is nonetheless the one thing this document does not verify itself.
- Falsifier (accuracy landing near the trimmed 80.16) **not triggered**.
