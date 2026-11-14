# The rules are not really 3-concept rules: effective connectivity is 1, and the "implausible rules" finding was measuring noise

Date: 2026-11-14. Checkpoint audited: `runs/phase2_sweep/CLEAN14_A1p0/final_seed0.npz` (the recommended alpha 1.0 arm, 14
de-duplicated concepts, 32 rules). Tools written for this: `tools/rule_polarity_audit.py`,
`tools/rule_effective_connectivity.py`, `tools/rule_implausibility_cause.py`. Predictions in
`lens/runs/RULE-PLAUSIBILITY-PREDICTIONS.txt`. Raw output: `lens/runs/RULE-POLARITY-AUDIT.txt`,
`RULE-EFFECTIVE-CONNECTIVITY.txt`, `RULE-IMPLAUSIBILITY-CAUSE.txt`.

## What was claimed before this run

Three documents carried the line **"19-20 of 32 rules wire together concepts that co-occur less than chance (median rule-pair
lift 0.79-0.92), so the rules are hard for a human to justify"**, produced by section 5 of `tools/phase2_reaudit.py`.

## Finding 1: the instrument ignored polarity, and that mattered

The audit binarises both concepts and computes `P(A and B) / (P(A) P(B))` with no reference to the sign of the rule's weights. A
rule weighted `(+A, -B)` means "A high AND B low", so `P(A and B)` below chance is **expected** for it.

```
rules: 32   same-signed top-2 weights: 10   opposite-signed: 22
                          median   rules below 1.0
as the audit measures it   0.79     19 of 32
with the rule's polarity   0.96     18 of 32
```

22 of 32 rules are opposite-signed (P1 held), and correcting polarity flips several rules from "implausible" to plausible - rule
24 goes **0.53 -> 1.47**, rule 26 **0.59 -> 1.41**, rule 29 **0.68 -> 1.32**, rule 2 **0.68 -> 1.32**. But the count of
below-chance rules barely moves (19 -> 18), so polarity alone does **not** explain the finding (P2 failed).

## Finding 2: the real problem - most rules have only one concept that matters

```
nonzero (kept) weights per rule      min 3   median 3   max 3       <- the advertised "2-3 concepts"
effective concepts at |w| > 0.10     median 2   rules with <=1: 11 of 32
effective concepts at |w| > 0.25     median 2   rules with <=1: 14 of 32
effective concepts at |w| > 0.50     median 1   rules with <=1: 27 of 32
share of a rule's total |weight| in its top concept: median 0.71, 11 of 32 above 0.90
```

Every rule keeps exactly 3 non-zero weights, so the *span* is honest. But the weights are not: real examples from the checkpoint,

```
rule 23   outline_ratio +1.14    light_region_count -0.01    diag_edges  +0.01
rule 14   lowres_symmetry +1.09  row_profile_std    -0.01    left_heavy  +0.01
rule 22   outline_ratio -1.07    border_ink          -0.00    top_heavy   -0.00
```

Three of those are **single-concept rules wearing two near-zero weights**. `explain()` would faithfully print three concepts while
two of them contribute nothing a human could act on.

## Finding 3: the "implausible rules" were mostly this artefact

```
|second weight|      n     naive lift median   polarity lift median   below 1.0
< 0.10              11          0.72                 0.89              6 of 11
0.10 - 0.40          8          0.71                 0.85              5 of 8
> 0.40              13          1.25                 0.97              7 of 13
```

**The 13 rules that genuinely read two concepts have a naive lift median of 1.25 - above chance, i.e. plausible.** The below-chance
figures come overwhelmingly from the rules whose second weight is noise, where a co-occurrence test has nothing to measure. The
correlation between |second weight| and naive lift is **+0.494**, and 6 of the 18 "implausible" rules have a second weight below
0.10 outright.

## What this changes

1. **The claim "19 of 32 rules pair non-co-occurring concepts" is withdrawn** as a description of rule quality. It described a
   measurement artefact over nominally-multi-concept rules.
2. **The claim "each rule connects to 2-3 concepts" is true only in span, not in effect.** At |w| > 0.50 the median rule reads
   **one** concept and **27 of 32 rules read one or fewer**. This is the honest statement of the architecture's effective
   structure, and it is the number `explain()` and any public claim should use.
3. **What is actually good here**: 13 of 32 rules really do combine two concepts with both weights above 0.40, and those 13 show
   above-chance co-occurrence (median lift 1.25). The architecture can produce genuine multi-concept rules - it just does not force
   them.

## The fix this implies (next run)

Enforce a **minimum contribution** for a connection to count, instead of only a nonzero weight: during pruning, drop connections
whose |weight| falls below a stated floor and re-prune until every rule either has two weights above that floor or is marked
single-concept. Then report two honest rule counts - multi-concept rules and single-concept rules - rather than one number that
implies all 32 behave the same way.

## Prediction scorecard (against my own file)

- **P1 opposite-signed >= 15 - HOLDS** (22).
- **P2 corrected median > 1.00 and implausible < 10 - FAILED** (0.96 and 18). Polarity was necessary but not sufficient.
- **P3 same-signed rules mostly above 1.0 - HOLDS** (7 of 10).
- Falsifier ("fewer than 15 opposite-signed AND same-signed also below chance") was **not** triggered, so the instrument flaw is
  real - but Finding 2 shows the bigger problem was elsewhere.
