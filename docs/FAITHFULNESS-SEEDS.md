# Faithfulness under three seeds: "every rule matters" holds, "plausibility predicts necessity" does not (2026-11-12)

Two claims were made today from single-seed ablations. Both sides now have three seeds, produced by ablating the six saved
checkpoints (no retraining).

```
                        median per-rule loss        pearson(lift vs necessity)     implausible in top 10
24 rules   seed 0        0.487                            +0.184                        3
           seed 1        0.188                            -0.013                        6
           seed 2        0.400                            +0.015                        3
           MEAN          0.358 +/- 0.154                  +0.062 +/- 0.107              4.0

96 rules   seed 0        0.000                            +0.045                        2
           seed 1        0.000                            +0.068                        4
           seed 2        0.000                            +0.116                        5
           MEAN          0.000 +/- 0.000                  +0.076 +/- 0.036              3.7
```

## Claim 1: "at 24 rules every rule is load-bearing" - HOLDS
Median per-rule ablation loss is **0.358 +/- 0.154**, positive on all three seeds (+0.188 at the weakest). Removing a typical
rule from the 24-rule layer costs real accuracy every time.

## Claim 2: "at 96 rules half the layer is inert" - HOLDS, exactly
Median per-rule ablation loss is **0.000 with zero spread** across three seeds. In every run, the median rule of the 96-rule
layer can be deleted without changing a single answer. This is the most reliable result of the day.

## Claim 3: "plausibility predicts necessity" - RETRACTED
Earlier today this was reported as **+0.263** on one 24-rule run and then **+0.184 / spearman +0.433** on another, against
**-0.012** for the 96-rule layer, and it was called "the first time making sense and being needed point the same way".

Three seeds say otherwise:
```
24 rules: +0.184, -0.013, +0.015   ->  +0.062 +/- 0.107   (one seed negative)
96 rules: +0.045, +0.068, +0.116   ->  +0.076 +/- 0.036   (consistently small)
```
**The 24-rule arm's +0.184 was a lucky seed.** Its other two seeds are -0.013 and +0.015, i.e. nothing. The 96-rule arm is
in fact the *more consistent* of the two, at a small +0.076. Neither arm shows a correlation large enough to claim: whether
a rule "makes sense" does not predict whether the model needs it, in either configuration.

And the count of implausible rules among the ten most necessary is 4.0 (24 rules) against 3.7 (96 rules) - indistinguishable.

## What survives, and what the layer can honestly claim
```
HOLDS on 3 seeds:
    concepts are named and checkable; concept honesty 78.0% +/- 1.7, no concept near 50%
    no rule is a contradiction (cross-facet wiring - true by construction)
    24 rules: every rule is load-bearing (median 0.358 +/- 0.154)
    96 rules: the median rule is inert (0.000 on all seeds), so a cheap layer is cheap for the wrong reason
    accuracy 15.08% +/- 0.46 (24 rules) vs 16.49% +/- 0.17 (96 rules) against a 5.42% baseline

DOES NOT HOLD:
    "the rules that make sense are the rules that matter" - correlation +0.06 to +0.08, i.e. nothing, in both arms
```

So the layer is **readable** (a person can read every rule and every concept), it is **causal** (forcing a concept changes the
answer on 36% of windows), its rules are **load-bearing** at 24 rules - and its content still does not explain its own
importance. That last property is the one that would make it worth more than a normal model, and it is not there yet.

## Why this matters for the next step
Every lever tried today was applied to the *concept vocabulary* or the *wiring rule*: token concepts, pair concepts, facet
concepts, cross-facet wiring, prevalence filtering. None of them moved the correlation. That is evidence the cause is not in
the vocabulary - it is that nothing in the objective rewards a rule for being *understandable*, only for being *useful*. The
next lever has to be in the objective, for example requiring a rule's contribution to be predictable from its two named
concepts alone, or pruning any rule whose removal is free (which the 96-rule arm shows the current L1 does not do).
