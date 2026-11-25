# 2026-11-25 - three ideas tested, all falsified: what actually teaches a mechanism is COVERAGE

Sources: lens/runs/META-VS-MEMORISE.out, lens/runs/TRACE-CARRY.out (both exit 0).

## The three failed ideas, measured

    1. hand the mechanism over as INPUT (carry flag token)
       no-carry 100.0% | units-carry 100.0% | UNSEEN tens-carry 0.0%     gain over plain: +0.0
    2. hand over a FEW examples of the mechanism (2% of training = 80 pairs)
       no-carry 100.0% | UNSEEN tens-carry 5.0%                          gain: +5.0
    3. give the mechanism its own OUTPUT slot (write the carry digit first)
       trained classes 98.8% / 99.6% | UNSEEN tens-carry 0.0%             gain: +0.0
    baseline (direct answer)
       no-carry 98.8% | units-carry 99.4% | UNSEEN tens-carry 0.0%

Every pre-registered falsifier fired. Telling the model about the mechanism, showing it a handful of examples, and giving it a
place to write it down all failed to transfer the carry to a column position it had never seen carrying.

## What this does NOT mean

It does not mean these models cannot do arithmetic. The shipped maths model reaches 100% on held-out 2-digit addition with
9,084 parameters (MODELCARD-MATH.md) - and it was trained on a MIXED set that contained carries. The difference is not
cleverness, it is coverage: the mechanism was present in the training data.

So the finding is precise and narrow: **a model learns a mechanism only from examples that exhibit it. It cannot infer a rule
from the absence of a rule.** No hint, trace, or slot substitutes for seeing the thing itself in the position where it occurs.

## The design consequence, which matches the strongest lever already measured

Today's levers, ranked by measured effect on held-out loss:

    distinct data coverage    3.445   (20k -> 200k pool, controlled; replicated 3/3 seeds)
    matched-epoch data scale  0.485   (1M vs 200k pool at 2 epochs each)
    step budget               0.042
    model width              -0.017   (3.5x params, inside GPU noise)
    meta-hints / traces      +0.0     on unseen mechanism positions

Everything points the same way: the work is to GENERATE COVERAGE - every position, every operation shape, every combination -
with the answer verified by execution rather than guessed from text. That is the "learn fast by understanding" path that is
actually available to us, and it is a data-generation problem, not a hint problem.

## Next experiment

How much coverage does a mechanism need? Train on no-carry plus units-carry, plus N tens-carry examples for
N in {0, 25, 100, 400, 1600}, and find where the held-out tens-carry accuracy crosses over. That turns "coverage matters" into
a per-mechanism price in examples, which is what a data budget needs.
