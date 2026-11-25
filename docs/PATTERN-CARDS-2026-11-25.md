# 2026-11-25 - PATTERN CARDS WORK: a unit with AUROC 1.000 for "units column carries", verified three ways

Source: lens/runs/PATTERN-CARDS.json, exp_pattern_cards.py, lens/patterns.py.
Model: 256 hidden units, trained 24,000 steps on stratified coverage (5,600 examples), held-out tens-carry accuracy **90.8%**.

## The cards

    256 units total | 28 given an honest name (11%) | 228 reported UNNAMED with their numbers

    unit  34 -> units_carry   AUROC 1.000   fires 54.8% of inputs
    unit 179 -> tens_carry    AUROC 0.997   fires 37.0% of inputs
    unit  35 -> tens_carry    AUROC 0.996   fires 52.4% of inputs

A unit whose activation ranks "units column carries" above every other input, perfectly, inside a model that solves the task.
That is the pattern idea, measured rather than asserted.

## The honesty test, which is the part that makes the names trustworthy

Three steps, and step (b) is the one that separates a concept from a lookup:

    concept                     unit    exact examples   FRESH examples   other concepts quiet   verdict
    units_carry                  34         100.0%           100.0%              96.0%            CONCEPT
    tens_carry                  179         100.0%           100.0%              92.0%            CONCEPT
    result_reaches_100          179         100.0%           100.0%              92.0%            CONCEPT
    a_tens_greater_than_b_tens   71          76.0%            76.0%             100.0%         NOT a concept
    no_carry_at_all               -        NO_UNIT - no unit reached the naming bar
    units_sum_is_exactly_10       -        NO_UNIT - no unit reached the naming bar

The last row matters as much as the first: a name that fails gets reported as a failure, not smoothed into a friendly label.

## P1 was falsified and it taught us the important thing

Predicted: at least 60% of units nameable. Measured: **11%**. The reason is not that the model is unreadable - it is that the
concept list only contains SIX computable concepts. A unit can only be named if we can compute a candidate concept to test it
against, so nameability is capped by the size of our concept vocabulary, not by the model.

That turns "make the model transparent" into a concrete data problem: enumerate more measurable concepts and more units become
nameable. It is the same lesson as coverage - the ceiling is what we can generate and test, not what the network can hold.

## What this is good for, plainly

  * a name is a claim, and every claim here carries its AUROC, its firing rate and its three-step test result
  * reuse: a verified pattern can be taught, checked and cited in an explanation ("I said 132 because the units column carried
    and the tens column then carried")
  * honesty at scale: 228 units said "UNNAMED" instead of receiving invented meanings
