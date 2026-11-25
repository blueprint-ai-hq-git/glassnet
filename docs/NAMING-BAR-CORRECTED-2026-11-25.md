# 2026-11-25 - CORRECTION: with exclusivity required, only 10.5% of units earn a name, not 36.7%

This corrects the write-up committed as `0d6546c` (docs/history/CONCEPT-VOCAB-2026-11-25.md). That document said naming rose
from 10.9% to **36.7%** when the concept list grew from 6 to 24, and concluded "transparency is a data-generation problem".
Both statements were measured with a bar that accepted a name on AUROC alone. **That bar was too weak, and the conclusion
does not survive it.**

## What changed in the instrument

`build_cards` now requires EXCLUSIVITY as well as ranking: a unit is named only if it fires on the concept it is named after
AND stays quiet when that concept is absent (false-fire rate <= 0.15). The parameter had been added earlier but the body never
used it - a silent no-op flag, the exact failure this project is supposed to refuse.

## The same measurement, honest

    bar                        named units        of 256
    AUROC only (wrong)              94             36.7%
    AUROC + exclusivity             27             10.5%

So the 18 extra concepts added **no honest names at all** (6 concepts: 28; 24 concepts: 27). The pre-registered prediction
("a bigger concept list raises naming to >=30%") is FALSIFIED under the correct bar, and the falsifier fired.

## What the surviving 27 actually look like

    unit  34  units_carry        AUROC 1.000  fires when present 1.00  ABSENT 0.10
    unit 141  units_sum_below_5  AUROC 1.000  fires when present 1.00  ABSENT 0.12
    unit 179  tens_carry         AUROC 0.997  fires when present 0.99  ABSENT 0.06
    unit 247  result_below_150   AUROC 0.986  fires when present 0.98  ABSENT 0.00

These are defensible: near-perfect catch rate, and they mostly stay quiet otherwise.

## And 115 units now say what they really are

Instead of a clean label they have not earned, they carry the truth on the card, for example:

    unit 1  ->  UNNAMED (ranks result_below_150 first but also responds to no_carry_at_all, both_units_zero, ...)
    unit 4  ->  UNNAMED (ranks both_units_zero first but also responds to b_units_is_zero, b_is_multiple_of_10, ...)

That is superposition stated plainly. A model this size uses each unit for several things, and the honest report of that is
worth more than 67 extra names that would not have held up.

## Consequence

Transparency is NOT solved by enlarging the concept list - that only enlarges the number of claims. Either the units must be
pushed toward exclusivity by training (a sparsity or decorrelation pressure on hidden units), or the cards must present mixed
units as combinations rather than single names. That is the next question, and the earlier conclusion is withdrawn.
