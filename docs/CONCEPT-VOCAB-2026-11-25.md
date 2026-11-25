# 2026-11-25 - naming scales with the concept list (10.9% -> 36.7%), but AUROC naming OVER-CLAIMS

Sources: lens/runs/CONCEPT-VOCAB.json, lens/runs/NEW-NAMES-VERIFIED.json. Same cached weights as the pattern-card run
(lens/runs/PATTERN-MODEL.npz), so the model is identical and only the concept list changed.

## Naming scales with what we can measure

    6 computable concepts  ->  28 of 256 units named (10.9%)
   24 computable concepts  ->  94 of 256 units named (36.7%)

No retraining, no bigger model, no cleverer labelling - only more measured properties. P1 (>=30%) MET. That confirms the cap
is our concept vocabulary, which makes transparency a data-generation problem, the same shape as coverage.

Top new cards, all at AUROC 1.000: unit 23 units_sum_15_or_more, unit 51/58/87 both_units_zero, unit 34 units_carry.

## But AUROC naming over-claims, and the three-step test proves it

Naming by AUROC alone is necessary and NOT sufficient. Five of the biggest new names went through the full test (exact
examples fire, FRESH examples of the same concept fire, other concepts stay quiet):

    concept                    unit   exact   FRESH   quiet   verdict
    result_below_150            214   96.0%   100.0%   80.0%  CONCEPT
    units_sum_below_5           141  100.0%   100.0%   84.0%  CONCEPT
    both_units_zero              51  100.0%   100.0%   60.0%  NOT a concept
    units_sum_15_or_more         23  100.0%   100.0%   72.0%  NOT a concept
    four_digits_sum_15_plus      95   92.0%   100.0%   60.0%  NOT a concept

**2 of 5 survive.** The failures all pass the exact and FRESH steps and fail EXCLUSIVITY: they fire on concepts other than
their name. A unit can rank its concept first (AUROC 1.0) and still be busy with several things - that is superposition, and
AUROC cannot see it.

So the honest position is: 36.7% of units have a concept they rank first, and the subset that is ALSO exclusive is smaller -
around 40% of those sampled. The naming bar must therefore require the three-step test, not AUROC, and the card must state
which bar it passed.

## Design consequence, next code change

`build_cards` must run the exclusivity step before printing a name, and any unit that fires on more than one concept gets a
card saying "ranks X first but also responds to Y" rather than a clean label. That is the difference between a name we can
defend and a name that merely sounds right.
