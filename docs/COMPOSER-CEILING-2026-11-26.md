# 2026-11-26 - the composer's ceiling is 65.7% on the same 300 held-out questions where the trained model gets 3.0%

Before building a composer, measured whether one COULD work: for each of the 300 held-out-by-original-question problems, can the
reference answer be reached by arithmetic on numbers that literally appear in the QUESTION? That is the upper bound for any parser,
even a perfect one. `tools/probe_composer_coverage.py`.

    question already contains the answer as a number        38/300 = 12.7%
    answer reachable by ONE operation on two question nums  94/300 = 31.3%
    answer reachable by TWO chained operations              65/300 = 21.7%
    questions that state an expression like "12 + 35"       33/300 = 11.0%

    COMPOSER CEILING with a perfect parser: 197/300 = 65.7%

Compare, on the same 300 questions, all measured tonight:

    trained 15M next-token model, best of six configurations    3.0%
    composer ceiling (perfect parser + exact arithmetic)       65.7%

## Why this matters more than another training run

Six training configurations landed inside 1.4-3.0%, and the spread was noise. The composer's ceiling is twenty times higher on the
same questions, and it uses exactly the parts this project has already proved rather than the one part that keeps failing:

    single-operation arithmetic     100.0% on 300 unseen (9,084 params, shipped)
    four-digit addition w/ carries 100.0% on 500 problems
    composition of two operations  69.7-74.6%
    exact arithmetic in code       whatever we ask for, no <unk> on large numbers

## Caveats, stated because they bound the claim

* 65.7% is a CEILING assuming a perfect parser. Building the parser is the actual work, and the parser is precisely the part that
  requires reading the question - so a real composer scores below this, not at it.
* The 12.7% "answer already appears in the question" bucket deserves suspicion: either some MetaMathQA questions are near-trivial, or
  the reference-answer extraction is loose. It should be inspected before that bucket is trusted.
* The search allowed 1-2 operations over up to 3 numbers. Real word problems sometimes need more; those are counted as failures here,
  so 65.7% is a lower bound on what a perfect parser could reach and an upper bound on what a simple one can.

## The decision this puts in front of the owner

Spend the next GPU budget on a seventh training configuration of a line that has produced 1.4-3.0% six times, or build the composer
whose ceiling is 65.7% and whose components are already proved. The evidence favours the composer.
