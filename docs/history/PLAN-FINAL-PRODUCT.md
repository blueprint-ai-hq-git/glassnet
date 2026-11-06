# PLAN - from where we are to the final product (written 2026-10-17)

Written so that any session, including one starting cold, knows the order and the bars. One change at a
time. Every claim in this file is checked against a result file before it is repeated anywhere.

## Where we are, in one line each
* Readable machine: works. Named concept wall, per-concept audit against its own name, `explain()` is
  the model's own arithmetic. Six lines pass their own label-honesty gates 100%, speech 7/7.
* Smart: no. Best honest next-word accuracy on unseen dialogue is **9.84%**; every model we have is
  **word salad** on the text meter; MMLU 26.0% against 25% chance; GSM8K 0/200; code that parses 1/100.
* The old records (54.89 / 56.07 / 63.85) were measured on val windows that appear verbatim in the
  models' own training text (63.21% / 66.07% / 37.87% of them). They are not accuracy. Do not quote them.

## The order (each step gated on the one before it)
1. **H-block on the repaired corpora** (running/queued): reproduce control, no-concept-supervision,
   window 64, 192 epochs, d512 192 epochs. Answer: does readability cost accuracy, or is the ceiling
   data and schedule?
2. **Price readability properly**: train a plain model with the SAME parameter budget, same data, same
   seeds, no concept layer, no audit. Both numbers printed side by side on the same honest val. This is
   the head-to-head that decides whether the readable design is a cost or a free win.
3. **Seeds and error bars**: every claim above gets 3 seeds, mean +/- spread. Until then a margin is a
   direction, not a win.
4. **Make it talk**: no "it talks" claim is allowed until the text meter clears the salad bar
   (midpoint between real held-out text and the same text shuffled). Current best: -11.50 against a bar
   of -7.94. Fixes to try in order: longer training on the honest corpus, wider concept wall, a
   tokeniser that respects whole words, and only then architecture changes.
5. **General knowledge**: re-run the benchmark ruler (MMLU, GSM8K, code) on the honest-corpus models,
   with the datum and the chance baseline printed beside every number.
6. **Code line, small and big**: close the named gap - pass@1 with a sandboxed runner instead of
   "does it parse". Both model sizes, same tasks, same sandbox.
7. **Then** the release questions: what the licence is, what is claimed, what is measured, what is a
   named gap.

## Bars fixed in advance (no bar may be invented after a result)
* Any new talk number must be reported on the repaired val with the overlap count shown.
* A model may be called an improvement only if it beats the previous best on the SAME held-out text by
  more than the seed spread, and does not get worse on the text meter.
* "Readable costs nothing" may only be said if the plain same-size control is not better by more than
  the seed spread.
* Capability numbers must print their chance baseline beside them.

## The picture side (part of the product, not a debug dump)
`bash lens/scripts/run_make_figures.sh` rebuilds every figure from result files.
`lens/figures.py` holds the rules: numbers and plain words only, never internal paths on the face,
provenance kept in `lens/runs/figures/MANIFEST.json`, and "not measured" shown as absent rather than
guessed. `bash lens/scripts/product_check.sh` is the one command a normal person runs: it checks every
line against its gates, refreshes the pictures, and prints a plain verdict.
