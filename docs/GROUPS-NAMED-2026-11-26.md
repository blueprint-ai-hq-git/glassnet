# 2026-11-26 (CORRECTED) - naming GROUPS lifts explained margin from 8.1% to 48.4%

This file previously reported "0 of 8 groups nameable, grouping does not rescue explainability". **That conclusion was wrong, and
the wrongness was in my instrument, not the model.** Corrected record below.

## The instrument bug

A group's signal was the SUM of its units' activations. ReLU activations are never negative, so a sum over 20-70 units is
positive on EVERY example. Firing was tested as `sum > 0`, which is always true, so "fires when the concept is absent" came out
as **1.00 for all 8 groups** and the exclusivity gate could never pass. A test that cannot pass is worse than no test.

Fix: a group fires when its signal exceeds **its own median** on the probe set - a data-derived threshold instead of zero.

## Corrected result

    units            256, clustered into 8 groups by their 24-concept fingerprint (own k-means, numpy only)
    named groups     2 of 8 -> group 4 "four_digits_sum_15_plus", group 6 "result_below_150"
    margin carried   48.4% by the named groups        (single units managed 8.1%)
    top 3 groups     75.7% of the margin

Both predictions now hold: named groups carry >= 40% (48.4%, **6x the single-unit figure**) and at least 2 groups pass the bar.

## What this means

A group of mixed units, scored and tested AS A WHOLE, behaves like a clean concept that the individual units inside it could not
be. That is the answer to yesterday's problem: we could not name the deciding units one at a time (8.1%), but we can name the
groups they belong to and that covers nearly half the decision, and three groups cover three quarters of it.

So the explanation format changes shape: not "neuron 34 decided this", but "this group of 34 units, which as a whole means
*the four digits sum past 15*, decided this" - and the arithmetic still sums to the margin exactly.

## Still open

* 48.4% is not 100%. The remaining margin comes from groups that failed the naming bar, and the output states that share rather
  than hiding it.
* The group test is one reading (thresholded sum). A group described by the INPUTS that drive it may name better still - that is
  the next refinement, and it is how the single-unit cards were first built before being matched to concepts.
