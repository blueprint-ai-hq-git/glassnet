

---

# REDUNDANT IS NOT DISPENSABLE: 6 CONCEPTS PREDICT ALL 14, AND DROPPING THE OTHER 8 COSTS 6.34 POINTS (2026-11-22 14:33)

The original brief asks for one number that no accuracy figure answers: **how many rules must a human actually read?** With 14 named
concepts, the honest answer had better not be "14" if most of them are near-copies.

## First measurement: the redundancy is enormous

How much of each concept is already explained by the other 13 (exact least squares, real labels, 10,000 samples):

```
ink_amount  1.000    col_ink_std  1.000    ink_entropy  1.000    ink_cols_nonzero 1.000
grad_ratio  1.000    symmetry_lr  1.000    unnamed_10   1.000
quad_UL_ink 0.894    ink_rows_nonzero 0.898    unnamed_13 0.804    unnamed_12 0.791
corner_ink  0.731    row_ink_std  0.632        unnamed_11 0.403
```

**Seven of the fourteen are perfectly predictable from the others (R^2 = 1.000).** Greedy selection says **6 concepts predict all 14
at average R^2 >= 0.90**, which looks like the answer: a person reads 6, not 14.

## Second measurement: and that answer is wrong, because redundant does not mean dispensable

A concept can be *predictable* from the others and still carry class information the others do not. So the cost was measured instead
of assumed - same linear classifier, all 14 concepts versus the 6, fitted on 5,000 images and tested on a different 5,000:

```
all 14 concepts   -> held-out accuracy 61.98%
the 6 concepts    -> held-out accuracy 55.64%      difference -6.34 points
most-frequent-class baseline 10.46%   (both beat it)
FALSIFIER FIRED: pruning is not free
```

**P1 NOT MET** (needed to be within 1.0 point; it lost 6.34). **P2 MET** (both far above the 10.46% baseline).

## The honest conclusion, and it reverses what I was about to report

- **The concepts are 90% redundant in predicting each other, and that redundancy cannot be cashed in.** The residual 8-10% of each
  concept's variance is exactly where the class information lives.
- **So "how many concepts must a human read" is NOT 6.** Dropping 8 of them to save a reader 8 names costs **6.34 accuracy points**.
- What redundancy DOES buy is different and still useful: it means the layer can be **summarised** for a reader (6 numbers describe the
  state to 90%) without claiming the discarded ones can be **deleted from the model**.

Two separate questions were being conflated - *how compactly can this be described to a person* (6 concepts) and *how few can the model
use without losing accuracy* (more than 6). The first is a readability gain, the second is not real.

I had the "6, not 14" line written and was one step from publishing it as a readability win. The cost test is what stopped it.
