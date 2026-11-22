

---

# CORRECTION: THIRTEEN OF FOURTEEN CONCEPT NAMES WERE ATTACHED TO THE WRONG COLUMNS (2026-11-22 15:02)

**This withdraws the "12 of 14 concepts mean their own name" result and the "`row_ink_std` is a wrong-name liar neuron" result.**
Both were published earlier today. The names were wrong, so the audit was grading the wrong pairs.

## What was found

The stored concept labels in `data/concepts.npz` and the names in `data/concepts_disp10.json` **are not in the same order** - the
name file is sorted by its own `probe` score, the data file is in a different order from an older concept pool. Matching every stored
column against the **named definitions** in `glassnet/concept_pool.py` by recomputing that pool from the same images:

```
 0 stored as ink_amount         is really ink_amount         |r|=1.0000  OK
 1 stored as quad_UL_ink        is really symmetry_lr        |r|=1.0000  MISLABELLED
 2 stored as col_ink_std        is really top_heavy          |r|=1.0000  MISLABELLED
 3 stored as corner_ink         is really grad_x_mean        |r|=1.0000  MISLABELLED
 4 stored as row_ink_std        is really grad_y_mean        |r|=1.0000  MISLABELLED
 5 stored as ink_entropy        is really quad_UL_ink        |r|=1.0000  MISLABELLED
 6 stored as ink_cols_nonzero   is really quad_UR_ink        |r|=1.0000  MISLABELLED
 7 stored as grad_ratio         is really quad_LL_ink        |r|=1.0000  MISLABELLED
 8 stored as symmetry_lr        is really quad_LR_ink        |r|=1.0000  MISLABELLED
 9 stored as ink_rows_nonzero   is really center_ink         |r|=0.9851  MISLABELLED
10 stored as unnamed_10         is really left_heavy         |r|=1.0000  MISLABELLED
11 stored as unnamed_11         is really ring_ink           |r|=0.8444  MISLABELLED
12 stored as unnamed_12         is really ink_rows_nonzero   |r|=0.9665  MISLABELLED
13 stored as unnamed_13         is really aspect             |r|=0.9332  MISLABELLED
```

**Thirteen of fourteen.** Nine of them match their true definition at |r| = 1.0000, which is exact, not approximate. The corrected
map is written to `data/concepts_names_corrected.json`.

## The second problem, independent of the first: the audit was partly circular

Even with correct names, `concept_honest.py` asked whether a **linear decoder fitted to column j** tracks **column j** - which it does
by construction. Its only genuinely informative outputs were the **redundancy numbers** and the **collinearity flags**. So the
"12 of 14 mean their own name" figure was measuring its own fit, and it is withdrawn on that ground as well as on the naming ground.

## What still stands, and what does not

**STANDS (name-independent, computed on column indices only):**
- **7 of 14 columns are perfectly predictable from the other 13 (R^2 = 1.000)** - a property of the numbers, not of the names.
- **6 columns predict all 14 at average R^2 >= 0.90**, and pruning to those 6 costs **6.34 accuracy points (61.98% -> 55.64%)**.
  **Redundant is not dispensable** is unaffected by the naming error.

**WITHDRAWN:**
- "12 of 14 concepts mean their own name."
- "`row_ink_std` is a WRONG NAME neuron tracking something else" - column 4 is really `grad_y_mean`, and its near-collinearity with
  `aspect` (|r| = 0.93) is a genuine property of two ink measures, not a lie.
- The **named** list of the six concepts a reader must read. The **counts** are unchanged; only the names on them were wrong.

## Why this one matters beyond the arithmetic

The readable middle layer is this project's main bet, and its recorded failure - "in the chat model 19 of 36 concepts oppose their
own label" - is a claim about **names**. If the name file and the concept file can be silently out of order for the image concepts,
the same class of mismatch is a live suspect for the chat model's 17-supply/19-oppose result. That audit must now be re-checked for
exactly this bug **before** any repair is attempted, because repairing a naming artefact as though it were a model defect would waste
the next day of work.

Three times today the defect was in the *test*; this time it was in the *labels the test trusted*. The rule that keeps surfacing is
the same one: **a measurement is only as good as the pairing it assumes, and that pairing must itself be measured.**
