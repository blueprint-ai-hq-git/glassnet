# explain() on the readable Fashion-MNIST model: real output, and the two defects it exposes (2026-11-12)

The Phase 2 gate asked for: *"Build an explain() that prints: I said X because of rule #a and rule #b, listing the concepts
involved."* This is that, running on the best honest arm - `a3 / 32 rules / pruned`, 76.60% on seed 0, 29 rules kept - with the
concept names and class names a person can read. `tools/explain_image.py <checkpoint> [n] [--pruned]`.

## Verbatim output, two real test images

```
checkpoint final_seed0.npz  (pruned)   rules kept 29   test accuracy 76.60%

==============================================================================
image 0: predicted Ankle boot (score +2.648, logit 2.65)   truth: Ankle boot   CORRECT
------------------------------------------------------------------------------
  I said Ankle boot partly because of RULE #7: fires 0.51 -> pushes Ankle boot +1.798
     when: symmetry_lr=0.47  AND  corner_ink=0.03  AND  row_ink_std=0.68
  I said Ankle boot partly because of RULE #14: fires 0.70 -> pushes Ankle boot +0.965
     when: grad_ratio=0.27  AND  row_ink_std=0.68  AND  grad_x_mean=0.16
  if 'symmetry_lr' were switched off: Ankle boot -> Ankle boot

==============================================================================
image 1: predicted Pullover (score +3.201, logit 3.20)   truth: Pullover   CORRECT
------------------------------------------------------------------------------
  I said Pullover partly because of RULE #18: fires 0.85 -> pushes Pullover +1.410
     when: ink_amount=0.82  AND  left_heavy=0.60  AND  symmetry_ud=0.55
  I said Pullover partly because of RULE #21: fires 0.72 -> pushes Pullover +1.363
     when: ink_amount=0.82  AND  left_heavy=0.60  AND  symmetry_ud=0.55
  if 'ink_amount' were switched off: Pullover -> Sandal
```

Every number is real and reproducible. Contributions are exactly the rule's firing times its answer weight for the winning
class, so they sum to the score minus the answer bias - no approximation.

## Defect 1: rules #18 and #21 are the same rule twice
The explanation for image 1 cites two rules with **identical conditions** (`ink_amount=0.82 AND left_heavy=0.60 AND
symmetry_ud=0.55`) and nearly identical pushes (+1.410, +1.363). Counting it across the kept layer:

```
kept rules                                      29
distinct 3-concept wirings                      20
wiring sets used by MORE than one rule           5  (involving 14 rules)
  rules 1, 13, 18, 21 read the same 3 concepts, answer-row cosine 0.91
  rules 4, 17, 27, 31 read the same 3 concepts, answer-row cosine 0.95
  rules 5, 6          read the same 3 concepts, answer-row cosine -0.96
  rules 8, 26         read the same 3 concepts, answer-row cosine -0.83
```

**CORRECTED 2026-11-12 (this paragraph was wrong when published).** These rules share a wiring but NOT a threshold: grouped
by wiring their biases differ (`+0.2925 / -0.0201 / -1.1766 / -0.6214` on concepts (0,2,10)), so they are a piecewise response
in those concepts, not duplicates. Measured cost of collapsing each group to its strongest member: **-46.76 / -26.30 / -36.95
points** across the three seeds of this arm, with one two-rule group alone worth 12.22 points. So **29 rules is the correct
reading count** and nothing is overstated; what changes is how `explain()` should present a group. Full measurement:
`docs/RULE-CONSOLIDATION.md`.

## Defect 2: the two most honest concepts are causally dead
Switching a concept off and re-reading the predicted class, over all 10,000 test images:

| concept | honesty (rank agreement with its own label) | changes the prediction on |
|---|---|---|
| top_heavy     | 0.968 |  0.01% |
| col_ink_std   | 0.918 | 30.49% |
| ring_ink      | 0.910 |  0.00% |
| corner_ink    | 0.892 |  1.49% |
| row_ink_std   | 0.836 | 38.15% |
| grad_x_mean   | 0.743 |  3.14% |
| symmetry_lr   | 0.626 | 37.85% |
| edge_balance  | 0.618 |  0.00% |
| ink_amount    | 0.605 | 46.79% |
| symmetry_ud   | 0.589 | 15.79% |
| left_heavy    | 0.556 | 29.23% |
| grad_ratio    | 0.556 | 47.87% |

```
mean |spearman| 0.735 (all 12 concepts >= 0.556, none below 0.50)   vs   mean causal influence 20.90%
correlation between honesty and causal influence across the 12 concepts: -0.483
```

`top_heavy` is the single most label-honest concept in the model (**0.968**) and switching it off changes **0.01%** of
predictions. `ring_ink` is 0.910 honest and changes **0.00%**. Meanwhile `grad_ratio` at 0.556 honesty changes **47.87%**, and
`ink_amount` at 0.605 changes **46.79%**.

This is the gate item that says *"whether each concept neuron really fires on its named concept"*. It passes - 0 of 12 below
0.50, mean 0.735 - and it is **not** the property that makes the model readable in use. A neuron can mean exactly what its label
says and still be irrelevant to every answer the model gives. The three concepts with influence under 1.5% (`top_heavy`,
`ring_ink`, `edge_balance`) are 19.3% of the concept vocabulary and carry no decision.

## What this measurement does NOT claim
- It does not claim the model is unreadable. The explanation path is load-bearing: forcing `ink_amount` off turns Pullover into
  Sandal, and 9 of 12 concepts change the answer on more than 1% of images.
- It does not claim the 76.60% is wrong - the audit reproduces the trainer's number exactly.

## Reproduction
`tools/explain_image.py runs/phase2_sweep/A3R32_s0/final_seed0.npz 2 --pruned`, `GLASSNET_CONCEPTS=$PWD/data/concepts_sel12.npz
.env-gpu/bin/python3 tools/phase2_reaudit.py` for the honesty column, and the duplicate count from `rules.W` non-zeros.
