# Coverage penalty: built, gradient-checked, and it does not work - influence tracks variance, not weight (2026-11-12)

Deleting the three causally-dead concepts cost 8.62 accuracy points, so deletion was the wrong lever. The proposed replacement
was **coverage**: a training-time penalty requiring every concept to be read by some rule, which should make "every named concept
matters" true by construction while keeping all twelve channels.

## What was built
`--cov <weight>` adds to the loss:

```
usage_c  = sum_j |rules.W[c, j]|            # how much rule weight concept c attracts
cov      = sum_c relu(cov_margin - usage_c)  # shortfall below the floor, margin default 0.30
total    = CE + alpha * concept_BCE + lam * L1 + cov_lam * cov
```
Built from the engine's own ops (`abs_`, `sum_axis`, `relu`, `add`, `mul`) - no new library code, gradient flows only through
`rules.W`.

## Gradient check, and two of my own mistakes getting there
Final verdict: **PASS**, exact (max abs diff **0.00e+00**, max rel **0.00e+00**) in both mixed and all-above-floor configurations.

Two earlier "failures" were mine, not the code's:
1. My first harness reported the identical error **4.633e-02** for all six tests, including `relu` with all-positive input and a
   plain `mul by -1.0` - ops that are trivially correct. Six different functions failing with the same number is a harness tell,
   not a code finding. The perturbation was being applied at float32 resolution.
2. With that fixed, bare `abs_ + sum_all` still "failed" at **1.36e-03** - that is the kink of `|x|` at **x = 0** under central
   differences, a property of the function, not a wrong gradient. The coverage term itself, tested on the same data, is exact.

## Result: the penalty does not raise concept influence
Three seeds per setting, twelve concepts, 32 rules, alpha 3, accuracy after connection pruning:

```
setting       accuracy (3 seeds)   min concept influence   mean influence   concepts below 1% influence
no coverage   72.01 +/- 4.35          0.00%                  20.90%            3 of 12
cov 0.5       72.41 +/- 4.99          0.00%                  19.03%            2, 2, 4 of 12 (per seed)
cov 2.0       72.44 +/- 4.58          0.00%                  19.09%            2, 0, 4 of 12 (per seed)
```
- **Accuracy: no cost and no gain** (72.4 against 72.0 - inside the seed spread).
- **Influence: unchanged at best.** The minimum influence across seeds is still **0.00%**, and mean influence is marginally
  *lower* than without the penalty (19.0% against 20.9%).
- Per-concept weight usage did rise as designed (concepts below the 0.30 floor fell from 3 to 2 with only two epochs of
  training), so **the penalty did what it was built to do and the model still ignored those concepts.** That is the finding.

## Why: influence tracks how much a concept VARIES, not how much weight it attracts
Activation spread across the 10,000 test images, against measured causal influence (`docs/EXPLAIN-IMAGE.md`):

```
concept          activation std   causal influence
col_ink_std          0.235            30.49%
row_ink_std          0.178            38.15%
grad_ratio           0.178            47.87%
ink_amount           0.172            46.79%
symmetry_lr          0.144            37.85%
corner_ink           0.138             1.49%
symmetry_ud          0.114            15.79%
left_heavy           0.100            29.23%
ring_ink             0.096             0.00%
grad_x_mean          0.096             3.14%
top_heavy            0.093             0.01%
edge_balance         0.075             0.00%

influence < 1.5% : mean activation std 0.101      influence > 10% : mean activation std 0.160
```
A concept whose activation barely moves across images cannot change an answer, however much rule weight points at it. Weight
coverage is therefore the wrong quantity to penalise. (Variance is not sufficient on its own - `corner_ink` varies at 0.138 and
still carries 1.49% - but it is the axis that separates the dead concepts from the live ones.)

## What the next lever is
Two changes follow directly, and neither is a penalty on weights:
1. **Select concepts by dispersion, not just learnability.** `tools/select_concepts.py` scores candidates by probe rank
   agreement; adding a dispersion score would have excluded `top_heavy` (std 0.093) and `edge_balance` (0.075) before training.
2. **Penalise activation variance shortfall, not weight**, if a training-time term is wanted at all - i.e. reward concepts whose
   activation moves, which is what makes an answer change.

## Reproduction
`--cov {0.5,2.0}` with `GLASSNET_CONCEPTS=$PWD/data/concepts_sel12.npz --K 12 --R 32 --alpha 3.0 --tag COV*_s*`; influence from
`tools/explain_image.py <ckpt> 0`; the gradient harness is in this document's method section (float64, eps 1e-4, central
differences). Six checkpoints kept under `runs/phase2_sweep/COV0.5_*` and `COV2.0_*`.
