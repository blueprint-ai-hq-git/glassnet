# The concept-honesty dial on Fashion-MNIST: honest names are expensive here (2026-11-12)

On the chat corpus, raising the concept-loss weight moved concept honesty from 64.9% to 84.0% for 1.86 accuracy points. The
same dial exists in the original Phase 2 Fashion-MNIST trainer (`--alpha` weights the concept BCE against the class
cross-entropy) and this measures its price. Nine models: three weights, three seeds each, 16 rules, everything else identical.

```
| 1.0 | 77.93 +/- 1.40 | 0.587 | 4.7 of 14 | +0.19 | 0.7 | 8.7% |
| 3.0 | 73.65 +/- 3.06 | 0.759 | 2.3 of 14 | +1.38 | 2.0 | 16.0% |
| 10.0 | 63.32 +/- 1.54 | 0.885 | 1.7 of 14 | +1.84 | 3.3 | 10.8% |
```

## The trade, measured
- **Honesty climbs the same way it did on the chat corpus**: mean rank agreement with the named feature rises **0.587 -> 0.759
  -> 0.885**, and the number of neurons better explained by a *different* named feature falls **4.7 -> 2.3 -> 1.7 of 14**.
- **But the price is far steeper here**: accuracy falls **77.93% -> 73.65% -> 63.32%**. Halving the misnaming costs **4.3
  points**; nearly eliminating it costs **14.6 points**. On the chat corpus, x5 concept weight cost 1.86 points.
- **Rule necessity rises with concept pressure** (median per-rule ablation **+0.19 -> +1.38 -> +1.84**), which is the opposite
  of what more rules does - the layer is doing harder work because the concept representation is more constrained.
- **Interaction climbs then falls** (8.7% -> 15.9% -> 10.8%), so per-concept explanations are least complete at the middle
  setting, where the class head is still strong but the concepts are becoming decisive.

## What I recommend, and why it is not the honest maximum
For this task the defensible operating point is **alpha = 3 with 16 rules**: 73.65% +/- 3.04 accuracy, mean rank honesty
0.759, 2.3 of 14 concepts misnamed, median rule worth +1.38 points, interaction 15.9%.

- Going to alpha=10 buys 0.6 fewer misnamed concepts and costs 10.3 more accuracy points - a bad trade by any reading.
- Staying at alpha=1 keeps 77.93% but leaves 4.7 of 14 concepts better described by another name, which is the defect the
  re-audit flagged.
- The 3.6-points-per-misnaming ratio is the honest headline: **on this task you cannot buy honest concept names cheaply**, in
  contrast to the chat corpus where the same dial was nearly free.

## A hypothesis this falsified
I expected the misnamed neurons to be an artefact of nearly-collinear features (a neuron unable to tell `top_heavy` from
`quad_LL_ink`). Measured on 20,000 training images:

```
misnamed pair                        |spearman| between the two named features
top_heavy   / quad_LL_ink                    0.275
horizontal_edges / wide_shape                0.465
vertical_edges   / wide_shape                0.782
quad_LR_ink / symmetry                       0.160
```

Only one of the four is collinear. The worst offender, `quad_LR_ink` (own correlation **-0.084**), pairs with its apparent
target `symmetry` at just 0.160. So misnaming is **not** explained by indistinguishable labels; the neuron learned something
else entirely. The set does contain genuine near-duplicates - `ink_amount`/`quad_LL_ink` at **0.926**,
`ink_amount`/`center_ink` at **0.922** - which is a separate design flaw worth fixing (two names a human cannot distinguish),
but it is not the cause of the misnaming.

## Method notes
- Accuracy is the trainer's final-epoch test accuracy, independently re-measured by `tools/phase2_reaudit.py` on all 10,000
  test images; the two agree exactly, which is what validates each audit.
- All nine checkpoints kept under `runs/phase2_sweep/A*_s*/`; `runs/phase2/` untouched.
- One alpha=10 seed reports a negative median per-rule ablation (-0.04), i.e. deleting a typical rule slightly *helps* that
  model - recorded, not smoothed.
