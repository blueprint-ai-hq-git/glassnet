# Phase 2 rule layer: order is worth +6.2 points, and it costs concept honesty (2026-11-11)

Three runs of the same three-layer model (36 concepts -> 96 rules, 3 wires each pruned to 2 -> answer), differing only in
what the concept layer is fed. Held-out next-token accuracy on the matched split, with the trivial baseline printed by the
script itself:

```
input to the concept layer        accuracy   concept honesty (mean)   connections/rule
mean of the 32-token window         9.20%          86.5%                 3 -> 2
last token only                    15.38%          70.7%                 3 -> 2
trivial (always the common token)   5.42%           -                     -
ablation (no rule layer, same input) 16.65%          -                     -
```

**Prediction on record: "beats the printed trivial baseline by at least 1 point". PASSED - 15.38% against 5.42%, a
9.96 point gap.**

## The two things this establishes
1. **Word order is worth +6.18 points.** Mean pooling destroys order, and a next-token predictor without order cannot do
   much better than frequencies. This is the largest single effect measured on this layer, bigger than any optimiser or
   sparsity setting.
2. **Concept honesty and accuracy pull in opposite directions here.** The concepts are named for window contents ("this
   window contains 'the'"), so they need the whole window to be honest - honesty 86.5% with mean pooling. The rules need
   the last token to predict anything - accuracy 15.38%. Each input helps one job and hurts the other, in the same run.

## The honest cost of readability
The ablation, with no rule layer at all, scores **16.65%** against the rule model's **15.38%**: the 96-rule bottleneck
costs **1.27 points** for the ability to read every decision as two concepts and a rule. That is the price of the middle
layer, measured, not assumed.

## The fix now running
Feed the concept layer BOTH the window mean and the last token, so the names stay checkable and the rules get order.
Prediction: accuracy at or above 15% AND concept honesty at or above 85% in the same run. If that fails, the layer needs
two concept groups (window concepts and position concepts) rather than one input that tries to serve both.
