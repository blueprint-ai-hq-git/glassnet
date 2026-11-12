# Additivity, replicated: a per-concept explanation accounts for at least 96.3% of the effect (2026-11-12)

The earlier additivity check used one model per configuration. It is now measured on **nine** models - three seeds for each
of the three configurations - by comparing the sum of individual concept effects against switching every concept off.

```
configuration          seed 0   seed 1   seed 2     mean +/- std     explained by concepts
UNTIED-11               2.1%     2.7%     3.4%       2.7% +/- 0.6         97.3%
TIED-23                 0.4%     3.7%     2.1%       2.1% +/- 1.6         97.9%
TIED-23 distinct pairs  1.5%     0.1%     2.3%       1.3% +/- 1.1         98.7%

worst single measurement: 3.7%   ->  at least 96.3% of the effect is accounted for by the per-concept decomposition
```

**Prediction on record: "interaction stays under 3% on all seeds". Passed on the means, failed on one seed**
(TIED-23 seed 1 at 3.7%). The claim that survives replication is the worst case, not the mean: across nine models, no
concept interaction exceeded 3.7%.

## Why this is the result that matters for the transparency claim
A reader who is shown the true concepts in a window and each concept's measured effect on the chosen word is reading an
explanation that is **quantitatively complete to within 3.7%**. The remaining 96.3% is not a story about the model, it is
the model's own arithmetic, obtained by intervention rather than by inspecting weights.

That is a stronger and simpler claim than anything the rule layer could support, and it does not depend on the tie:
- tied models give the extra guarantee that a rule's push direction is a function of its concepts (rank 22 of 23);
- untied models give the extra guarantee that every rule is load-bearing (median ablation 0.36);
- **both** give an additive, measured, per-concept explanation.

## Caveats that remain
- Attribution measures each concept's total effect on the predicted word, not a unique share when several concepts push the
  same word. Two concepts that always fire together would each get partial credit.
- The comparison is between "sum of individual effects" and "all concepts off", so it bounds the interaction between concepts
  on that one quantity, not the full joint distribution of effects.
- Window sample is 2,000 held-out windows per model; the numbers move by about a point across seeds, which is why the worst
  case rather than the mean is quoted.
