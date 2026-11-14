# What the direct pathway is worth: the concept layer alone keeps about 18% of the model's accuracy

Date: 2026-11-14. Tool `tools/direct_path_cost.py`, runner `lens/scripts/run_direct_path_cost.sh`, predictions
`lens/runs/DIRECT-PATH-COST-PREDICTIONS.txt`, output `lens/runs/DIRECT-PATH-COST-ALL.txt` and per-checkpoint JSON.
Inference only, no training.

## Why

The open decision was whether to constrain/remove the direct pathway (`lens/model.py:427`,
`logits += relu(x @ dmix) @ E.T`) and retrain for readability, or keep it and report both shares. A decision needs a cost, and
the cost can be estimated with zero training: evaluate the checkpoint with that term removed, i.e. `logits = h @ W + b`, which is
exactly the concept path.

**This is an UPPER BOUND on the cost of removal**, not a prediction of a retrained model: this checkpoint was trained WITH the
pathway, so a model trained without it can recover part of the loss. It is never quoted as "removal costs exactly this".

Metric: next-token accuracy over 2,000 held-out positions (32-token window -> the token that follows), using the project's own
boundary rule so no window crosses a conversation.

## Result

```
checkpoint        full model acc   concept-path-only acc   loss       concept path keeps   agreement
d512  e9 s0           28.25%              5.25%           -23.00 pts        18.58%          18.75%
d1024 e9 s0           29.30%              5.25%           -24.05 pts        17.92%          19.70%
```

- **P1 FAILED**: I predicted full accuracy in the 15-25% band; it is **28.25% / 29.30%**, slightly above. The metric is sane, my
  band was wrong.
- **P2 HOLDS**: concept-only accuracy is **5.25%**, far below half of full.
- **P3 HOLDS**: the loss is larger at d1024 (**24.05** vs **23.00** points), consistent with the share shrinking with width.
- **Falsifier NOT TRIGGERED**: the concept path is **not** redundant at inference.

## What this means

**The layer that the audit rates 36/36 name-honest reproduces less than a fifth of the model's own accuracy on the model's own
task.** Only 18.75% (d512) / 19.70% (d1024) of the full model's predictions even agree with what the concept path alone would
predict.

Three lines of evidence now say the same thing from different directions:

```
instrument                                  d512            d1024        direction
own-label audit                             0.9097          0.9754       concepts look BETTER
held-out synonyms                           0.6980          0.7178       meaning lags     (gap widens)
share of the predicted logit                18.00%           8.89%       coverage shrinks
accuracy retained without the direct path   18.58%          17.92%       performance collapses
```

## Consequence for the decision

**Removing the direct pathway is not a cheap cleanup - it is a research bet.** At inference the concept path alone loses 23-24
accuracy points, so constraining the path during training would have to make the concept layer learn far more than it currently
does, and the audit's 36/36 is no evidence that it can, because that audit scores concepts against their own training target.

Two honest options, with what each costs:

1. **Report both shares and keep the direct path** - costs nothing, ships today, and every explanation states that concepts account
   for a median 8.9-18% of the decision. This is the truthful description of where the project actually is.
2. **Constrain the path and retrain** - a real experiment: at least one full-length arm at matched parameters, with the accuracy
   and copy numbers measured alongside the share. Expect to pay points; the question is how many and whether the concept layer can
   grow to carry the work.

Recommending (1) as the default while (2) runs as an experiment, because (1) is honest today and (2) is not guaranteed to succeed.
