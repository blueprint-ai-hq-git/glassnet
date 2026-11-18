# QUALIFICATION: 36/36 measures RANKING, not detection - median F1 is 0.18 (2026-11-18)

The previous entry reported **36 of 36 concepts passing the label-honesty gate**. That number is correct on its stated gate and it is **not** the claim a reader would naturally take
from it. Measured properly, the same neurons are **weak detectors**.

## The measurement that settles it

`lens/api.py#audit` now computes precision and recall from the **same score and target vectors** it already used for AUROC - the only correct basis, after two failed attempts that compared
firing against token-level support (a mismatched denominator; see `docs/history/ALL-36-PASS.md`). Then `tools/concept_pr_curve.py` sweeps the firing threshold per concept, because a single
fixed 0.5 cut is itself a calibration choice:

```
                                              boosted readable model, 800 windows
median F1 at the shipped 0.5 cut                       0.081
median BEST achievable F1 (any threshold)              0.181
concepts with precision >= 0.5 AND recall >= 0.5
   at the shipped 0.5 cut                               0 of 36
   at each concept's own best cut                       3 of 36
median precision at the 0.5 cut                        0.068
median recall at the 0.5 cut                           0.113
median positives per concept                       32 of 800 windows
```

## What this means, stated plainly

- **The gate asks a weak question.** AUROC >= 0.55 with a firing floor asks whether the neuron ranks positives above negatives, which a neuron can do while being a poor detector.
- **Detection is weak.** Even after per-concept threshold calibration, the median achievable F1 is **0.181**, and only **3 of 36** concepts reach both precision and recall above 0.5. A
  neuron named `mentions_sport` that fires mostly on windows where sport is not mentioned does not mean what its label says, however good its ranking.
- **So "36/36 label-honest" must always be quoted with its gate.** The accurate sentence is: *36 of 36 concepts pass the AUROC-0.55 ranking gate; as detectors their median best F1 is 0.18
  and 3 of 36 reach precision and recall above 0.5.*

## Why this is worth publishing immediately

This project's whole claim is readable, checkable meaning. A number that reads like "every concept is honest" while the underlying detectors are weak is exactly the failure mode the project
exists to avoid - and it would have been the headline number if it had gone unqualified. The gate is not wrong; it is **insufficient**, and the sufficiency question now has numbers attached.

## The question this opens

Not "which concept is dishonest" but **"how do we make a concept neuron a detector rather than a ranker"** - a training objective with precision in it (the current concept loss is balanced
BCE, which is ranking-shaped), or a readout that reports PR at a chosen operating point rather than a fixed 0.5. Both are testable, and either would finally let the word "honest" mean what
it says.

## Status of the claims

- **36/36 on the AUROC-0.55 gate**: stands, verified, first time.
- **"All concepts are label-honest"**: **not supported** - qualified here, with the numbers above.
