# CORRECTED: concept.W alone does nothing; the pointer is the whole mechanism (2026-11-13)

**This document replaces a version published minutes earlier that quoted unpaired arm means. The correction is the point, so it is
first.**

## The error, and what pairing does to it
The first version compared arm averages against each other: `concept.W`-only held-out copy **75.30** against the unfrozen arm's
**74.02**, read as **+1.28 points of protection for no language cost**. That is an **unpaired** comparison, and these arms are
**seed-matched**: the unfrozen arm's three seeds are 73.9 / 75.9 / 75.7 and the `concept.W` arm's are 74.1 / 75.9 / 75.8. Paired on
seed the differences are **+0.2 / 0.0 / +0.1**.

```
arm              copy gain (paired)   sd     t     seeds up   language change (paired)   t
concept.W only        +0.10          0.10   1.7      2 of 3            -0.03            -1.0
pointer only          +2.07          0.91   3.9      3 of 3            -0.79            -6.4
full freeze           +1.95           -     6.2      5 of 5            -0.74             -
```

**`concept.W` alone does nothing measurable** - +0.10 points of copy (t = 1.7 on three seeds) and no language change. The pairing
error is the second time this has happened in this line (`docs/COPY-FREEZE-SEEDS5.md` documents the first), which is exactly why the
rule is to report paired numbers on seed-matched arms and never the difference of two means.

## What the arms actually show
```
arm                frozen                      held-out copy   language   paired copy gain
none               -                           74.02           16.30      -
concept.W only     concept.W                    75.30           16.46      +0.10  (nothing)
pointer only       ptrW, Wq, Wk                 77.20           15.70      +2.07  (t = 3.9, 3/3 seeds)
full freeze        ptrW, Wq, Wk, concept.W      75.97           15.56      +1.95  (t = 6.2, 5/5 seeds)
```

- **The pointer matrices are the whole mechanism.** Freezing `ptrW`/`Wq`/`Wk` gives **+2.07** points of held-out copy on three seeds,
  essentially the same protection as freezing all four (+1.95 over five seeds).
- **`concept.W` contributes nothing** to the protection, despite the earlier ablation showing that removing it costs 15 points of
  held-out copy. Those are different questions: an ablation says the tensor is *used* by the copy path, and this says freezing it
  against language gradients does not *preserve* the ability. A component can matter to a computation without being the thing
  language training overwrites.
- **The cheaper set wins on simplicity, not on language cost.** Pointer-only and all-four cost about the same language (-0.79 against
  -0.74, and those come from arms with different seed counts, so they are not distinguishable here). Keeping the extra tensor buys
  nothing, so the ship configuration is **freeze `ptrW`, `Wq`, `Wk` only**.

## The claim I withdraw
The first version also said freezing `concept.W` **on top of** the pointer makes copy worse (75.97 against 77.20). Both numbers come
from separate arms with different seed counts and spreads, and the paired measure of each against the unfrozen arm is the same within
error (+1.95 against +2.07). **That claim is withdrawn** - there is no evidence the extra tensor hurts, only that it does not help.

## Ship configuration after this
```
goal                                    configuration      copy      language
maximum held-out copy (simplest set)    freeze ptrW/Wq/Wk  77.20     15.70
unfrozen baseline                       -                  74.02     16.30
```
One measured trade, not two: **+2.07 points of held-out copy for -0.79 points of language**, from freezing the three pointer
matrices on language batches. Three seeds per arm; the copy gain is 3.9 standard errors, the language cost 6.4.

## Reproduction
`lens/scripts/run_freeze_subset.sh` (prediction and falsifier in the header), logs `logs/sub/{concept,ptr}_s{0,1,2}.log`, checkpoints
`sub_concept_s*` / `sub_ptr_s*` in `lens/runs/chat/`. Late held-out copy = mean of the last three evaluations.
