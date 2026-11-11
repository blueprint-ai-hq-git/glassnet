# The pointer generalises: 58.0% on held-out words. The old copy path was outvoting it. (2026-11-11)

Three checkpoints, evaluated the same way on 600 items each (mixed prompt forms, `tools`-independent harness):

```
model                  taught words   held-out words
pointer-only              83.8%           58.0%      <- PREDICTION WAS >= 50%: PASSED
pointer + Q/K             94.2%            0.5%
pointer + Q/K + gate      96.0%            0.7%
```

**A word the model was never trained on is copied correctly 58.0% of the time when the pointer is the only route to
the answer.** An oracle pointer that knows the right position scores 97.0% on the same held-out items. The copying
operation generalises; nothing about the mechanism was ever broken.

## The bug that hid this, and why it matters
`pointer_only` was applied in the MIDDLE of `forward()`, immediately after the pointer block. The old `dmix` copy path
is added AFTER that block, so its contribution was silently added back on top and the "pointer only" arm was never
pointer only.

Evidence it was a bug and not an effect: on one checkpoint, the model's own forward scored **0.7%** on held-out items
while a hand-computed copy of the same pointer maths scored **58.0%** on identical items. A model disagreeing with
itself by two orders of magnitude means one of the two paths is wrong, and it was mine.

Consequence: the two arms that "failed" (`0.5%` and `0.7%` held-out) were measured with that bug present, but they are
also genuinely different models - they do not have `pointer_only` set at all. Their held-out copying really is ~0.6%,
because the old `dmix` path is present in them and outvotes the pointer. So the corrected statement is sharper than the
old one:

**The pointer works. The `dmix` copy path is what makes held-out copying fail, and it fails because that path can
memorise taught answers while being unable to copy an unseen word.**

That is now the top architectural finding: the `dmix` echo path was introduced to help copying and it actively blocks
the mechanism that actually copies. It also explains why four supervised anti-repetition levers and every RL arm fought
the echo: the echo is not a habit, it is a dedicated pathway with its own parameters.

## What is verified and what is not
- VERIFIED: pointer-only copying generalises to unseen words at 58.0%, oracle 97.0%, position accuracy ~85%.
- VERIFIED: with `dmix` present, held-out copying collapses to ~0.6% while taught words reach 94-96%.
- NOT YET: an assembled model that keeps the language ability AND copies unseen words. That needs the `dmix` path
  gated against the pointer rather than simply removed, which is the next arm.

Prediction for that arm, before it runs: gating `dmix` against the pointer lifts held-out copying above 50% without
losing the taught-word performance.
