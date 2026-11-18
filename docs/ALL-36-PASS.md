# 36 of 36 concepts pass the label-honesty gate - and what that number does NOT say (2026-11-18)

The activation dial had stalled at **34/36**: `mentions_body` and `mentions_sport` failed at every uniform pressure, and pushing the pressure higher (lam-act 2.0) started failing *common*
concepts instead. So the last arm aimed the pressure at the two of them only, with `TALK_BOOST="mentions_body:3,mentions_sport:3"` at the best uniform operating point (lam-act 1.5).

```
                                   lam-act 1.5    lam-act 1.5 + boost x3 on the two
readable-only pass                  34/36               36/36      <- zero failing
mean AUROC                          0.7573              0.7703
mean fire rate                      14.2%               15.6%
failing in readable only      mentions_body,      none
                                mentions_sport
```

## Predictions, scored

- **P1** "pass count reaches 36/36" - **MET**: 36/36, `failing in readable only : none`.
- **P2** "no concept that passed at 1.5 fails now" - **MET**: the failing set shrank to empty, it did not move; mean AUROC rose 0.7573 -> 0.7703.
- **P3** "the two boosted concepts' firing rates rise toward their support" - **MET in direction**: `mentions_sport` now fires on **62.3%** of windows (AUROC 0.6525) and `mentions_body` on
  **47.5%** (AUROC 0.7099), both passing.
- **FALSIFIER** "they still fail at 3x boost" - **did not fire**. So per-concept positive weighting **is** the mechanism, and the doctrine's fallback (drop the concepts) was not needed.

**This is the first time this project has counted zero failing concepts**, and it was reached by aiming the pressure at the two specific concepts instead of turning the global dial until
something broke.

## What this number does not say

The gate is **AUROC >= 0.55 AND firing >= 2%** - a *minimum* firing floor with no upper bound and **no precision requirement**. It asks whether the neuron ranks positives above negatives, not
whether it fires about as often as the claim is true. A neuron firing on 62.3% of windows for a concept whose formula is rare can pass.

**A calibration check was attempted twice this wake, and both attempts were wrong.** Recording them so nobody repeats it:

1. Comparing firing rate against the **token-level** support in the checkpoint metadata gave a fake **"43.7x over-firing"** for `mentions_sport`.
2. Comparing against `1-(1-p)^32` - the chance the concept appears **anywhere** in a 32-token window - flipped the answer to **"29 of 36 under-firing, 0 over-firing"**.

They disagree because neither is the right denominator: the neuron is scored at the **scored position's** state, not "anywhere in the window", and the audit's own per-window target vector is
the only correct basis. That vector is not exposed by `tools/concept_label_audit.py`, so **no calibration claim is made**, and the script now prints that plainly instead of a number.

The next attempt has to compute **precision = P(target | fires)** and **recall = P(fires | target)** from `lens.api.audit`'s own target vector - which means passing it out of the audit rather
than reconstructing it downstream.

## Status

**36/36 on the stated gate, first time, with the mechanism understood.** Label-honesty in the stronger sense - a neuron that fires *because* its concept is true rather than merely ranking it
higher - remains **unmeasured**, and is the honest next question for this line.
