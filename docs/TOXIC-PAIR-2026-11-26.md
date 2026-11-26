# 2026-11-26 - the toxic pair is REPLAY + COVERAGE: +1.29 nats together from two mechanisms that are harmless apart

Source: lens/runs/BRAIN-INTERACTION.json (values transcribed from the journal of unit glassnet-interact, which crashed on
KeyError 'plain' AFTER printing and BEFORE saving), lens/runs/BRAIN-ABLATION.json. 8,000 vocabulary, 12,000 fresh-equivalent
examples, three seeds each (3, 4, 5), same architecture.

    arm                  held-out loss      vs plain     accuracy
    plain                  5.4407            -          14.17%
    replay alone           5.4224          -0.018       15.10%
    concept alone          5.4470          +0.006       14.43%
    coverage alone         5.5511          +0.110       13.83%
    concept+coverage       5.5626          +0.122       13.75%
    REPLAY+COVERAGE        6.7328          +1.292       11.63%    <- the poison
    all three              6.9035          +1.463       11.53%

## The interaction is now located

Replay alone is harmless (-0.018). Coverage alone is mildly harmful (+0.110). Together they cost **+1.292 nats** - more than ten
times the sum of their separate effects, and only 0.17 short of the full method's 1.463. Adding the concept loss changes almost
nothing either way (replay+concept 5.4352 against replay alone 5.4224), so the third mechanism is innocent.

## The likely mechanism, stated as a testable claim

Coverage as implemented draws fresh examples from two strata **with replacement** from a limited pool, and replay then fills half
of every batch. The fresh half therefore comes from a small stratified subset, the same windows recur constantly, and the model
drifts onto them. That is a diversity collapse caused by the two mechanisms interacting, not by either idea being wrong: if it is
true, drawing the stratified sample WITHOUT replacement, or stratifying over a much larger pool, should remove most of the 1.29.

## Decisions this settles

* **KEEP replay.** +0.93 accuracy points at three sigma, three times less collapsed, no loss cost.
* **DROP coverage as implemented.** It costs on its own and it is catastrophic in combination with replay.
* The concept loss is neutral for prediction and can be kept or dropped on readability grounds alone - a separate question.

## Also fixed

The ablation script's verdict block assumed a `plain` arm was always present, so the subset run crashed after printing its
numbers. A run without the control now says so and writes its results to BRAIN-INTERACTION.json instead of dying.
