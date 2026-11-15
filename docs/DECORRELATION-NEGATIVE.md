# Decorrelation works, and it does not buy breadth - the fourth explanation is dead (2026-11-15)

Predictions in `lens/runs/DECORR-PREDICTIONS.txt`. The term was built and verified before launch: analytic gradient derived and checked
against finite differences (max abs error 1.3e-10 random, 1.0e-9 on a deliberately correlated case), registered through the engine's own
`OPS` table, smoke-tested for a full epoch, and proven to enter the loss only when the weight is non-zero.

## Result

```
arm (53 concepts, lam_concept 1.0)   effective dims   pairs |r|>0.50   distinct tokens   final VAL   final ppl   concepts
baseline                              10.09 of 53          4               364           21.08%      269.30      52/53
+ decorrelation 0.05                  11.13 of 53          1               366           21.13%      269.17      52/53
+ decorrelation 0.20                  12.72 of 53          1               366           21.19%      270.67      52/53
```

**The term does exactly what it was built to do.** Effective dimensionality rises **+26%** (10.09 -> 12.72), and the number of concept
pairs above |r| = 0.50 falls **from 4 to 1**, with the specific redundancy it targeted (`cp_certainty` against
`cp_emotion_pos`, 0.671 at lam 0.3) no longer among the top pairs at all. Accuracy is unchanged within noise (21.08 / 21.13 / 21.19)
and 52 of 53 concepts still pass their name audit.

**And breadth does not move: 364 -> 366 -> 366 tokens.** A 26% increase in the effective dimensionality of the concept cloud bought
**two tokens**.

## Prediction scorecard

- **D1 SPLIT**: predicted effective dimensionality above 11.5. 0.05 came in at **11.13 (fails)**, 0.20 at **12.72 (passes)**. The
  lever works, the dose matters.
- **D2 FAILED**: predicted more than 400 distinct tokens; got **366** on both arms.
- **D3 PASSED**: accuracy stayed above 20.50% (21.13%, 21.19%), and marginally *improved* over the 21.08% baseline.
- **D4 PASSED**: the specific pair separates - pairs above |r| 0.50 fall 4 -> 1 and the targeted pair leaves the top list.
- **FALSIFIER NOT TRIGGERED formally** (dims were above 11.0 on both arms) **but its substantive condition held**: breadth stayed
  below 380 on both arms. So the falsifier's conclusion applies even though its literal condition did not fire, and I am stating that
  rather than hiding behind the literal wording.

## What this settles: four explanations, all dead

The readable chat model's vocabulary is capped at about 422 distinct tokens, against 539 for its matched direct-path control. Every
candidate explanation has now been measured:

```
candidate                                  test                                              verdict
answer-layer rank (53-dim family)           sample the same family at random: 1,859 reachable   DEAD
activation spread via supervision           breadth 3x while effective dims stayed 9.7-11.0     DEAD
concept count (36 -> 53)                    +47% concepts bought +8 tokens                      WEAK
effective dimensionality of the cloud       +26% dims (10.09 -> 12.72) bought +2 tokens         DEAD
```

**None of them is the binding constraint.** The remaining structural difference between the two models is the width of the interface
between the hidden state and the answer: the readable model routes everything through **53 scalars**, the direct-path control has
**1024 dimensions** available. That is now the only untested hypothesis, and it is testable without new code because the model already
has a low-rank direct path (`--direct --direct-rank r`) that adds exactly **r** scalars to the interface while leaving the concepts in
place.

This becomes a deliberate diagnostic rather than a ship candidate: adding a direct path costs explanation purity, so the experiment
measures **how much breadth each additional interface scalar buys**, and the answer decides whether widening the interface is worth the
honesty price at all. `lens/runs/INTERFACE-WIDTH-PREDICTIONS.txt`.
