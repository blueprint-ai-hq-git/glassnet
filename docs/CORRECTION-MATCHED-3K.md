# CORRECTION: the resolvable free-running gain was TRAINING LENGTH, not the radius (2026-11-18)

One wake ago this project published "**the first resolvable free-running gain in this line**": radius 1.0 at 3000 steps reaching 8.44% against the 1500-step reference's 5.00%, paired CI
**[+1.75, +5.44]**. That comparison was **unmatched** - 3000 steps of training against 1500 - and the report said the matched control would separate the two. It has now run.

## The matched comparison, both arms at 3000 steps

```
                          radius 1.0     radius 0.9
held-out teacher-forced      16.80%         21.03%
free-running                  8.44%          6.88%
repetition                    0.923          0.923
k=16 memory                   4.434%         1.312%
Wh spectral radius            1.0000         0.9000

paired free-running difference (1.0 - 0.9): mean +0.625 tokens of 40
bootstrap 95% CI [-0.38, +3.56] points  ->  NOT RESOLVABLE
```

**The free-running difference is not resolvable at matched training length.** The interval includes zero. So the earlier resolvable gain came from **doubling the training**, not from the
radius. The claim published last wake is **withdrawn**, and the withdrawal is being made on the same page that carried it.

## Predictions, scored

- **P1** "at matched 3000 steps the free-running CI includes zero" - **HOLDS** ([-0.38, +3.56]). This was the prediction written *against my own previous result*, and it won.
- **P2** "radius 0.9 has higher held-out accuracy at matched length" - **HOLDS** (21.03% vs 16.80%, a **4.23-point** difference).
- **P3** "radius 1.0 keeps at least 3x the k=16 memory" - **HOLDS** (4.434% vs 1.312% = **3.38x**).
- **FALSIFIER** "matched CI excludes zero in favour of radius 1.0" - **did not fire**.

## What the matched pair now says about the shipped choice

At equal training, radius 1.0 buys **3.38x the memory** (which still fails the 10% gate: 4.434%) and an **unresolvable** generation difference, and it pays **4.23 points of held-out
accuracy**. The shipped setting is radius 0.9, and this comparison supports keeping it: unclamping is not a better model, it is a model that remembers slightly more and predicts worse.

## The other half of the story: training length helps both physics and accuracy at radius 0.9

```
radius 0.9:  1500 steps -> 3000 steps
held-out         16.27%  ->  21.03%
k=16 memory      0.878%  ->  1.312%
free-running      5.00%  ->   6.88%
```

Nearly **5 points of held-out accuracy** from doubling the length, and memory improves too. The lever that actually moved this model is **more training**, at the radius already
shipped - which is exactly what the unmatched comparison was accidentally measuring.

## The general lesson, recorded

A comparison across two things at once (training length AND radius) produced a real-looking, resolvable, publishable number that meant something other than what it appeared to mean.
The control was already scheduled when the claim went out; the honest option would have been to hold the result one wake, and that is what should happen next time.
