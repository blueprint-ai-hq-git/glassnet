
## The family is complete: accuracy is monotone in supervision, coverage is NOT (2026-11-16)

Numbers from `lens/runs/LAM0.txt`, `lens/runs/LAM0-DECODE.txt`, `lens/runs/LAM-NEXT.txt`, `lens/runs/LAM05-DECODE.txt`. Predictions for every arm were
written before it ran; the 3-seed check on the no-supervision arm is running now (`lens/runs/LAM0-3SEED-PREDICTIONS.txt`).

```
arm                     supervision   VAL acc   VAL ppl   coverage T=0.7   repetition
no supervision (lam 0)       0.0        26.98%    147.81       0.190          0.013
next-target lam 0.25        0.25        26.69%    150.64       0.150          0.079
next-target lam 0.5         0.50        26.52%    158.85       0.167          0.002
next-target lam 1.0         1.00        26.12%*   169.67       0.173          0.002
shipped (window labels)     1.00        24.58%*   181.04       0.292          0.027
                                        * three-seed mean; others one seed
```

## Accuracy is monotone. Coverage is not, and the correction matters

**Accuracy** rises perfectly monotonically as supervision weakens: 26.12 -> 26.52 -> 26.69 -> 26.98. That is four points, same data, same split, same
seed, same 12 epochs, with the concept loss printing exactly **0.0000** on the no-supervision arm so the control is verified rather than assumed.

**Coverage does not follow that pattern.** Among the supervised arms it rises with strength (0.150 -> 0.167 -> 0.173), which is the direction I reported
earlier today - but the **no-supervision arm comes in at 0.190**, above all three of them. So coverage has no monotone relationship to supervision
strength: it is lowest at quarter strength, higher at half and full, and highest with supervision switched off entirely.

**This corrects a claim I made an hour ago.** I wrote that breadth moves monotonically the other way as supervision weakens (0.173 -> 0.167 -> 0.150). With
the no-supervision number in, that is false, and it is corrected here rather than quietly dropped. The honest version is narrower: **accuracy is a
monotone function of supervision strength; coverage is not.**

## What the concept supervision actually buys, measured from both ends

- Removing it entirely gains **+0.86 accuracy points** (26.98 vs 26.12) and **+0.017 coverage** (0.190 vs 0.173)
- The shipped configuration, which uses *window* labels rather than next-token labels, holds **0.292 coverage** - the only configuration in this project that
  reaches that level - at **24.58%** accuracy, i.e. **2.40 points below** the no-supervision arm

So the concept layer's cost is now measured from both ends, and the benefit is concrete: it is the only thing in this family that produces the vocabulary
breadth of the shipped model, and it buys an exact explanation that no other arm here can give. It pays for both in accuracy.

## Nothing in this family ships

`lam 0` fails the same promotion test as every other arm: coverage **0.190** is below both the shipped 0.292 and the 0.27 bar fixed before these runs. It also
has no concepts, so it cannot explain anything. The shipped model stays the shipped model.

The remaining claim - that the concept term costs ~2.4 points on chat - rests on one seed for the no-supervision arm. Seeds 1 and 2 are running with their
predictions recorded: **S1 all three above 26.12**, **S2 mean 26.8-27.1 with sd at or below 0.20**, **falsifier: any seed at or below 26.12 withdraws the
+0.86 and I say so.**
