
## Accuracy rises as concept supervision weakens: 26.69% at quarter strength, the best this project has measured (2026-11-16)

Predictions and promotion criterion for these arms: `lens/runs/LAM-NEXT-PREDICTIONS.txt`, `lens/runs/LAM0-PREDICTIONS.txt`. Numbers from
`lens/runs/LAM-NEXT.txt`, `lens/runs/LAM05-DECODE.txt`.

```
arm                     concept supervision   VAL acc   VAL ppl   coverage T=0.7
shipped (window labels)       1.0 (window)      24.58%    181.04       0.292
next-target lam 1.0           1.0 (next)        26.12%*   169.67       0.173
next-target lam 0.5           0.5 (next)        26.52%    158.85       0.167
next-target lam 0.25          0.25 (next)       26.69%    150.64       pending
                                          * 3-seed mean; the others are one seed each
```

## What is measured, and what is not

**Measured**: at a fixed 12-epoch budget on the same data, the same split and the same seed, weakening the concept supervision raises both validation
accuracy and perplexity, monotonically across three settings. The quarter-strength arm is the best pair of numbers this project has recorded.

**Not yet measured**: whether that continues to zero. A **lam 0 control with no concept supervision at all** is running now, with its prediction written
before the launch (`lens/runs/LAM0-PREDICTIONS.txt`): I predict 26.4-27.0%, i.e. the trend continues, because a loss that constrains the hidden state
toward 53 fixed directions should cost a little prediction. The falsifier is recorded too: **lam 0 below 26.12% falsifies the monotone reading entirely**
and the whole "supervision costs accuracy" story gets withdrawn rather than reinterpreted.

**And a caveat in the same breath**: these are one seed per strength. If the lam 0 arm lands inside 26.1-26.4%, all four arms are within seed noise of
each other and the honest statement becomes the coarse one - supervision strength does not move accuracy on this budget at all, while labels still cost
breadth and still buy an exact explanation.

## Why this is uncomfortable for the project's own claim

The front page says readability has a price of **-1.53 +/- 0.44 points at matched parameter budget** - measured on the image task against a plain
control. If the chat side behaves the same way, then for chat that price is **understated**, and the correct statement is not "readability costs about a
point and a half" but "the concept term costs accuracy at this budget, and the readable architecture pays for itself in explanation, not in prediction."

That is a correction to a public claim. It will be published when the control is measured, not before, and it will be published whether or not it flatters
the readable architecture - which is the entire reason the control was launched the moment the trend appeared.
