# Anchored RL: the collapse is curable, the calibration is not, and it still does not beat supervised (2026-11-11)

One change against the reward arms: a supervised term (`ce_mix = 0.5`) inside the RL loss, to stop the policy running
away from reality. Same start checkpoint, 2,000 steps, batch 256, lr 0.05.

```
arm                          accuracy   window-echo   confidence error   entropy
control (supervised)          24.17%       31.0%          0.287           4.458
reward-only                   20.17%       57.4%          0.535           1.191
reward + charge 0.1           20.08%       42.4%          0.534            -
correctness paid 3x           21.17%       53.2%          0.573            -
anchored (charge 0.1 + mix)   24.50%       36.8%          0.340           3.529
```

## Scorecard on the predictions, written before the run
- entropy above 3.5: **PASSED**, 3.529 against the reward-only collapse to 1.191. The anchor does what it was for.
- echo below 31%: **FAILED**, 36.8% against the control's 31.0%.
- confidence error at or below 0.30: **FAILED**, 0.340 against 0.287.
- accuracy at or above the control: **PASSED very narrowly**, 24.50% against 24.17%.

## Inside the model, 1,200 held-out windows
```
checkpoint          mass on in-window   mass on correct   entropy   ECE
start                       0.860            0.143          4.259    0.298
supervised continued        0.855            0.150          4.458    0.294
reward-only                 1.410            0.211          1.191    0.530
anchored                    0.973            0.189          3.529    0.338
```

The anchored model puts **more probability on the right answer than any other arm** (0.189 against 0.150 for the
control), keeps a usable distribution (entropy 3.529), and still ends up less honest than plain supervised training.

## The conclusion that has to be said plainly
At this model size and this reward shape, **our own RL method does not beat supervised training on accuracy**, and it
always costs calibration. What it uniquely can do is trade accuracy for repetition control, which no supervised lever
managed. That is a real capability and it is not yet an improvement.

The two honest options from here:
1. use the reward as a small tie-break on top of a supervised run that is already good, rather than as the main
   objective (keeps accuracy, may keep some of the echo control);
2. accept that reward training needs sequence-level credit assignment and a much larger sample budget, and park it
   behind the tokeniser question until the model is worth aligning.

Neither is a claim. Both are choices, and the next arm takes option 1 because it is one change away from what the
control already does well.
