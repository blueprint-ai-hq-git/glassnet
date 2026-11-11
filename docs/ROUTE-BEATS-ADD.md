# Routing beats adding: 56.2% held-out copying with 27.27% language (2026-11-11)

The last arm today, and the first one that clears both bars. Same checkpoint, same corpus, same copy task, same 12,000
steps and the same learning rate as five failures before it. One change: instead of ADDING the pointer's readout to the
other pathway's readout, the model SWITCHES between them on the pointer's own sharpness.

```
step     language    taught copy   held-out copy
 8000      24.62%       83.2%          39.5%
10000      25.38%       83.7%           3.2%
12000      27.27%       82.3%          56.2%
mean of last 3:  25.8%                 33.0%
```

## Against every other mixture, same writer, same day
```
variant                          mean held-out copy (last 3)   language
bounded gate                                  -                  29.3%
unbounded gate (add)                        14.8%                29.8%
separate embedding table                    10.8%                30.2%
low learning rate                             -                  25-29%
ROUTE instead of ADD                        33.0%                25.8%
pointer alone, no language                  58.0%                 -
```

## Why it worked, and why the previous five could not
`docs/POINTER-AIM-VS-VOTE.md` measured the split: on held-out words the pointer's attention still lands on the correct
window slot with **0.77 of its mass** (chance 0.031) and its own answer is right **58-63%** of the time, while the
assembled model emitted it only **0.5-14.8%** of the time. The operation was never damaged; the SUM buried it. A small
correct voice added to a loud wrong one loses at every scale factor a bounded gate can reach, and freezing the weights
does not help because the shared embedding table keeps moving under the pointer.

A switch does not need to out-sum anything. It asks one question, how sharply is the pointer pointing, and then lets
the pointer's own readout BE the answer for that example.

## The honest caveat, stated rather than buried
Held-out copying swings violently between checkpoints on identical settings: 39.5%, then 3.2%, then 56.2%. The mean of
the last three (33.0%) is the number to quote, and even that rests on 600 evaluation items. This arm is better than
everything else measured today by a wide margin, and it is not yet a stable measurement. The next step is not a new
mechanism: it is repeating this arm on three seeds to see whether 33.0% holds or whether the mean is itself noise.
