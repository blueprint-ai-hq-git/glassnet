# RL dose response: the penalty works, and the REINFORCE objective is the thing fighting it (2026-11-11)

Same start checkpoint, same harness, 2,000 steps, batch 256, lr 0.05. `w_correct = 1.0` throughout this table, so the
only change between rows is the repetition charge.

```
arm                          accuracy        window-echo      confidence error (ECE)
control (supervised)     19.42 -> 24.17%   36.8 -> 31.0%        0.294 -> 0.287
rl1  lam_rep 0.0         19.42 -> 20.17%   36.8 -> 57.4%        0.294 -> 0.535
rl   lam_rep 0.1         19.42 -> 20.08%   36.8 -> 42.4%        0.294 -> 0.534
rl   lam_rep 0.3         19.42 -> 14.17%   36.8 -> 13.3%        0.294 -> 0.726
rl   lam_rep 1.0         19.42 ->  6.25%   36.8 ->  0.6%        0.294 -> 0.920
```

## What the dose response actually shows
1. **The penalty works and it is monotone.** Echo falls 57.4 -> 42.4 -> 13.3 -> 0.6 as the charge rises. There is no
   doubt about the mechanism.
2. **The accuracy price is monotone too**: 20.17 -> 20.08 -> 14.17 -> 6.25. Nothing here is free.
3. **The surprise: at `lam_rep = 0.1` echo went UP from the start point (36.8% -> 42.4%).** A penalty made the
   behaviour worse. The reason is the row above it: reward-only training pushes echo UP on its own (to 57.4%), so a
   small charge is doing nothing but failing to cancel that. The penalty is not creating echo; it is losing a fight
   against the REINFORCE objective, which prefers echoing.
4. **Confidence collapses as the charge rises**: ECE 0.287 -> 0.534 -> 0.726 -> 0.920. The model does not become
   honest, it becomes loudly wrong. That is the measured answer to why this model sounds confident: its confidence is
   a by-product of the objective, not evidence about the answer.

## So the finding is not "find a better dose"
P1 and P3 from the ladder were both falsified, and the dose sweep shows why: with `w_correct = 1.0`, the correctness
term is small (about 0.12 of the reward) compared with the side penalties, so the model is paid more for managing side
effects than for being right. The lever is the SHAPE of the reward, not the size of the penalty.

## Next arm, one change
`w_correct = 3.0` with `lam_rep = 0.1`: being right pays three times what it did, the charge stays small.
Prediction: accuracy within 2 pp of the 24.17% control AND echo below 31.0%. If that fails, reward shaping is not the
answer at this scale and the next move is a KL anchor to the starting policy so the model cannot wander off the
supervised solution while collecting rewards.
