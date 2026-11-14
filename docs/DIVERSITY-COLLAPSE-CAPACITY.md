# Why the readable chat arm is narrow: the concept bottleneck cannot separate 12,002 tokens

Date: 2026-11-14. Tool `tools/diversity_diagnosis.py`, runner `lens/scripts/run_diversity_diagnosis.sh`, predictions
`lens/runs/DIVERSITY-DIAGNOSIS-PREDICTIONS.txt` (written before the run), output `lens/runs/DIVERSITY-DIAGNOSIS.txt`.
2,000 held-out positions, both 3-epoch d512 checkpoints.

## The problem being diagnosed

The readable (no-direct) arm reaches 17.25% next-token accuracy but predicts only **28 distinct tokens** against **121** for its
matched control (`docs/history/READABLE-CHAT-NODIRECT-ARM.md`). A narrow talker is not shippable, so this looked for the cause.

## Result - the answer is not what I predicted

```
                                  no-direct (readable)   control (direct ON)
full accuracy                        17.25%                 21.20%
bias-only accuracy                    2.60%                  1.40%
concept-only accuracy (h @ W)        16.45%                  3.25%
distinct tokens, full                    28                    121
distinct tokens, bias only                1                      1
distinct tokens, concept only            27                     12
bias agreement with full              19.5%                   7.2%
concept agreement with full           84.8%                   6.6%
mean predictive entropy (nats)        6.3085                 5.1881
median concepts firing (>0.5)         18 of 36                7 of 36
bias max / second                     0.8722 / 0.7849        0.3458 / 0.3372
```

**Three things follow, and two of them contradict what I predicted:**

1. **In the readable model the concepts really do decide.** Concept-only predictions agree with the full model on **84.8%** of
   positions and score **16.45%** against the full model's 17.25%. The readable path is faithful, not decorative - the opposite of
   the direct-path model, where concept-only agreement is **6.6%** and concept-only accuracy 3.25%.
2. **The bias is NOT the cause.** It agrees with the full prediction on only **19.5%** of positions and scores 2.60% alone; its top
   two entries are now nearly tied (0.8722 vs 0.7849). My P1 (bias agreement >50%) FAILED, and so did P2 (concept-only should be
   more diverse than full - it is 27 against 28, i.e. the same).
3. **The model is not over-confident, it is UNDERTALKING.** Mean predictive entropy is **higher** for the readable arm (6.3085 vs
   5.1881 nats), the opposite of my P3. It is uncertain, and still only ever emits 28 different words.

## The cause: 36 nearly-saturated concepts cannot cover a 12,002-token vocabulary

**18 of 36 concepts fire on a median window** (control: 7). Half the concept vector is ON most of the time, so the 36 numbers carry
very little distinguishing information per window, and the 36 -> 12,002 projection has to squeeze every possible next token through
that narrow, mostly-saturated code. The model can learn roughly **28 output modes** with it and no more. That is a **capacity**
problem in the concept layer, not a bias problem and not a confidence problem.

## What follows

1. **Do not touch the bias and do not add confidence penalties** - both hypotheses are measured out.
2. **Widen the concept layer.** The trainer has **no `--k` flag**: K is whatever the concept module keeps (36 of 43 defined, after
   the 0.005 support floor). So widening means adding concept definitions - a change to the concept set, which is why it is being
   raised with the owner rather than done unilaterally. Target order of magnitude: 96-128 concepts, since 36 gives 28 output modes.
3. Meanwhile the readable arm is still the honest checkpoint: exact explanations (residual ~0 on 100 of 100 windows) at a measured
   4.67-point accuracy cost - it just talks in 28 words.

## Predictions, scored

- **P1 FAILED** (predicted bias agreement >50%; measured 19.5%).
- **P2 FAILED** (predicted concept-only more diverse; measured 27 vs 28).
- **P3 FAILED** (predicted lower entropy; measured higher, 6.3085 vs 5.1881).
- **Falsifier TRIGGERED**, which is what redirected the conclusion away from the bias and onto concept capacity.

Three failed predictions in a row on the same run is worth stating plainly: my intuitions about this model were all wrong, and the
measurements are the only reason the conclusion is right.
