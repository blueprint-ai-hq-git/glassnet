# The readable model's ceiling is higher than 9 epochs showed: 21.48% VAL, 78 distinct tokens, and still improving

Date: 2026-11-14 20:05. `nodirect1024long_s0`: same readable configuration as `nodirect1024_d512_s0` (d1024, seed 0, 600k windows,
full corpus, no direct pathway) with **18 epochs instead of 9**. Config-only change; predictions `lens/runs/READABLE-LONG-PREDICTIONS.txt`
written before launch. Output `lens/runs/READABLE-LONG.txt`, `DIVERSITY-nodirect1024long_s0.json`,
`READABLE-LONG-CONCEPT-HONESTY.txt`.

## Result: more training improves every dimension, and it had not converged

```
                          9 epochs    18 epochs    change
final VAL accuracy         19.95%       21.48%      +1.53
final VAL perplexity       291.04       253.02      -38.02
held-out best (position)   21.57%       23.67%      +2.10
next-token, 2000 positions 22.45%       23.15%      +0.70
distinct tokens predicted      49           78        +29
concept fidelity           34 of 36     36 of 36       +2
median concepts firing     15 of 36     13 of 36       -2
concept own-label AUROC     0.7193       0.8384      +0.12
concept held-out synonyms   0.5392       0.5676      +0.03
```

Predictions, all three held:

- **P1 HOLDS**: predicted at least 20.95% VAL; got **21.48%**.
- **P2 HOLDS**: predicted perplexity below 291.04; got **253.02**.
- **P3 HOLDS**: predicted diversity between 50 and 190 distinct tokens; got **78**.
- The falsifier (flat accuracy and rising perplexity, i.e. already converged at 9 epochs) did **not** trigger.

**The held-out best was set at the LAST epoch (23.67% at epoch 18)**, so this arm is still improving - the ceiling is above 21.48%, not
at it. Anything that quotes the 9-epoch numbers as the readable model's ability is understating it.

## What this does and does not say about the control

The d1024 direct-path control reached **18.57% final VAL, 23.46% held-out best, VAL perplexity 987.76** - but it was trained for
**9 epochs**, and this readable arm for **18**. **That is not a matched comparison, so no head-to-head claim is made.** What can be
said from each arm's own trajectory is that the control's validation accuracy peaked at epoch 6 and then degraded while its
perplexity rose 3.8x, whereas the readable arm improved monotonically through 18 epochs.

**The matched 18-epoch control is the outstanding experiment** and has been queued rather than launched, because it is another hour
of GPU.

## Concept honesty: better than at 9 epochs, still below the direct-path model

```
checkpoint                       own-label median   held-out synonym median
direct path, d512, 9 ep              0.9097               0.6980
readable, d512, 9 ep                 0.7193               0.5392
readable, d1024, 18 ep               0.8384               0.5676
```

Extra training recovers most of the own-label deficit (0.7193 -> 0.8384) and a little of the synonym deficit (0.5392 -> 0.5676), so
the readability/concept-quality trade-off found in `HONESTY-TRADEOFF-READABLE.md` **narrows with training but does not close**.
Held-out synonyms remain the weak spot: 0.5676 against the direct-path model's 0.6980.
