# The vocabulary deficit is CAPACITY, not convergence (27-epoch readable arm, 2026-11-15)

Predictions were written in `lens/runs/DIVERSITY-CONVERGENCE-PREDICTIONS.txt` before the run. The question: diversity had climbed
49 -> 78 from 9 to 18 epochs, so was 78 tokens simply **under-training**? If so, training longer fixes breadth and the concept layer
does not need widening.

**Answer: it is capacity. Training longer does not fix breadth.**

## The trajectory

```
epoch   ppl      VAL      epoch   ppl      VAL
  1    663.45   10.02%      15    253.45   21.52%   <- VAL peak
  3    404.41   16.05%      16    253.08   21.43%
  6    308.88   19.12%      18    253.55   21.40%
  9    273.94   20.72%      19    253.77   21.43%   <- held-out best 23.69%
 12    258.97   21.31%      21    257.84   21.27%
 14    254.79   21.34%      24    264.23   21.14%
                            27    273.68   20.91%   <- final
```

The readable arm **peaks at epoch 15 (21.52%, ppl 253.45), holds until epoch 19, then decays** - validation perplexity rises from
253.08 to 273.68 over the last 11 epochs while training accuracy keeps climbing. So even the readable arm eventually overfits; it just
does it far later and far more gently than the direct-path control (which reached ppl 54,751 by epoch 18).

## Diversity

```
checkpoint                     distinct tokens   concept-only acc   concept agreement   median concepts firing
18-epoch readable (78 was)          78               22.75%               83.4%                     13
27-epoch readable (best, ep19)      83               22.10%               82.8%                     12
```

**+50% more training bought +5 distinct tokens (78 -> 83, +6%), and cost accuracy (21.48% -> 20.91% final).** Meanwhile the
direct-path control reaches **226** distinct tokens. So breadth is not a function of training length.

## Predictions, scored honestly

- **D1 FAILED**: predicted more than 90 distinct tokens; got **83**.
- **D2 FAILED**: predicted final VAL above 21.48%; got **20.91%** (best epoch 21.52%).
- **D3 FAILED**: predicted final perplexity below 253.02; got **273.68** (best 253.08).
- **Falsifier: its substantive clause fired, its literal second clause did not.** The falsifier was "diversity plateaus below 85 tokens
  **while accuracy still rises**". Diversity did plateau below 85 (**83**), but accuracy did *not* keep rising - it peaked at epoch 15
  and decayed. So the falsifier does not fire in its written form, and I am not claiming a clean hit. What the data supports is stated
  directly rather than through the falsifier: **a 50% increase in training bought 6% more vocabulary and no accuracy, so vocabulary
  breadth is not a training-length problem.**

## What this settles

1. **The widening question is now evidence-based rather than a hunch**: the only remaining deficit of the readable model cannot be
   closed by training longer, because training longer makes it slightly worse.
2. **The useful epoch count is about 15-19, not 27.** Both the 18-epoch arm and its 27-epoch repeat peak there; future arms should stop
   around 18 and spend the budget on architecture instead of epochs.
3. **The 18-epoch numbers stand as the arm's ceiling**: 21.48% VAL, ppl 253.02, 78 tokens, 36/36 concepts.

The widening arm (18 additional concepts whose tokens are disjoint from all 283 already in use) is the next experiment, and this
document is why it is the next experiment.
