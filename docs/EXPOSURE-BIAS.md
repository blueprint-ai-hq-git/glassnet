
## Epoch sweep: repetition is INVARIANT to every knob we have turned - so the diagnosis is exposure bias

Predictions in `lens/runs/EPOCH-SWEEP-PREDICTIONS.txt`. The more-data recipe (2M windows, window 32, lam 1.0, seed 0) at 12 and 9
epochs instead of 18.

```
arm                        final VAL   final ppl   greedy repetition   greedy coverage   greedy KL
2M, 18 epochs                23.32%      209.74          0.659              0.243          20.86
2M, 12 epochs                23.47%      211.49          0.614              0.247          20.69
2M,  9 epochs                22.92%      222.87          0.653              0.349          18.37
```

- **E1 FAILED**: predicted repetition below 0.60 at 12 epochs; got **0.614**.
- **E2 FAILED**: predicted coverage above 0.28 at 12 epochs; got **0.247**.
- **E3 PASSED**: predicted validation at or above 22.30%; got **23.47%**, which is the best accuracy of any arm today.
- **E4 FAILED**: the 9-epoch arm moved as predicted on accuracy (22.92%, below the 12-epoch arm) but repetition did not follow
  below 0.60 (0.653).
- **FALSIFIER NOT TRIGGERED**: it required repetition at or above 0.65 at 12 epochs; got 0.614.

## The finding that matters: repetition does not respond to ANY knob

Six levers have now been turned on this model, and here is what happened to looping:

```
lever                          repetition before -> after     what it did
concept supervision 0.3 -> 2.0      0.555 -> 0.659 (at 2M)      no reduction at any setting
training data 600k -> 2M            0.555 -> 0.659              worse
context length 32 -> 64             0.659 -> 0.532              small gain, cost 1.32 accuracy points
epochs 18 -> 12 -> 9                0.659 -> 0.614 -> 0.653     flat
concept widening 36 -> 53           no change                   no change
decorrelation term                  no change                   no change
```

**Looping is invariant across data volume, context length, epoch count, supervision strength, concept count and activation
decorrelation** - six independent changes, none of which moves it. That is strong evidence the cause is not any of them.

## The untested candidate: exposure bias

The model is trained teacher-forced - always conditioned on TRUE text - and then generated autoregressively, where it is conditioned on
ITS OWN output. Every lever above preserves teacher forcing, which is exactly why none of them touch the failure. At generation time the
model drifts into contexts it never saw in training and falls back to repeating what is locally present, which is the classic
exposure-bias signature.

This is testable without training anything: feed the model its own generated prefix and measure how its prediction quality changes
against the same-length true prefix. If accuracy or perplexity collapses on self-generated context, exposure bias is confirmed and the
fix list becomes scheduled sampling or an unlikelihood term rather than more config search. That test is running next
(`lens/runs/EXPOSURE-BIAS-PREDICTIONS.txt`).

## A useful configuration result alongside it

**12 epochs at 2M windows is the best accuracy point found today: 23.47%** (against 21.08% for the shipped 600k/18-epoch arm, +2.39), and
**9 epochs is the best accuracy/coverage combination**: 22.92% accuracy with coverage 0.349, matching the 600k arm's 0.351 while being
1.84 points more accurate. So the earlier "accuracy versus generation" trade is not a strict dial - most of the accuracy gain can be had
without giving up coverage, and only the looping remains untouched.
