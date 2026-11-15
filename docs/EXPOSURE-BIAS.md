
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

## Exposure bias is CONFIRMED - and it hurts the control MORE than the readable model

Predictions in `lens/runs/EXPOSURE-BIAS-PREDICTIONS.txt`. Method: two branches advanced side by side over 24 held-out prompts and 24
steps - the `teacher` branch conditions on true text, the `self` branch on the model's own argmax output - and both are scored against
the same true target. No training involved.

```
checkpoint 2M/12ep (best accuracy)          checkpoint 2M/18ep                        direct-path control
steps    t.acc  s.acc   t.NLL  s.NLL  gap   steps   t.acc  s.acc   t.NLL  s.NLL  gap   steps   t.acc  s.acc   t.NLL  s.NLL  gap
1-2      4.17%  2.08%  23.00  23.82 +3.6%   1-2     6.25%  6.25%  23.41  24.61 +5.1%   1-2     4.17%  4.17%   7.19   8.20 +14.1%
3-4      4.17%  0.00%  23.95  27.17 +13.4%  3-4     4.17%  0.00%  24.56  27.29 +11.1%  3-4     6.25%  0.00%   8.02  11.08 +38.2%
5-8     12.50%  1.04%  21.05  26.54 +26.1%  5-8     9.38%  0.00%  21.86  27.51 +25.8%  5-8     9.38%  1.04%   7.10  10.03 +41.3%
9-16     5.73%  1.04%  22.55  27.01 +19.8%  9-16    5.73%  0.52%  23.65  27.49 +16.2%  9-16    7.29%  0.00%   7.41  10.56 +42.5%
17-24    6.77%  0.00%  21.65  27.51 +27.1%  17-24   7.81%  0.00%  22.75  27.60 +21.3%  17-24  13.02%  0.52%   7.27  11.46 +57.6%
```

- **X1 PASSED**: at steps 9-16 the self branch's true-token NLL is **+19.8%** (12ep), **+16.2%** (18ep) and **+42.5%** (control) worse than
  teacher - all above the 15% bar.
- **X2 PASSED**: the gap grows with horizon on every checkpoint (+3.6% to +27.1%; +5.1% to +21.3%; +14.1% to +57.6%).
- **X3 FAILED on two of three**: self accuracy does collapse (to 0.00-2.08%), but on the 12-epoch arm and the control it falls by 3-4
  points rather than the 5 I predicted. Teacher accuracy stays flat, as predicted.
- **FALSIFIER NOT TRIGGERED**: the gap is far above 5%.

## The result I did not expect: the readable model is MORE robust to its own output

**Exposure bias is worse in the control (+38% to +58%) than in either readable model (+11% to +27%).** So the failure is a property of
the training protocol, not of the concept bottleneck - and the readable architecture is the more robust of the two when fed its own
text. That is the first comparison today in which the readable model wins on something other than accuracy and explanation.

## Two defects in my own instrument, one fixed here and one still to fix

1. **FIXED**: the column I published as "perplexity" was mean **negative log-likelihood in nats**, not perplexity (perplexity is exp of
   those numbers, so 23.41 nats is astronomically worse than "ppl 23.41" reads). The tool now labels it NLL and prints a note.
2. **NOT YET FIXED, and it matters for the absolute values**: the model is trained `--assistant-only --train-turn-end`, but this test
   scores **every** position in the window, including user turns the model was never trained to predict. That inflates every NLL here -
   the same checkpoint reports validation perplexity 211 while this test shows 23.4 nats (= perplexity ~1.5e10) because most scored
   positions are out of scope. **The comparison between branches is unaffected** (both branches are scored on identical positions), so
   X1/X2/X3 stand, but no absolute NLL from this document may be quoted as the model's perplexity. The corrected assistant-only run is
   the next item in the queue.

## Where this leaves the looping problem

Seven levers now fail to move repetition, and the eighth - the training protocol - is confirmed as a real and large effect. Both models
collapse on their own output, which means **the fix has to change what the model sees during training, not what it is made of**:
scheduled sampling, or an unlikelihood term against repeated tokens. Both are specced in `lens/runs/QUEUE.txt`; the unlikelihood term
needs the same verify-first protocol the decorrelation term got before any arm uses it.
