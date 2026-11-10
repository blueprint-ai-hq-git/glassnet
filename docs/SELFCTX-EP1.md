# Self-context, epoch 1: repetition down sharply, the copy curve barely moved (2026-11-10)

Interim, one seed, epoch 1 of 3, compared against the baseline's epoch 1 so both sides are equally trained.

Prediction on record (`FEEDBACK-LOOP-MECHANISM.md`): the copy rate at generation steps 9-24 would fall below the
90-97% measured on the previous models, repetition runs would shorten, at a small accuracy cost.

```
                       baseline ep1   self-context ep1
assistant accuracy        19.69%          14.74%        (-4.95 pp, larger than "small")
epoch wall clock           205 s           377 s         (1.84x)
symbol share in output     52.4%           24.3%        (-28.1 pp)
longest symbol run         16               7           (shorter)
distinct 4-grams           63.9%           85.7%        (+21.8 pp)
```

Step-wise copy rate:
```
                    st 1-4  st 5-8  st 9-12  st 13-16  st 17-20  st 21-24   mean st 9-24
baseline ep1         34.4%   68.8%    87.5%     93.8%     93.8%     90.6%        91.4%
self-context ep1     34.4%   84.4%    81.2%     87.5%     84.4%     96.9%        87.5%
```

## Verdict on the prediction: partially confirmed, and not enough
- **Confirmed on text quality:** repeated 4-grams rise from 63.9% to 85.7%, symbol share falls from 52.4% to 24.3%,
  and the longest symbol run halves from 16 to 7. The generated text is measurably less degenerate.
- **Barely confirmed on the mechanism:** the mean copy rate over steps 9-24 moves from 91.4% to 87.5%, about 4 pp,
  where the finding it was aimed at predicted a clear fall below 90%. It also *rises* at steps 5-8 (68.8% to 84.4%)
  and at steps 21-24 (90.6% to 96.9%). The curve is not being reshaped, it is being nudged.
- **Cost is larger than expected:** 4.95 pp of accuracy, against a prediction of "a small cost", and 1.84x the
  training time per epoch.

## The output, honestly
```
baseline ep1      hello , how are you ?  -> the function of the function of the function , the ` function ` function ` `
self-context ep1  hello , how are you ?  -> the , and the and the and the the , and the the and the the
```
Less punctuation, far more varied tokens, and still not an answer to the question. It has traded a backtick loop for
a comma loop.

## What this means for the next step
Exposure-bias training attacks the right object - it is the only lever today that improved the TEXT rather than a
proxy - but at this dose it does not reshape the copy curve. Two options, and they are the first genuinely
architecture-shaped choices of the day:
1. stronger exposure bias: all steps self-context (`--self-context-frac 1.0`) with the same 8 rounds, which is
   9x the forward cost and would give a clear answer on whether the dose or the mechanism is the problem;
2. or pair self-context with the copy-pressure penalty, so the text-level improvement and the mass-level pressure act
   at the same time.

Both are worth running; the second is a bigger change and breaks "one thing at a time", so the dose test comes first.
