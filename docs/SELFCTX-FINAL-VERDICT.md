# Self-context final verdict: the epoch-1 gain did not survive (2026-11-10)

Full 3 epochs, matched split, one change against the 25.52% baseline.

```
                      baseline        self-context
epoch 1 acc            19.69%           14.74%
epoch 2 acc            24.26%           18.76%
epoch 3 acc            25.52%           19.94%        (-5.58 pp)
epoch time             205 s            351-384 s     (1.8x)

final copy rate by step      st1-4  st5-8  st9-12  st13-16  st17-20  st21-24   mean st9-24
baseline                      25.0%  84.4%   68.8%    87.5%    90.6%    78.1%       81.3%
self-context                  43.8%  78.1%   84.4%    87.5%    90.6%    96.9%       89.9%

final copy_echo        in-window  ==prev  ==mode  window mass
baseline                 74.3%     0.0%   25.0%     30.8%
self-context             80.2%    30.2%   41.7%     32.6%

final generation       symbols  longest run  distinct 4-grams
baseline                 8.7%       3            54.0%
self-context            16.7%       6            76.2%
```

## Verdict
**The prediction is falsified.** I said the copy rate at steps 9-24 would fall below 90%; it rose to a mean of
**89.9%** against the baseline's **81.3%**. Every copy metric is worse: previous-token echo 30.2% against 0.0%, modal
echo 41.7% against 25.0%, window mass 32.6% against 30.8%. Accuracy costs **5.58 pp**.

The only thing that improved is distinct 4-grams (76.2% against 54.0%), and that came with more symbols and longer
symbol runs, so it is not a clean win - the model produces more varied tokens including more punctuation soup:

```
baseline      hello , how are you ?  -> by the current of the world of the world of the world of the world of
self-context  hello , how are you ?  -> the . the of the the function of the function of the the number of of
```

At epoch 1 this arm looked like the winner of the day (symbols 24.3% against 52.4%). At epoch 3 it is behind on every
copy measure. **This is the second lever today whose epoch-1 improvement reversed by epoch 3**, and that pattern is
now itself a finding: single-epoch comparisons in this setup are not evidence.

## The second prediction, also falsified
I guessed that confidence sharpening explains why loops return with training - entropy falling, argmax settling on
visible tokens. Measured:
```
checkpoint                      entropy   top1 p    in-window
baseline ep1                      4.54     34.64%     78.5%
baseline ep2                      4.19     34.68%     75.0%
baseline ep3                      5.02     31.78%     70.8%
self-context ep1                  6.81     10.51%     78.5%
self-context ep3                  5.47     27.89%     79.9%
```
The baseline gets LESS confident by epoch 3 (entropy 4.19 -> 5.02) and its in-window rate FALLS (75.0% -> 70.8%). No
sharpening story survives.

## What the day adds up to
Four levers were built and measured against the loop: symbol mask, decode guard, copy-mass penalty, exposure bias.
None reduced the looping. Three explanations were falsified (corpus repetition rate, boilerplate density,
memorisation - 0.0% of generated 16-grams are verbatim training text), and the mechanism that did survive is
autoregressive feedback: 90%+ of emitted tokens are already in the window by step nine.

The one thing that ever fixed the loop was at DECODING time: no-repeat 4-gram blocking took distinct 4-grams from
39.3% to 100.0% on the 6-epoch model (`DECODE-GUARD.md`). It left the text meaningless, but that is a separate
problem - lack of knowledge, not lack of novelty.

So the honest position at the end of today: **the loop is a decoding artefact and has a working fix; the content is a
model-and-data problem that no decoding trick touches.** Further investment should go into content and accuracy, not
into more loop levers, and any future arm should be judged at its final epoch on the fair split.
