# More data improves prediction and DEGRADES generation: the objective is misaligned (2026-11-15)

Predictions in `lens/runs/MORE-DATA-PREDICTIONS.txt`. One change against the shipped arm: training windows 600,000 -> 2,000,000 (3.3x,
19.2M -> 64.0M tokens), same recipe, same 18 epochs, same seed, same split. The trainer's own under-trained warning goes from "1.31
(UNDER-TRAINED)" to "4.36 (adequate)".

## Result: a large accuracy win and a generation regression in the same arm

```
arm                         tokens/param   final VAL   final ppl   greedy repetition   greedy coverage   greedy KL   distinct tokens
shipped (600k windows)          1.31         21.08%      269.30          0.555              0.351          18.11           364
more data (2M windows)          4.36         23.32%      209.74          0.659              0.243          20.86           278

change                                      +2.24 points   -59.56            +0.104            -0.108         +2.75
```

Controlled decoder (top-k 50 + rep 1.3 + no-repeat-3gram + end banned 8), same 24 prompts:

```
more data : repetition 0.000   coverage 0.212   KL 20.545   stop_rate 0.000
shipped   : repetition 0.000   coverage 0.229   KL 19.990   stop_rate 0.250
```

**So 3.3x more data bought the largest accuracy gain of the day (+2.24 points, perplexity 269 -> 210) and simultaneously made the
generated text worse on every quality measure**: more looping (0.555 -> 0.659), fewer realistic tokens (0.351 -> 0.243), further from
the real distribution (KL 18.11 -> 20.86), and a narrower argmax set (364 -> 278).

## Prediction scorecard

- **M1 FAILED**: predicted coverage above 0.45; got **0.243**.
- **M2 FAILED**: predicted repetition below 0.35; got **0.659**.
- **M3 PASSED**: predicted final VAL above 21.50%; got **23.32%**, far above.
- **M4 FAILED**: predicted KL below 17.0; got **20.86**.
- **FALSIFIER NOT TRIGGERED**: it required coverage at or below 0.40 **and** VAL at or below 21.30%. Coverage was 0.243 (yes) but
  accuracy was 23.32% (no), so the falsifier's conclusion - "more data buys nothing" - is not supported. More data buys a great deal;
  it buys the wrong thing.

## What this establishes, and why it matters more than the numbers

**The metric this project has been optimising is not the metric the product needs.** Next-token accuracy on assistant turns rose 2.24
points while every measure of generated-text quality fell. A model can get better at predicting the corpus and worse at producing text
from it, and this arm is a clean demonstration at fixed seed, fixed recipe, one changed variable.

That reframes the last four hours: five architectural explanations of "narrow vocabulary" were probed and rejected
(`docs/history/GENERATION-MEASURED.md` summarises them), and the actual problem is that **validation accuracy and generation quality
point in opposite directions on this recipe**. Any further arm that reports only VAL accuracy can now be assumed to be optimising the
wrong thing until it also reports repetition, coverage and KL on a stated prompt count.

## Next

The dissociation needs a cause, and the cheapest one to test is **context length**: the model sees 32 tokens and, being more accurate
yet more loopy with more data, is plausibly leaning on a locally-repetitive solution. Window 64 is a config-only change and is running
next (`lens/runs/WINDOW64-PREDICTIONS.txt`).
