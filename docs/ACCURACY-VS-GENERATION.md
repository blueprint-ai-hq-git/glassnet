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

## Context length 64: generation improves a little, accuracy falls, all four predictions miss

Predictions in `lens/runs/WINDOW64-PREDICTIONS.txt`. One change against the more-data arm: window 32 -> 64 (2M windows, lam 1.0,
18 epochs, seed 0, same split).

```
arm                              tokens/param   final VAL   final ppl   greedy repetition   greedy coverage   greedy KL   controlled coverage
2M windows, window 32                 4.36        23.32%      209.74          0.659              0.243          20.86            0.212
2M windows, window 64                 7.69        22.00%      241.26          0.532              0.290          19.42            0.208
```

- **W1 FAILED**: predicted repetition below 0.40; got **0.532**.
- **W2 FAILED**: predicted coverage above 0.35; got **0.290**.
- **W3 FAILED**: predicted validation at or above 23.00%; got **22.00%**.
- **W4 FAILED**: predicted KL below 19.0; got **19.417**.
- **FALSIFIER NOT TRIGGERED, but only just**: it required repetition at or above 0.55 **and** coverage at or below 0.28. Repetition came
  in at 0.532 (below the bar) and coverage at 0.290 (above it). Both missed by tiny margins, so per the rule I wrote I am reporting the
  numbers rather than claiming either conclusion.

**Directionally, longer context helps generation and costs accuracy**: repetition 0.659 -> 0.532, coverage 0.243 -> 0.290, KL 20.86 ->
19.42, at a cost of 1.32 accuracy points. That is the same trade the supervision dial made, in the opposite direction - and it means the
dissociation from `docs/history/ACCURACY-VS-GENERATION.md` is not an artefact of one arm.

## A defect found and fixed in the measurement tool, and a false COMPLETE

The generation tool hardcoded a 32-token window, so both measurement calls died on the window-64 checkpoint
(`matmul: size 65536 is different from 32768`) - and the runner still printed `WINDOW64 COMPLETE`. That is the false-completion defect
this project has already hit twice. Both were fixed:

1. `tools/chat_generate_compare.py` now derives the context length from the checkpoint itself
   (`concept.W.shape[0] // emb.shape[1]`) and the run prints it, so a window mismatch cannot recur silently.
2. `lens/scripts/run_window64.sh` now checks each measurement's exit code via `PIPESTATUS` and prints `MEASUREMENT FAILED` instead of
   `COMPLETE` when either fails.

The numbers above were produced after both fixes, with the tool reporting "context length from the checkpoint: 64 tokens".

## Where the three arms leave the trade-off

```
arm                                VAL accuracy   greedy repetition   greedy coverage
600k windows, window 32, lam 1.0      21.08%           0.555              0.351
2M windows,   window 32, lam 1.0      23.32%           0.659              0.243
2M windows,   window 64, lam 1.0      22.00%           0.532              0.290
```

No arm is good on both axes, and the best accuracy arm is the worst on generation. The consistent reading across all three: **this recipe
buys accuracy with confidence, and confidence is what makes the text loop.** The next lever must therefore attack confidence directly,
not the architecture - a training-time signal against repeated tokens (unlikelihood), which is the candidate the window-64 falsifier
named.
