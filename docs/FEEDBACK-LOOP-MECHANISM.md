# The loop is autoregressive feedback, measured by generation step (2026-11-10)

Three explanations were falsified today. This one held.

```
share of emitted tokens that were ALREADY in the window, by generation step
checkpoint                        st 1-4   st 5-8   st 9-12  st 13-16  st 17-20  st 21-24
talk_ext_d512 (conversational)      6.2%    53.1%     96.9%     93.8%     84.4%     78.1%
matched_copy_d512 (merged)         28.1%    78.1%     90.6%     93.8%     87.5%     90.6%
```

PREDICTION before measuring: the copy rate would be low at the first generated tokens and rise with step number,
because early steps see real prompt text while later steps see mostly the model's own output.

**Confirmed.** At the first four generated tokens the copy rate is 6.2% and 28.1%. By step 9-12 it is 96.9% and
90.6%. The model is not searching for a loop - it writes its own tokens into the window, the window fills with them,
and after about eight steps almost anything it can say is something it can already see.

## The three falsifications that led here
1. **Corpus repetition rate** - falsified (`CONV-ARM-FALSIFIED.md`): the conversational corpus repeats itself LESS
   (38.32% of targets in-window against 51.75%) yet its model loops MORE.
2. **Boilerplate density** - falsified (`tools/boilerplate_density.py`): the math/code corpus carries MORE boilerplate,
   not less (3.84% quote/backtick tokens against 1.30%, longest single-token repeat 3.3 against 1.4), and its model
   loops less.
3. **Memorisation** - falsified (`tools/verbatim_check.py`): every 16-token window of the training corpus was hashed
   (38,829,567 hashes over 38.8M tokens) and **0.0%** of the generated 16-gram windows appear verbatim in it. The
   looping text is generated, not recalled.

That last one matters beyond this bug: the model is not replaying memorised sentences when it loops. Its failure is a
decoding failure, not a memory failure.

## Why every data-side fix failed
The symbol mask, the copy-mass penalty, the conversational-only arm - all of them changed the data or the objective,
and none could touch a problem that only appears after the model has been talking to itself for eight steps. The
measured quantity that predicts the loop is *how much of its own output is in the window*, which is a property of the
decoding loop, not of the corpus.

## The next arm, and the prediction
The trainer already has the machine for this: `--self-context-rounds` / `--self-context-frac`, which train on windows
whose last tokens are the model's OWN greedy predictions (exposure bias). Prediction, recorded before running it: a
self-context arm will show a lower copy rate at steps 9-24 than these models, and shorter repetition runs, at a small
cost in validation accuracy - because it will have been trained on the situation it actually meets at generation time
rather than only on clean human text.

If the step-wise copy curve does not fall under self-context training, the feedback explanation is wrong too, and the
next place to look is the copy path's response to its own embeddings rather than to the window contents.
