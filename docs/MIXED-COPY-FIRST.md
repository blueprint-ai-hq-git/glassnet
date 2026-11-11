# A model that talks AND copies: 29.82% language, 85.7% taught copying, 4.2% held-out (2026-11-11)

First arm that trains both abilities at once: 50% of batches from the real chat corpus, 50% from the synthetic copy
task, the readable pointer instead of the old `dmix` echo path, d512, 8,000 steps.

```
step    language (matched harness)   copy, taught words   copy, held-out words
6000            29.30%                     82.5%                  2.7%
7000            29.23%                     84.3%                  3.7%
8000            29.82%                     85.7%                  4.2%
```

## Two rulers, stated so nobody mixes them
- **language 29.82%** is plain next-token accuracy on matched validation windows, ALL positions. On that same ruler
  this model's starting point was **24.27%**, so the arm is worth **+5.55 pp**.
- the project's headline **25.52%** is assistant-turn, by-position accuracy on a different model. It is NOT the same
  ruler and 29.82 is not "better than 25.52" on the project's own scale. Both are labelled wherever they appear.

## What it proves and what it does not
- PROVES: one model can hold a corpus-trained language ability (29.82%) and copy a taught word out of its own window
  (85.7%) at the same time, with the pointer as its only copy route.
- DOES NOT PROVE: general copying. Held-out words are still only 4.2%, against 58.0% for a model trained on copying
  alone. Mixing in the corpus costs the copy operation most of its generality.

## Why held-out copying falls from 58.0% to 4.2%
The corpus batches teach the network to predict chat text. That objective rewards the pathways that generalise over
common words, and 50% of the steps are spent there. The copy operation needs many distinct examples to learn an
operation rather than a word list, and here it gets 4,000 copy batches instead of the pointer-only arm's 6,000.

The held-out curve was still RISING at every checkpoint (2.7 -> 3.7 -> 4.2%), so the honest read is "under-trained",
not "impossible" - which is why the next arm runs three times longer rather than changing the design.

## Predictions on record for the longer run
Held-out copying 20% or better, language at or above 25%. If held-out plateaus below 10% while taught words sit above
85%, then the mixture itself is the limit and the design needs the pointer to be trained harder than the corpus (for
example a warm-up phase on copying before the mixture).
