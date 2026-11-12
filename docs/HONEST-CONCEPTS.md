# The 12 honest concepts: a cleaner concept set fixes the names but not the honesty (2026-11-12)

The concept set was rebuilt to contain only concepts a TRAIN-split probe could support (12 of 36, no leakage into the
held-out split). Result:

```
concept set                  accuracy   balanced honesty   ablation   readability price
36 concepts (incl. words)      18.07%        67.1%           20.27%          2.20 pp
12 honest concepts             16.65%        71.8%           17.40%          0.75 pp
trivial baseline                5.42%          -               -               -
```

**Both predictions failed.** I predicted honesty **>= 90%** and accuracy **8-14%**. Honesty came in at **71.8%**, and
accuracy at **16.65%** - above my range, which means the honest set costs less accuracy than I feared, only 1.42 points
against the 36-concept model.

## The important part: honesty is not limited by the concept set
Those 12 concepts probed at **96-100%** balanced accuracy on training windows. In the trained model they average
**71.8%**, with the worst at **49.5%** - dead. The concepts cannot be the problem any more: they are the ones the input
supports best. What is left is the **training objective**: the concept loss is one term beside a token cross-entropy
several times larger, and the embedding table keeps moving underneath the concept weights, so a neuron that becomes
honest early drifts away as the token objective reshapes the embedding.

This is the same mechanism already measured once on this project from the other direction: freezing a pathway that reads
the shared embedding table protects nothing, because the table is what keeps moving.

## Prediction for the arm now running
Same 12 concepts, concept loss at **5x**. Prediction: balanced honesty **at or above 90%**, accuracy **14-16.5%** (at most
2.7 points below the 16.65% measured at 1x). If honesty stays under 80% even at 5x, the objective is confirmed as the
binding constraint and the next arm must train the concept layer to honesty FIRST and then hold it fixed while the rules
learn - which also means the token objective has to give up some freedom over the embedding.

## What is established either way
- A readable middle layer with 96 rules, 2 connections per rule, costs **0.75 points** of held-out accuracy at the best
  concept set, and it beats the trivial baseline by 11.23 points.
- The gate item "does each concept neuron fire on its named concept" now has a complete, reproducible answer for every
  concept in the model, produced by `tools/concept_table.py`, including the cross-firing check that catches neurons
  detecting the wrong thing (15 of 36 in the full set).
