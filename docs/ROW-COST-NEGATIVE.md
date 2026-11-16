
## The meaning penalty is retired by a 60-second measurement: pushing own-word mass costs far more than it buys (2026-11-16)

Predictions in `lens/runs/ROW-COST-PREDICTIONS.txt`, written before this ran. The fix arm queued an hour ago costs ~70 GPU-minutes, so
its central question was asked directly first: **if a concept's answer row is forced to prefer its own words, how much prediction does it
lose?** That is an experiment on the answer matrix alone - the trained rows, the activations the model itself produces, no retraining - so
it isolates the term's cost from every other moving part.

## Method

Optimise **only** the 53x12,002 answer matrix with `L = mean_k[logsumexp(W[k,:]) - logsumexp(W[k,own_k])]` at three weights, 400 steps;
measures own-word mass and next-token accuracy on **20,000 held-out windows the optimisation never saw**.

## Result

```
baseline: own-word mass 0.0014 | held-out next-token accuracy 0.1565

weight   own-word mass   held-out acc   cost (points)
2.0         0.0014          0.1559          0.06
8.0         0.0016          0.1537          0.28
30.0        0.0041          0.1266          2.99
```

- **P1 FAILED**: predicted own-word mass could reach above 0.10 for under 3 points; it reaches **0.0041** at that price.
- **P2 PASSED**: the cost is monotone in the weight.
- **P3 FAILED**: predicted 0.05 for under 1 point; 0.05 was never reached at any weight.
- **FALSIFIER FIRED**, in the exact form it was written: own-word mass cannot exceed 0.02 without losing more than 5 points - in fact a
  weight of 30 buys **0.0041 for 2.99 points**. **The penalty approach is retired, and the retraction is of the fix, not of the
  measurement** that motivated it.

## Why it fails, in plain terms

The answer row is being asked to do two jobs that pull against each other. Its whole training signal says "produce the distribution of
the next token", which needs the mass spread across thousands of plausible continuations. This term says "concentrate the mass on the
concept's own handful of words". **The second is nearly orthogonal to the first**, so every unit of meaning has to be paid for with
prediction - and at ~700 points of accuracy per unit of own-word mass, it can never be afforded.

## What replaces it

The falsifier named the alternative before the numbers arrived: **make meaning structural instead of penalised**. Rather than asking the
rows to hold their own words by force, **derive the rows from the embeddings of the concepts' own words** - the concept's answer direction
is then a function of its vocabulary by construction, so there is nothing to pay for and nothing to drift. That is an architecture change,
not a penalty, and it must go through the same verify-first gate.

The queued penalty arm is not launched. Nothing is deleted: the term, its verified gradient and this curve stay in the tree as the record
of why the cheap-looking fix was the wrong one.
