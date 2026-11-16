
## Meaning is free if it is structural and prediction is dead if it is not: the three-measurement bracket (2026-11-16)

Predictions in `lens/runs/STRUCTURAL-ROWS-PREDICTIONS.txt` and `lens/runs/BLEND-PREDICTIONS.txt`, written before each ran. All three
measurements use the shipped 4M/12ep checkpoint, the model's own activations, and **20,000 held-out windows**.

## 1. The penalty (already retired): meaning is unaffordable if paid for

```
weight 30 on the output-content term: own-word mass 0.0014 -> 0.0041 for 2.99 accuracy points  (~700 points per unit of meaning)
```

## 2. Pure structural rows: meaning is FREE, and prediction collapses

Row built only from the concept's own words: `row_c = mean(E[own_c]) @ E^T`, rescaled to the trained row's norm so only the direction is
compared.

```
rows                       median own-word mass   held-out accuracy
trained (baseline)              0.0014               0.1565
structural from own words       0.9200               0.0073        (-14.92 points)
```

**S1 PASSED** (0.9200, against a 0.10 bar - the penalty could never buy anything like this at any affordable price). **S2 PASSED** (0.0073,
below the 0.08 bar) and below the 0.03 that its falsifier set: **pure tying is dead.** The row's direction is now exactly its vocabulary's
geometry, and the model's prediction dies with it.

## 3. Post-hoc blend: there is no useful middle

```
a      median own mass   held-out acc   cost (points)
0.0        0.0014           0.1565          0.00
0.1        0.0030           0.1500          0.65
0.2        0.0079           0.1326          2.39
0.3        0.0189           0.1009          5.56
0.5        0.1082           0.0309         12.55
0.7        0.4307           0.0129         14.36
1.0        0.9200           0.0073         14.92
```

- **B1 FAILED**: at a = 0.3 the mass is 0.0189 (bar: above 0.10) and accuracy 0.1009 (bar: above 0.14).
- **B2 PASSED**: a = 0.5 costs 12.55 points.
- **Falsifier NOT triggered**: a = 0.1 costs only 0.65 points, so the blend is nearly free at the small end - it simply buys almost no
  meaning there.

## What the three results say together

1. **Paying for meaning with a penalty costs ~700 accuracy points per unit** - it can never be afforded.
2. **Getting meaning for free from the embedding geometry costs the whole output path** - 0.92 own-word mass, 0.7% accuracy.
3. **Blending the two after training does not rescue it** - the useful-meaning region (mass above 0.10) begins where accuracy has already
   been destroyed (a = 0.5, 3.1% accuracy).

**Therefore the rows cannot be made meaningful after training, by any means.** The direction "built from this concept's own words" is
almost orthogonal to the direction "predicts the next token" **in the geometry this model happened to learn**. That geometry was never
under pressure to make the two agree, which is the same structural fact the look-inside measurement found, seen from the other side.

**The only remaining option is to train the constraint in from the start**: build the answer row as a structural term from the concept's
own word embeddings **plus a learned low-rank correction**, and let the embeddings reorganise under that pressure so that the structural
direction becomes predictive. Then meaning is not something paid for or blended in - it is a property of the trained geometry.

That is the next arm, and the three curves above are why it is the next arm rather than another penalty or blend attempt.
