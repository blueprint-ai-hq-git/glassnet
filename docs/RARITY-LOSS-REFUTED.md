

---

# RARITY-WEIGHTED LOSS: REFUTED BY ITS OWN FALSIFIER (2026-11-19 11:00)

The first attempt at "our own training method" that follows from a measured failure. The measured failure: 11,002 of 12,002
vocabulary entries sit beyond rank 1,000 and occupy 14.41% of all training positions, while the rare-token bands scored **0.00%**.
So the loss reweighted every scored position by inverse frequency,
`w = (median_count / (count + 1)) ** alpha`, clipped to [0.05, 50], normalised to a weighted mean.

The predictions were written down before the run, including the falsifier:

- **P1** weighted arms gain >= 3 points on the tail band and on ranks 501-2000 - **FAILED**
- **P2** they lose <= 3 points on top-20 - **MET** (-2.22)
- **P3** overall falls <= 2 points - **MET** (-1.63)
- **FALSIFIER** "if the tail band does not improve (<= +1 point), the weighting does not actually change rare-token behaviour and
  the method is dead" - **FIRED**

```
band                 alpha=0 (plain)   alpha=1 (rarity)   change
top-20                    32.74%            30.52%        -2.22
ranks 21-500               9.67%             8.26%        -1.41
ranks 501-2000             0.00%             0.00%         0.00   <- the band it was aimed at
TAIL (rank > 2000)         0.00%             0.00%         0.00   <- the band it was aimed at
overall                   20.07%            18.44%        -1.63

held-out (trainer line)   22.91%            21.08%        -1.83   (loss 9.3447 vs 9.6752)
```

**Rarity weighting made every band worse, including both bands it was designed to help, which stayed at exactly zero.**

## What this means

The weighting works on the loss exactly as specified - the gradient check passed at alpha = 0, 1 and 2 (worst 5.9e-10) and was
proven inert at alpha = 0. It simply does not change what the model can represent: at this budget the model has **no** rare-token
ability to reallocate toward, so redistributing loss weight only distorts the frequent band it already handles. The binding
constraint is representational capacity and/or training budget, not how the loss is weighted.

**So the training-method idea is dead, and the falsifier says so before I could rationalise it.** The next method must attack
capacity or the data, not the loss weighting - and the parameter audit already points at where the capacity went (62.3% into an
answer table, 1.3% into computation).

Honest scope: 2,500 steps, 4.7M parameters, one seed. A single seed cannot rule out a small effect, but it can and does rule out
the >= 3-point gain that was predicted, and the bands it targeted did not move at all.
