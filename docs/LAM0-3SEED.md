
## Confirmed on three seeds: no concept supervision reaches 27.01% +/- 0.03, +2.43 over the shipped model (2026-11-16)

Predictions written before the run: `lens/runs/LAM0-3SEED-PREDICTIONS.txt`. Numbers from `lens/runs/LAM0-3SEED.txt`.

```
seed   VAL acc   VAL ppl   concept loss
 0      26.98%    147.81       0.0000
 1      27.03%    148.38       0.0000
 2      27.03%    145.87       0.0000
mean    27.01%    147.35
sd       0.03       1.34
```

## All three predictions passed

- **S1** all three seeds above 26.12 (the full-supervision 3-seed mean): **passed**, worst seed 26.98
- **S2** mean 26.8-27.1 with sd at or below 0.20: **passed**, mean 27.01, sd 0.03
- **S3** perplexity mean at or below 150.0: **passed**, 147.35
- **Falsifier** (any seed at or below 26.12 withdraws the +0.86): **did not fire**

**Paired on seed**, which is this project's rule: against the full-supervision arm the deltas are **+1.01 / +0.79 / +0.87, mean +0.89 sd 0.11, three
of three positive**. Against the shipped configuration they are **+2.41 / +2.50 / +2.39, mean +2.43 sd 0.06**.

The control is verified on every seed by its own instrument: the concept loss printed exactly **0.0000** at all three endpoints, so none of these runs
silently trained the ordinary objective.

## The claim, now publishable

**Removing concept supervision from this recipe is worth +0.89 points of validation accuracy (paired, 3 seeds, 3/3 positive) and 2.43 points against the
shipped configuration.**

That is a cost, not a defect: the concept layer is what makes the model explainable, and it is what produces the vocabulary breadth of the shipped
model (coverage 0.292, the only configuration in this family to reach it). The readable architecture pays for both in accuracy.

## What is still single-seed and still labelled as such

The per-strength points - lam 0.5 at 26.52% and lam 0.25 at 26.69%, and their coverage figures 0.167 and 0.150 - remain one seed each. The monotone
accuracy ordering across all four settings is therefore suggestive, not established; only the two endpoints (no supervision 3 seeds, full supervision 3
seeds) carry a three-seed bar. Stated here so the tidy four-point curve is not mistaken for a measured one.

## Nothing in this family ships

No arm here beats the shipped model on both halves. Coverage remains the binding constraint: 0.190 for this arm against 0.292 shipped, and the
promotion bar of 0.27 was fixed before any of these runs. The shipped model stays the shipped model.
