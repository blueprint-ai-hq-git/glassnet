
## Structural answer rows: accuracy survives (24.33% vs 24.58%) and meaning does not arrive (own-word mass 0.0008) (2026-11-16)

Predictions in `lens/runs/STRUCTURAL-ARM-PREDICTIONS.txt`, written before the run. Numbers from `lens/runs/STRUCTURAL-ANSWER.txt`,
checkpoint `nodirect1024_structural_ac_data4m_e12_s0_best.npz`. One change from the shipped arm: the answer row is
`row_c = (mean E over concept c's own words) @ E^T + A_c B` with a zero-initialised rank-32 correction, so `answer.W` is not a parameter.

```
final VAL accuracy        24.33%   (shipped 4M: 24.58%, so -0.25)
final VAL perplexity     167.38    (shipped 181.04, better)
tokens per parameter       8.87     (adequate)
concepts passing          53/53
explain() max |residual| 1.927e-06  | concept share 1.000000 / 1.000000
trained-label AUROC       0.9991    | held-out synonym 0.6423
own-word mass (R1)        0.0008    | 0 of 53 concepts at or above 0.05
```

## Predictions scored

- **R1 FAILED** - the one that mattered. Own-word mass is **0.0008**, identical to the free answer matrix, against a 0.10 bar.
- **R2 PASSED** - final VAL 24.33% is well above the 20.58% bar: the structural architecture costs **0.25 points**, not four.
- **R3 PASSED** - the explanation is still exact to 1.9e-06 with concept share 1.000000.
- **R4 PASSED** - input-side honesty unchanged (0.6423 held-out), as predicted, because the term touches no activation.
- **Falsifier NOT triggered** (VAL is far above 18%).

## What this settles, and what it does not

**Settled: the architecture is not the obstacle.** Replacing the free answer matrix with rows built from the concepts' own word
embeddings costs a quarter of a point of accuracy. If meaning were going to be cheap anywhere, it would be here.

**Not settled, and now measured twice: the correction eats it.** The row's own-vocabulary mass stays at 0.0008 because the learned
correction `A_c B` grows to dominate - median norm **87.42** against the structural term's **1.06** at epoch 6, carrying **0.9999** of
every row's energy, up from 46.89 at epoch 1. A zero-initialised low-rank correction is an **escape hatch**: nothing stops it growing
until it supplies the entire answer.

**So the remaining question is exactly one: can the correction be restrained without destroying accuracy?** Two attempts to restrain it by
pricing it diverged to NaN (weights 30 and 1.0, both far too large; the measured stable weight is ~2e-4), and a third is running. If a soft
price cannot do it at any stable weight, the next step is a hard per-row normalisation, and if that cannot either, the honest conclusion is
that at 16M parameters a low-rank free correction is load-bearing and meaning cannot be forced into the rows.
