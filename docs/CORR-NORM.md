
## Priced correction: the price works, it is not enough, and its falsifier fired at share 0.9968 (2026-11-16)

Predictions in `lens/runs/CORR-NORM-3-PREDICTIONS.txt`, written before the run (third attempt; the first two diverged to NaN at weights 30
and 1.0, and the weight here was **measured**, not guessed). Numbers from `lens/runs/CORR-NORM.txt`.

```
final VAL accuracy        23.66%     (unpriced structural arm: 24.33%, shipped 4M: 24.58%)
final VAL perplexity     172.06
concepts passing          53/53
median correction share   0.9968     (N2 predicted 0.50-0.99)        -> FAILED, narrowly
median ||A_c B||         27.1553     (unpriced at its end: 87.42)    -> the price cut growth 3.2x
median ||structural||     1.5466
median own-word mass      0.0013     (N3 predicted above 0.002)      -> FAILED
explain() max |residual| 1.697e-06   | concept share 1.000000         -> PASSED
no NaN in any epoch                                                -> N1 PASSED
```

## What the price bought, and what it did not

**It bought real restraint.** The correction's median norm ended at **27.16** instead of **87.42** - a **3.2x** reduction in growth for a cost
of **0.67 accuracy points** (24.33 -> 23.66). At the measured weight (about 11% of the cross-entropy gradient) the term is stable and does
exactly what it prices.

**It did not buy meaning, and the falsifier fired in its written form**: the correction still carries **0.9968** of every row's energy, above
the 0.99 bar I set in advance. Own-word mass is 0.0013 - barely above the free matrix's 0.0008, and *below* the 0.002 that N3 required.

## The sequence this completes

```
arm                                  correction norm   correction share   own-word mass   VAL
unpriced                             87.42             0.9999             0.0008          24.33%
priced at 11% of the CE gradient     27.16             0.9968             0.0013          23.66%
priced + structural scale 10         (running)                            (running)       22.9% at ep6
```

Each intervention moves its own quantity and none of them moves meaning. **Pricing reduces the correction's size by 3.2x and the meaning
share by 0.003.** That is the third independent confirmation that the share cannot be bought down by acting on the correction, because the
structural term's own mass is what is missing, not the correction's excess.

## What the falsifier demands

The falsifier said: *share above 0.99 at 11% of the CE gradient means a soft price cannot restrain an unbounded low-rank correction at any
stable weight; the answer is a HARD per-row normalisation fixing the structural share.* That is now the next construction: renormalise the
correction after each optimiser step so that `||A_c B||` cannot exceed a fixed multiple of `||structural_c||`. It is a projection, not a
penalty, so there is no weight to fail at.
