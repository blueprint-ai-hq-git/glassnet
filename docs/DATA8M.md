
## 8M windows: +1.07 points of accuracy, better perplexity, and the honesty gap does not narrow (2026-11-16)

Predictions in `lens/runs/DATA8M-PREDICTIONS.txt`, written before the run. One change from the shipped 4M arm: twice the training windows.
Numbers from `lens/runs/DATA8M.txt`, checkpoint `nodirect1024_wide60_lam1.0_data8m_e12_s0_best.npz`.

```
                                     4M (shipped)      8M            change
final VAL accuracy                     24.58%         25.65%        +1.07
final VAL perplexity                  181.04         157.89         better
tokens per parameter                    8.72          17.44         (well trained)
concepts passing                       52/53          52/53          same
greedy repetition                       0.275          0.208         better
greedy coverage                         0.343          0.356         +0.013
greedy KL                              17.21          16.79          better
T=0.7 repetition                        0.027          0.053         worse (within noise)
T=0.7 coverage                          0.292          0.271         -0.021
trained-label AUROC                     0.9990         0.9996         same (saturated)
held-out synonym AUROC                  0.6398         0.6336         same
explain() max |residual|               2.9e-06        2.4e-06        exact
concept share                          1.000000       1.000000       exact
```

- **D1 PASSED**: accuracy above 24.58%, at **25.65%**.
- **D2 FAILED**: T=0.7 coverage fell to 0.271 instead of rising above 0.292.
- **D3 FAILED, narrowly**: T=0.7 repetition 0.053 against a 0.05 bar. Under the project's own noise rule a 0.026 difference is uninterpretable, so
  this is not evidence that more data made looping worse - only that it did not make it better.
- **D4 PASSED, and this was the prediction I expected to fail**: the meaning gap **did not narrow**. Held-out synonym transfer is
  **0.6336** against trained-label **0.9996** - a gap of **0.366**, essentially unchanged from 4M's 0.359.
- **Falsifier NOT triggered** (accuracy rose, so the data lever has not flattened).

## What this means

**Twice the data buys about one point of accuracy and better perplexity, and buys nothing for meaning.** Held-out concept honesty moved from
0.6398 to 0.6336 across a doubling of training text - that is noise, not progress. This is consistent with the mechanism already measured:
the answer rows are diffuse directions learned from next-token prediction, and more text sharpens the same machinery rather than changing
what it does.

The best model in the project is now the 8M/12ep configuration at **25.65% +/- 0.06-class validation accuracy** (single seed; the 4M figure
that carries a 3-seed error bar of 0.055 is 24.58%), perplexity 157.89, explanation exact to 2.4e-06 with concept share 1.000000, and
concept naming still about 0.63 honest on unseen wording.
