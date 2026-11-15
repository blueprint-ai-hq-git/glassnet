
## Withdrawal: my per-token NLL is not a sound measurement (falsifier fired)

Predictions in `lens/runs/NLL-RECONCILE-PREDICTIONS.txt`. The reconciliation was pre-registered with a falsifier: *if the two numbers
still disagree by more than 2x and no difference is found, the per-token NLL is withdrawn entirely as a measurement, and the
exposure-bias conclusion must be re-derived from accuracy alone.*

```
checkpoint                      instrument                accuracy   mean NLL
2M/12ep (assistant-only)        my exposure-bias tool      6.25%      20.38-22.07 nats
2M/12ep (trainer's own VAL)     train.py                   23.47%      5.35 nats (= ppl 211.49)
direct control (assistant-only) my exposure-bias tool      2.08-8.33%  7.32-8.13 nats
direct control (trainer's VAL)  train.py                   17.73%      10.85 nats (= ppl 54,751)
```

- **R1 PASSED**: assistant-only masking changes mean NLL by only **-5.7%** (21.27 vs 22.55 at steps 9-16), so masking was not the
  explanation.
- **R2 FAILED**: the two populations do **not** agree once matched - my tool reads **21 nats where the trainer reads 5.35** for the same
  checkpoint (4x), and **6.25% accuracy where the trainer reads 23.47%** (also a large gap).
- **R3 NOT SATISFIED**: I have not named a concrete cause. Candidate differences (per-batch versus per-token averaging, which positions the
  trainer scores, how it builds windows) remain candidates, not findings.
- **FALSIFIER FIRED**: disagreement is 4x, above the 2x threshold, with no cause identified.

**Therefore the per-token NLL from `tools/exposure_bias_test.py` and `tools/exposure_bias_assistant_only.py` is WITHDRAWN as a
measurement.** No NLL from those tools may be quoted, and the quantitative claim published earlier - that the control suffers exposure
bias more severely (+38-58%) than the readable model (+11-27%) - **is withdrawn with it**, because that comparison was NLL-based.

## What survives, re-derived from the instrument that does not have this problem

Accuracy is computed identically for both branches inside the same run, so the *relative* comparison is unaffected by whatever makes the
absolute values disagree. From the assistant-only run:

```
checkpoint     teacher accuracy by bucket            self accuracy by bucket
2M/12ep        6.25 / 6.25 / 6.25 / 6.25 / 6.25%      8.33 / 0.00 / 0.00 / 1.04 / 2.08%
control        8.33 / 2.08 / 5.21 / 7.81 / 5.73%      2.08 / 0.00 / 1.04 / 2.08 / 0.00%
```

**Self-conditioned accuracy collapses to 0-2% in both models while teacher-forced accuracy stays flat or rises.** That is exposure bias
confirmed on a same-instrument comparison, and it does not depend on the withdrawn NLL. What is NOT established is which architecture
suffers more - that was the NLL-based claim and it is gone.

## The honest state of the looping question

Seven levers fail to move repetition; the eighth candidate (the training protocol) is supported by an accuracy-based collapse on
self-conditioned input, but **the size of the effect is unmeasured** until the two instruments are reconciled. The next action is
therefore not another model arm: it is to reconcile the two instruments against each other, because every future claim about generation
quality depends on which one is right. That work is `lens/runs/INSTRUMENT-RECONCILE-PREDICTIONS.txt` in the queue.

This is the fourth correction I have made against my own work today, and the second where the fix was to withdraw a number rather than
add one.
