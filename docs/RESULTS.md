# Results

Every chart below is drawn from a recorded run. Script: `docs/assets/make_charts.py`.

## Accuracy

![Results](assets/results.png)

| Configuration | Held-out |
|---|---:|
| d=512, full corpus, 6 epochs | 37.37% |
| d=256, full corpus | 35.26% |
| d=256, 2M windows, batch 1024 | 24.20% |
| d=256, 2M windows, batch 4096 | 17.64% |

## Our optimiser vs plain SGD

![Optimiser](assets/optimiser_vs_sgd.png)

23.46% mean vs 19.14% mean over 3 seeds, 20k windows, everything else identical. Lower perplexity too
(117 vs 181).

## Data is the only lever that moved the number

![Data curve](assets/data_curve.png)

Seven other levers were tested and failed: wider models, more passes, attention, tokens-per-example,
concept supervision, multi-token prediction, dropping UNK tokens. Only genuinely more distinct data moved
held-out accuracy.

## Published negative results

![Ablations](assets/ablations.png)

| Attempt | Result |
|---|---|
| Bare single-head attention vs none | **-5.41 points** (18.69% vs 24.10%, 7 runs) |
| Our architecture vs plain shape | **-1.73 points** mean (3.87/4.07 vs 5.53/5.87) |
| Recurrent carry path for arithmetic | unseen operands still 0.0% |

These are published because a project that only shows wins is not measuring anything.
