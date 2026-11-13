# Removing the direct pathway costs ~5 points at every epoch - it earns its budget (2026-11-13)

`docs/PARAM-BUDGET-RETRACTION.md` established that `dmix` matters in a *trained* model (+18.33 points on the pretrained checkpoint).
An ablation cannot say whether training needs the component, which was the open question. This is the training answer: the shipped
pretraining recipe with **one change** - `--direct` omitted, so `dmix` is never built.

## Result - three seeds, by position on the matched split
```
epoch            1        2        3      params
WITH dmix     19.69    24.26    25.52    15,567,566   (matched_d512_s0, its own log)
WITHOUT       15.41    18.70    20.30     7,178,958   seed 0
WITHOUT       15.58    18.83    20.31     7,178,958   seed 1
WITHOUT       14.91    18.66    20.26     7,178,958   seed 2
WITHOUT mean  15.30    18.73    20.29     7,178,958   (sd 0.36 / 0.09 / 0.03)

deficit vs dmix   -4.39    -5.53    -5.23

VAL perplexity   ep1            ep2            ep3
WITH dmix      187.93         141.99         126.79
WITHOUT        367.68/370.57/375.07   278.58/281.50/283.00   247.75/250.86/251.45
```

**Removing the direct pathway costs 4.4-5.5 points of held-out accuracy at every epoch and roughly doubles perplexity**, for 8.4M
fewer parameters. The prediction written before the run ("within 1.0 point on >=2 of 3 seeds, i.e. removable") is **falsified**, and
its falsifier ("losing >=3 points means dmix earns its budget") is **triggered** in all three seeds.

## Honest comparison caveat
The reference is **one seed** (`matched_d512_s0`); the no-dmix arm is three. The no-dmix spread is tiny (sd **0.03** at epoch 3), so
the ~5-point gap is far outside any plausible seed variation, but the reference side carries no spread and that is stated rather than
smoothed over. A 3-seed reference run is the clean version of this table and is not done.

## What this settles
- **`dmix` earns its parameter budget during training, not only at inference.** The 50M design keeps the direct pathway.
- Together with the retraction, the ledger for this component reads: **+18.33 points** of trained-model ablation damage, **~5 points
  at every epoch** of training deficit when it is absent, and **0.00** whenever `route` is enabled - because the route switch discards
  its term (`docs/PARAM-BUDGET-RETRACTION.md`).
- The open engineering item is therefore the ordering, not the parameter count.

## Reproduction
`lens/scripts/run_nodirect_d512.sh` (prediction and falsifier in the header), logs `logs/nodirect/nodirect_s{0,1,2}.log`, checkpoints
`nodirect_d512_s*` in `lens/runs/chat/`. Reference `lens/runs/MATCHED-D512.txt` (the `NEW BEST held-out ... [val_asst_acc (BY
POSITION)]` lines).
