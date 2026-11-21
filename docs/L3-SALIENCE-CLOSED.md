

---

# L3 SALIENCE CLOSES AS FAILED, ON A CLEAN INSTRUMENT, TWELVE ARMS, WITH THE REASON NAMED (2026-11-21 14:40)

The certificate was re-run on the corrected model, because every earlier arm had been measured through the batch leak. The leak was not
the explanation.

## Measured on the FIXED instrument, copy1, three seeds

```
seed   AUROC    free norm   margin    Spearman   probe acc   verdict
 0     0.7409    0.4682    +0.2727    0.2996     100.0%      SALIENCE_FAILED
 1     0.7922    0.6351    +0.1570    0.3276     100.0%      SALIENCE_FAILED
 2     0.7655    0.6062    +0.1593    0.3989     100.0%      SALIENCE_FAILED
mean   0.7662    0.5698    +0.1963    0.3420
```

- **P1 (AUROC >= 0.90) missed on all three seeds.** Across twelve arms and two instrument versions, L3 cleared 0.90 exactly **once**
  (0.9542, on the leaky instrument, and its seed-mates were 0.7130 and 0.9353).
- **P2 (margin >= +0.02) met on all three, decisively** (+0.16 to +0.27). The layer does carry information the hidden-norm baseline does
  not - that part is real and reproducible.
- **P3 (Spearman >= 0.50) missed on all three** (0.30-0.40).
- **The probe was solved perfectly at 100.0% on every seed**, so this is not a broken instrument: the model answers the task right and
  still cannot expose, through a linear read, which input position decided its answer.

## The reason, named

`AUROC 0.77` with `margin +0.20` and `100%` accuracy describes a specific thing: **importance is present in the final hidden state but not
linearly readable from it.** The free baseline collapsing to 0.47-0.64 on the clean instrument also explains the earlier confusion where
"a bigger margin" looked like progress - part of that margin was the baseline getting worse, not the layer getting better.

## Verdict and the redesign, stated so it cannot be confused with tuning

**L3 as specified - a linear read-out of the final hidden state - is FAILED.** It is not being re-tuned to pass. The honest redesign is
architectural: salience must be **inside the forward path and inside the loss** (jointly trained, with its own parameters, its score
used by the model rather than inspected after the fact), because a post-hoc probe can only report what the network happened to encode, and
this network happens not to encode it linearly.

That matters for the owner's "100% on every layer" rule: a layer that is a probe is not a layer. L3 stays UNCERTIFIED and the plan changes
shape, rather than the bar being moved.

## Across the whole day, on this one layer

| attempt | instrument | arms | P1 met | best AUROC |
|---|---|---|---|---|
| copy1, head 400 | leaky | 3 | 2 of 3 | 0.9542 |
| max2, head 400/1500 | leaky | 6 | 1 of 6 | 0.9357 |
| copy1, head 1500 | leaky | 3 | 0 of 3 | 0.8058 |
| copy1, head 800 | **fixed** | 3 | **0 of 3** | **0.7922** |
