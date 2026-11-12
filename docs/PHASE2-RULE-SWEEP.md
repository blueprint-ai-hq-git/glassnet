# Rule-count sweep on the ORIGINAL Phase 2 Fashion-MNIST task (2026-11-12)

The chat-corpus rule layer showed that rule COUNT decides whether every rule matters (24 rules: median per-rule ablation 0.36;
96 rules: 0.000). The same sweep is now run on the original Phase 2 task, because the re-audit showed that model's 30-rule
layer was largely inert. Twelve models: four rule counts, three seeds each, identical schedule (10 epochs, batch 128, lr 0.1,
plain SGD, K=14 concepts, 3 connections per rule). Training takes seconds, so this is a full sweep rather than a spot check.

**The sweep reproduces the recorded Phase 2 number exactly at R=30**: 79.04 +/- 2.38 here against the gate's 79.04 +/- 1.97.

```
| rules | accuracy (3 seeds) | interaction | misnamed concepts | median per-rule ablation | dead rules | rules joining unrelated pairs | mean concept |rho| |
|---|---|---|---|---|---|---|---|
| 8 | 76.75 +/- 0.44 | 16.4 | 6.7 of 14 | 3.01 | 0.3 | 3.0 | 0.6 |
| 16 | 77.93 +/- 1.40 | 8.7 | 4.7 of 14 | 0.19 | 0.7 | 5.7 | 0.6 |
| 24 | 78.93 +/- 0.73 | 3.9 | 6.3 of 14 | 0.28 | 3.7 | 10.3 | 0.6 |
| 30 | 79.04 +/- 2.42 | 4.9 | 5.7 of 14 | 0.01 | 6.7 | 8.3 | 0.6 |
```

## What the sweep shows
1. **Accuracy rises with rule count**: 76.75% (8 rules) -> 77.93% (16) -> 78.93% (24) -> 79.04% (30). The extra 2.3 points from
   8 to 30 rules cost you the readable property that matters most.
2. **Rule necessity collapses as rules are added**: median per-rule ablation loss is **+3.01 points at 8 rules**, **+0.19 at
   16**, **+0.28 at 24**, and **+0.01 at 30**. At the count the original gate used, deleting a typical rule changes accuracy by
   one hundredth of a point.
3. **Dead rules grow monotonically**: 0.3 (8) -> 0.7 (16) -> 3.7 (24) -> **6.7 of 30** at the original setting.
4. **Concept interaction moves the opposite way**: 16.4% (8 rules) -> 8.7% -> 3.9% -> 4.9%. With few rules each one carries
   more of the decision, so a per-concept explanation is less complete; with many rules the explanation is more additive but
   most rules are inert.
5. **Concept honesty does not depend on rule count at all**: mean |Spearman| stays 0.55-0.62 across the sweep, and 4.7-6.3 of
   14 concepts are better explained by a different named feature at EVERY size. That is a property of the concept set, not of
   the layer above it - the same conclusion the chat model produced.
6. **Unrelated rule pairs do not improve with size**: 3.0 (8) -> 5.7 (16) -> 10.3 (24) -> 8.3 (30) rules join features that do
   not co-occur.

## The recommendation for this task
**16 rules.** It keeps 77.93% of the 79.04% ceiling (a 1.11 point cost), cuts dead rules from 6.7 to 0.7, keeps the median rule
worth +0.19 rather than +0.01, and holds interaction to 8.7%. Eight rules is the maximally-faithful end (+3.01 per rule) but
costs a further 1.2 points and pushes interaction to 16%.

The original gate's 30-rule configuration is the least faithful of the four: the same accuracy as 24 rules within noise
(79.04 vs 78.93), 6.7 dead rules, and a median rule worth 0.01 points. "30 rules a human must read" was true as a count and
false as a description of what the model uses - a human would need to read about 24.

## Method notes
- Accuracy is the final-epoch test accuracy from the trainer, re-measured independently by `tools/phase2_reaudit.py` on all
  10,000 test images (the two agree exactly, which is what validates the audit).
- Per-rule ablation zeroes a rule's input weights; the rule's bias remains, as it does for a pruned rule.
- Rule plausibility binarises each continuous concept at its own median, stated because a 0.5 threshold would be meaningless
  on features whose mean ranges from 0.11 to 0.84.
- All twelve models are kept under `runs/phase2_sweep/R*_s*/`; the original `runs/phase2/` results were not touched.
