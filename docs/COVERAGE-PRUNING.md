# Coverage-aware pruning tested: the last sub-cause, and why it cannot work either (2026-11-12)

The dead-concept mechanism named three sub-causes: (1) a concept's last edge is pruned away, (2) the rules reading it are inert,
(3) the rule is saturated. Linear rules killed (3). This tests (1) directly with `tools/prune_rules.py --keep_coverage`: never
remove a rule that is the last reader of some concept.

**Prediction, written before the run:** keeping the last reader lifts the minimum concept influence above 1% because no concept
loses its readers, at an accuracy cost under 1.5 points against 81.19% +/- 0.98.

## What happened
The guard fired on **1 of 3 seeds**: `protecting rule #23: last reader of concept index [3]` (`ring_ink`). Seeds 1 and 2 had no
concept at risk, so pruning was unchanged there. On seed 0 it cost **0.79 points** (82.26% -> 81.47%).

And the concept stayed dead. Measured on that seed with the coverage mask applied:

```
rule #23 (the protected last reader of ring_ink): removing it changes accuracy by +0.000 points
switching off ring_ink, ordinary mask:            changes accuracy by +0.000 points
switching off ring_ink, coverage mask:            changes accuracy by +0.000 points
minimum concept influence, ordinary mask          0.00%
minimum concept influence, coverage mask          0.00%
mean concept influence, ordinary -> coverage      25.35% -> 25.50%
```

**The protected rule is itself worth exactly 0.000 points.** So the guard keeps a rule that decides nothing, in order to preserve
an edge into a concept that then still decides nothing. Both halves of the prediction failed: no influence gained, 0.79 points
paid.

## The three sub-causes collapse into one
A concept is influential only if some rule that reads it is **load-bearing**. Test (1) shows that *existence* of a reader is
irrelevant - what matters is whether that reader is worth anything. Sub-cause (2) is the same statement. Sub-cause (3)
(saturation) is the special case where the reader exists and is load-bearing in general but insensitive to this particular input.
So the whole defect is a property of the **rule layer's value distribution**, not of the concepts and not of the edges.

That also explains the one lever that did help: the rule-count sweep found the median rule is worth **+6.65 points at 8 rules**
and **+0.07 at 48**. Dead concepts are a symptom of a rule layer with too many rules for any of them to matter.

## The full ledger, six levers tested
```
lever                                  accuracy effect   effect on dead concepts
delete the dead concepts               -8.62 points      gone, but the model is worse
weight-coverage penalty (--cov)        none              none
dispersion-based concept selection     -5.97 (count-confounded)  none
concept dropout (--cdrop, inverted)    -10.52 points     none, mean influence fell
linear rules (--rule_linear)           +5.09 points      halved: 3 of 12 -> 0-2 of 12
coverage-aware rule pruning            -0.27 points      none (protected rule worth 0.000)
```
**Only linear rules did anything for dead concepts, and it did it by removing saturation** - which was never the whole cause.

## Two bugs in my own tools, caught in this run
1. **The coverage guard aborted instead of skipping.** The first version reset the keep-mask to all-ones and broke out of the loop
   when it met a protected rule, so seed 0 reported "removed 4 of 32" and an accuracy identical to the unpruned model. Fixed to
   protect that rule and continue pruning the others.
2. **Both prune modes wrote the same output filename.** The coverage run silently replaced the ordinary prune mask for seed 0.
   No checkpoint was touched, but a result file was. The mode is now in the filename (`pruned_<ckpt>_cover.json`) and **both
   masks were regenerated** so nothing is lost.

## Where this leaves the thread
Dead concepts cannot be engineered away from the concept side or the pruning side. They are a symptom of rule-layer redundancy,
and the honest options are the ones already measured: fewer rules (every rule matters, less accuracy) or more rules (more
accuracy, some rules and concepts inert). What is left untested is a training-time rule-necessity objective - making each rule
carry decision weight while it trains - and that is the next lever, with the caveat that every previous training-time attempt on
this defect (coverage penalty, dropout) changed nothing.

## Reproduction
`tools/prune_rules.py <ckpt> --keep_coverage`; mask selection in `tools/explain_image.py` via `GLASSNET_PRUNE_MASK`. Checkpoints
under `runs/phase2_sweep/L_A3.0_R32_s*/`; both mask JSONs kept per seed.
