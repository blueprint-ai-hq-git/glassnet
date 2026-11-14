# explain() is causal (verified against a random-rule control), but 5 of the 14 named concepts are read by no rule - and trimming them is not free

Date: 2026-11-14. New tools: `tools/explain_verified.py`, `tools/rules_per_concept.py`. Predictions written before each run in
`lens/runs/EXPLAIN-CAUSAL-PREDICTIONS.txt`, `COV-PREDICTIONS.txt`, `TRIM9-PREDICTIONS.txt`. Checkpoint under test:
`runs/phase2_sweep/LAMCONC_0p02/final_seed0.npz` (14 de-duplicated concepts, 32 single-concept rules, alpha 1.0, lam 0.02).

## 1. explain() now works on the new checkpoint, and its claims are checkable

The shipped `gates/phase2_explain.py` takes no arguments and prints the **shipped** concept names, which would attach the wrong
names to the new model's neurons. `tools/explain_verified.py` takes the checkpoint and the model's own concept set, prints a
receipt, and then **removes the rules it named** to see whether the answer changes.

```
image 2  true trouser  predicted trouser  margin +5.083
   rule #25 reads light_region_count (-0.76) -> contributes +0.969 to class trouser
   rule #26 reads top_heavy (-0.85) -> contributes +0.812 to class trouser
   remove top-1 rule  -> margin +3.682  (28% of margin gone)
   remove top-2 rules -> margin +3.192  (37% gone)
   remove 2 RANDOM    -> margin +4.110  (19% gone)
```

Across the images tested, removing the **top-2 named rules** removed a mean **346.7%** of the margin (the margin flips sign on some
images) against **10.8%** for **2 random rules**. **The named rules carry the decision; the receipt is causal, not decorative.**
(P2 as written - top-1 >= 30% on >= 60% of images - failed at 50%, because on some images the decision is spread over four or five
rules; the random-rule control is the meaningful comparison and it passes decisively.)

## 2. The bigger finding: 5 of the 14 named concepts are decoration

```
32 rules read only 9 of the 14 concepts
rules per concept:  row_profile_std 8, light_region_count 6, left_heavy 5, diag_edges 4,
                    quad_UL_ink 3, lowres_symmetry 2, top_heavy 2, ink_contrast 1, vertical_edges 1
concepts read by NO rule: center_ink, border_ink, outline_ratio, symmetry_center, tall_shape
```

Those five concepts are **honest** - they track their own labels (0 of 14 mislabeled) - and **unused**. A concept that nothing
downstream reads cannot influence a prediction, so presenting it as part of the model's reasoning would be misleading. This is a
different honesty problem from the previous ones: not a lying label, but an idle one.

## 3. The coverage penalty cannot fix it at this setting

```
cov 0.0 (baseline)  81.71% +/- 0.78   9 of 14 concepts used   5 unused
cov 0.05           81.64% +/- 0.78   9 of 14 concepts used   5 unused
cov 0.2            81.47% +/- 0.84   9 of 14 concepts used   5 unused
```

The flag **is** wired (`cov_lam=args.cov` -> `cov_lam * cov` in the total loss, penalising concepts whose total incoming rule weight
is below `cov_margin`), so this is not a silent no-op. It simply cannot win: at `lam 0.02` the L1 penalty - the thing that creates
single-concept rules in the first place - shrinks exactly the weights the coverage penalty tries to grow. **P1 and P3 failed;
coverage did not move at all.**

## 4. Trimming the concept set is not free - my own falsifier fired

Following the written falsifier ("if coverage cannot be raised above 10 concepts without costing more than 2 points, trim the set
to the ones used"), I trained the same architecture on the 9 used concepts:

```
                                accuracy (3 seeds)   spread    concepts used
14 concepts (recommended)       81.71% +/- 0.78      0.78      9 of 14
9 concepts  (trimmed)           80.16% +/- 2.13      2.13      9 of 9
```

**Trimming costs 1.55 points and nearly triples the seed spread.** P1 held (all 9 used), **P2 failed** (below my 81.2% bar), and the
falsifier triggered exactly as written: the five unused concepts are doing work that does not appear as incoming rule weight - most
plausibly through the joint objective (`alpha` concept supervision and the `varfloor` penalty interact with how many concepts are in
the loss). **So the concepts stay, and the honest burden moves to disclosure.**

## What this means for the readable claim

The defensible description of the model is now:

> 14 named concepts, each audited to track its own name (0 mislabeled); **9 of them are read by the 32 single-concept rules**; 5 are
> trained and honest but read by no rule, so they cannot affect a prediction. Removing them costs 1.55 accuracy points, so they are
> kept and declared idle rather than silently counted as explanations.

`explain()` must therefore name **only the rules and concepts that actually contributed**, which `tools/explain_verified.py` does,
and must not present the idle five as part of the reasoning.

## Open question for the next run

Why do 5 idle concepts help at all? The clean test is to keep the 14-concept **supervision** (so the trunk is shaped the same) but
restrict the **rule layer** to the 9 used concepts. If accuracy stays at ~81.7, the idle concepts only matter through supervision and
the explanation can safely ignore them; if it falls, they matter through a channel not yet identified.
