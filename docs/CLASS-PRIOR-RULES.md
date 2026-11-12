# Rules that read nothing are not dead: they are class priors, and deleting them raises accuracy (2026-11-12)

The 48-rule honest arm (alpha 3, 12 de-duplicated concepts) reported **13 of 48 dead rules** and a median per-rule ablation of
+0.07 points. A dead rule is not free for a readable model - it is a sentence a human must read that changes nothing - so this
run tried to delete them. What it found is more interesting than the deletion.

## Finding 1: the "dead" rules are class priors wearing a rule's name
Measured on `A3R48_s0/final_seed0.npz` over all 10,000 test images:

```
rule   |Wr| mean   |Wa| max   output spread across images   accuracy change from zeroing its inputs
  6      0.0001     0.340          0.0001                            +0.00 points
 32      0.0001     0.510          0.0001                            +0.00 points
 27      0.0001     0.620          0.0001                            +0.00 points
 ...
15 of 48 rules have an output that is CONSTANT across every image (spread < 0.001), while their answer row still pushes a
class with magnitude 0.34 - 0.79. Correlation between how much a rule reads and how much it pushes: 0.918.
```

So the rule layer contains **15 class-prior terms** (31% of the layer). They are not inert: zeroing their *inputs* costs nothing
because they never read an input, but their answer row turns the constant output into a hidden bias on one class. A human
reading rule #6 is handed a "rule" with no conditions that silently shifts a class. That is a readability defect, and the
per-rule ablation metric could not see it - it ablates inputs, and these rules have no inputs to ablate.

## Finding 2: two structural fixes, both tested
A rule can only be constant if its inputs vanish, so the fix is to make a constant rule contribute exactly zero.
1. `--nobias`: remove the rule bias vector.
2. `--center_rules`: the answer layer reads `(rule - 0.5)`.

Either alone is insufficient, which is a detail worth keeping: with a bias, `sigmoid(br) != 0.5`, so centering leaves a residual;
without a bias the output sits at exactly 0.5, so a constant rule still multiplies a 0.5 into its answer row unless it is
centred. Measured maximum logit shift caused by a constant rule:

```
plain 48 rules            0.3640
--nobias only             0.2966
--center_rules only       0.0383
--nobias --center_rules   0.0013     <- the class-prior channel is closed
```
Cost: accuracy 74.73% -> 68.66% (3 seeds) and the number of constant rules actually RISES to 21 of 48 - the optimiser finds
them cheap once they are harmless. Rules genuinely become disposable: **29 of 48 removable**, 19 left to read, and after
pruning that arm reaches 72.28% on seed 0.

## Finding 3: pruning makes the plain arm BETTER, not smaller
Pruning a rule means deleting its incoming weights AND its answer row, judged on a held-out slice of the train split (a rule is
removed if it costs less than 0.05 points there), greedy with re-measurement after each removal. Three seeds of the plain
48-rule alpha-3 arm:

```
seed   before     after    change    removed   rules to read
 0     76.56%    77.36%    +0.80       4 of 48        44
 1     74.51%    77.00%    +2.49       7 of 48        41
 2     73.12%    74.23%    +1.11       4 of 48        44
 MEAN  74.73%    76.20%    +1.47 +/- 0.90            43
```
**Pruning raises accuracy on all three seeds.** The removed rules were net-harmful: constants that shifted the argmax on
borderline images. So they were not "dead weight", they were quietly hurting the model - and they were unreadable either way.

## Where this leaves the trade
```
configuration                                accuracy          |rho|   misnamed   rules to read
12 concepts, a1, 16 rules, unpruned       79.88 +/- 0.32        0.478   7.0 of 12       16
12 concepts, a3, 48 rules, PRUNED         76.20 +/- 1.71        0.743   1.0 of 12       43
12 concepts, a3, 48 rules, no-bias+centre 72.28 (seed 0)          -      -               19
```
The gap between the accuracy champion and the honest configuration narrows from 5.15 points to **3.68 points**, and pruning
improves both accuracy and rule count at once. The class-prior channel is documented rather than hidden: 15 of 48 rules in the
shipped honest arm carry no conditions.

## A bug in my own pruning tool that produced a false null
The first version sorted candidates ascending and inspected the *most* harmful removal first, so it broke out immediately and
printed "removed 0 of 48" for models where 29 of 48 were individually removable. That is exactly the failure mode this project
keeps recording: a tool that reports a clean negative because of its own ordering. Fixed to sort descending (least harmful
first), and the fix is what produced every number above.

## Reproduction
`tools/prune_rules.py <checkpoint...>` reads `center_rules` from the checkpoint so a centred model is pruned with the forward
pass it was trained under; `PRUNE_TOL` sets the tolerance. Structural flags: `--nobias`, `--center_rules`. Pruning JSON written
beside each checkpoint.
