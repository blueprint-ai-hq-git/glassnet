# CORRECTION: "ortho dominates the frontier" was a protocol mismatch, not a result (2026-11-18)

One wake ago this project published: **"ortho dominates the frontier - 3.29 points more accuracy for 0.62 points less memory than radius 1.10."** That comparison was invalid and the
claim is **withdrawn**.

## The mistake

`ortho`'s held-out accuracy (13.34%) came from the **corrected 104-chunk protocol**. `radius 1.10`'s (10.05%) came from the **old 12-chunk protocol**, measured before the eval fix. The
two protocols disagree by 2.9-4.0 points on identical checkpoints, so subtracting one from the other made an artefact look like a 3.29-point win. This is the *same* error class the eval
fix had just documented - and it was made in the very next measurement.

## Every arm re-measured on one protocol (104 chunks = 6,552 scored positions, SE 0.50)

All at 1500 steps, same recipe:

```
config         memory k16    held-out (104 chunks)   held-out (12 chunks, old)
radius 0.90      1.322%            21.02%                 16.27%
radius 1.00      6.125%            18.64%                 13.23%
radius 1.05      6.695%            14.80%                 11.90%
radius 1.10      8.511%            13.06%                 10.05%
ortho s=0.98     7.121%            14.13%                  -
ortho s=1.00     7.888%            13.34%                  -
```

## The corrected verdict on orthogonal recurrence

- **ortho s=1.00 vs radius 1.10** (the comparison the claim rested on): **13.34% vs 13.06%** - a **0.28-point** difference, not 3.29. With SE ~0.50 per arm the difference has SE ~0.71,
  so **0.28 is 0.4 standard errors: NOT RESOLVABLE**.
- **ortho s=0.98 vs radius 1.05**: **14.13% vs 14.80%** - radius 1.05 is **0.67 points better** at **0.43 points less memory**. Also not resolvable, and pointing the other way.

**So orthogonal recurrence does not dominate the frontier. It sits on it, indistinguishably from the radius arms at matched memory.** The honest statement is that the two mechanisms
buy memory equally well and cost accuracy equally, and neither is better on this evidence.

## Predictions for the ortho arm, scored

- **P1** "k=16 memory above the 10% gate at scale 1.0" - **FAILED** (7.888%).
- **P2** "scale 1.0 trains worse than 0.98; 0.98 stays within 2 points of the radius-1.0 arm" - **half held, half failed**. First half **HELD**: 0.98 gives **14.13%** against 1.0's
  **13.34%** (+0.79). Second half **FAILED**: ortho 0.98 sits **4.51 points behind** radius 1.0's **18.64%**, not within 2.
- **P3** "memory operator mean singular value above 0.90" - **FAILED narrowly** (0.8969).
- **FALSIFIER** "k=16 below 6%" - did not fire (7.888%), so orthogonalising does buy memory.

A measured addition from the matched pair: **free-running is now RESOLVABLE in radius 1.0's favour**, 6.31% vs ortho 0.98's 4.31%. Orthogonalising the recurrence makes generation
worse, not just accuracy.

## The rule this earns

**When the measurement protocol changes, every number it produced must be re-measured before it is compared with anything.** A protocol fix is not applied retroactively to old numbers;
it invalidates them for comparison purposes. The project already had "audit the instrument before blaming the model" - this is its second half: **audit the instrument before comparing
anything, especially against a number you have not re-measured.**
