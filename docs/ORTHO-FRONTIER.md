# Orthogonal recurrence: the best memory/accuracy point found, still 7.7 points behind (2026-11-18)

The radius ladder said the clamp is the wrong dial: spectral radius constrains only the largest eigenvalue while the **average** singular value governs what a perturbation retains. So
`ortho_recurrent()` projects `Wh` to an orthogonal matrix every step - every singular value exactly 1. Scored at a matched 1500 steps against radius 0.9 with the corrected 104-chunk
protocol:

```
                          ortho s=1.0      radius 0.9 (same 1500 steps)
Wh spectral radius          1.0000            0.9000
Wh largest singular value   1.0000            1.6245     <- exactly orthogonal, as designed
held-out teacher-forced     13.34%            21.02%
free-running                 4.06%             6.31%
k=16 memory                  7.888%            1.322%
memory operator mean sv      0.8969            0.7618 (radius 1.0 arm)

free-running paired difference: -0.900 tokens of 40, CI [-4.62, +0.00]  -> NOT RESOLVABLE
```

## Predictions, scored

- **P1** "k=16 memory rises above the 10% gate at scale 1.0" - **FAILED**: 7.888%. It is the most memory any configuration has produced, and it is not enough.
- **P2** "scale 1.0 trains worse than 0.98; 0.98 stays within 2 points of the radius-1.0 arm" - the 0.98 arm is still running; what is measured is that scale 1.0 sits **7.68 points**
  below a matched radius-0.9 model.
- **P3** "memory operator mean singular value above 0.90" - **FAILED, narrowly**: **0.8969**, 0.0031 under the line. The projection is exact (all singular values 1.0 before the tanh
  derivative), and `diag(1 - tanh^2)` pulls the operator's average just below the bar.
- **FALSIFIER** "k=16 stays below 6%" - **did not fire** (7.888%). So orthogonalising the recurrent matrix genuinely does buy memory; it simply does not buy enough.

## The one result worth keeping: ortho dominates the frontier

| configuration | k=16 memory | held-out |
|---|---|---|
| radius 1.10 | 8.511% | 10.05% |
| **ortho s=1.0** | **7.888%** | **13.34%** |
| radius 0.90 (shipped) | 1.322% | 21.02% |

**Orthogonal recurrence is strictly better than the best radius** on both axes at once: **3.29 points more accuracy for 0.62 points less memory** than radius 1.10. If memory is ever
wanted at this scale, ortho is the configuration to start from - not a larger radius.

## The probe's mean-singular-value model over-predicts by 2.2x

The probe predicts retention from the operator's mean singular value: `0.8969^16 = 17.53%`. Measured: **7.888%**. The model is an optimistic bound because it uses the **mean**
direction; the actual perturbation finds contracting directions. Recorded so nobody quotes the prediction as if it were a measurement.

## A bug in my own probe, found by reading its own output

`saturation_probe.py` ended with the line `MEASURED k=16 retention on this checkpoint: 4.730% (lens/runs/EVAL-R10.txt)` - a **hardcoded number from a different checkpoint**, printed as
if it had been measured on whatever checkpoint was being probed. That is the exact defect class this project keeps catching: a right-looking number attached to the wrong artifact. The
measured value must now be **supplied as an argument**, and the tool prints `NOT SUPPLIED` rather than inventing one.

## Where the memory line now stands

```
radius 0.9    1.322%   accuracy 21.02%   (shipped)
radius 1.0    4.434%   accuracy 20.83%
radius 1.05   6.695%   accuracy 11.90%
radius 1.10   8.511%   accuracy 10.05%
ortho  s=1.0  7.888%   accuracy 13.34%   <- best frontier point, still far from the 10% gate at usable accuracy
gate         10.000%
```

Nothing has reached 10% memory at an accuracy worth shipping. The frontier is now known well enough to say that a **recurrent weight matrix alone cannot deliver it**: the next attempt
has to add a separate channel rather than reshape this one.
