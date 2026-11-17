
## Composition certified: the gradients were never why the harness failed to learn (2026-11-17)

The skipped check is now run. `tools/cell_compose_check.py` finite-differences the **whole stack** - the cell's nine parameters plus the readout `Wc` and bias `bc` - against the loss the
trainer actually minimises (softmax cross entropy through `logits = cs @ Wc + bc`), on small dimensions:

```
Wf   1.972e-07  PASS      Wxf  6.284e-08  PASS
Wi   2.417e-07  PASS      Wxi  3.857e-08  PASS
Wv   2.078e-08  PASS      Wxv  1.964e-09  PASS
bf   1.360e-08  PASS      bi   1.119e-08  PASS
bv   7.643e-10  PASS
Wc   6.765e-09  PASS      bc   5.738e-10  PASS

loss at the check point: 1.753214
COMPOSITION CERTIFIED - cell + readout + loss agree with finite differences
```

**Every parameter agrees with finite differences, from 5.7e-10 to 2.4e-07.** The cell's certified backward does compose correctly with the readout matmul and the loss.

## What that rules out, and what it leaves

**Ruled out: the composition.** `dcs = dl @ Wc.T` and the carry through the readout are right. So the previous entry's second suspect is dead, and the harness's failure to learn is **not** an
unverified gradient.

**Left standing: the cell state's magnitude.** The loss stayed within 0.1 nats of `ln(12002)` - exactly uniform - which is what happens when the readout's input is near zero or near constant.
The cell starts at `c = 0` with input weights of 0.08 and a forget gate biased open at 1.0; if `c` never grows, `logits = cs @ Wc + bc` stays near `bc` for every position and the model is
uniform by construction, no matter how correct the gradients are.

That is consistent with every memory measurement in this line: the k=1 perturbation shift on the *trained-attempt* cell was only **10.16%**, and the cell state's own response decays with distance -
a state that barely differs between positions cannot inform a readout.

## The next measurement, stated before it is run

**Measure the cell state's magnitude and spread during training**, not after: `||c_t||` per position, and the cosine between `c_t` and `c_t+1`. If `||c_t||` collapses toward zero or the
consecutive-state cosine goes to ~1 (states barely changing), the readout has nothing to read and the fix is in the cell's input/forget scale - not in the optimiser. If the states do vary and
the model still sits at uniform, the fault is in the readout or the data path and that is a different fix.

Either way it is a one-minute measurement and it must precede another training run. The rule this week has held every time: **instrument first, then train.**
