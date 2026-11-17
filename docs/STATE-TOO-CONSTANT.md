
## Why the cell harness sat at uniform: the state is not collapsed, it is nearly CONSTANT (2026-11-17)

The pre-registered measurement (`tools/cell_state_probe.py`), taken during training:

```
step 0             ||c|| mean 2.5140 (t=1 0.3529, t=64 5.5634) | cosine(c_t,c_t+1) 0.9414 | logit sd 0.5937
step 20 loss 9.345 ||c|| mean 1.8568                          | cosine            0.9581 | logit sd 0.3820
step 40 loss 9.313 ||c|| mean 2.0706                          | cosine            0.9626 | logit sd 0.4249
step 60 loss 9.230 ||c|| mean 2.1931                          | cosine            0.9650 | logit sd 0.4498
```

## Hypothesis A, exactly

`||c_t||` does **not** collapse - it sits near 2.2 and recovers after an early dip. What is wrong is the **direction**: **cosine between consecutive states is 0.9414, and it RISES to 0.9650 during training.**

**The state barely changes from one position to the next.** The readout sees almost the same vector at every step, so its output is near `bc` everywhere, and the distribution is uniform by
construction - regardless of how correct the gradients are, which is what the composition check already established.

Worse, **training makes it more constant, not less** (0.9414 -> 0.9650), and the logit spread shrinks (0.594 -> 0.450). The model is converging toward "predict the marginal distribution" because that is
the best a readout can do when its input carries no positional information.

## What the state is actually doing

A high consecutive-state cosine with a healthy norm means the carry is dominated by a **slowly drifting running average** of everything it has seen: memory of the *average*, not response to the
*present*. That is the mirror image of the earlier finding. There, memory decayed in ~8 tokens; here, the state changes so little that **successive positions are nearly indistinguishable**.

Both are the same underlying problem seen from two sides: the cell's carry has no reason to separate "what I remember" from "what I am looking at now".

## The design implication, stated before the change

The readout must see **the current input as well as the state**: `logits = [c_t ; x_t] @ Wc + bc`, rather than `c_t @ Wc`. That gives the readout position-specific information (the token right
now) while `c` carries the long-range information. It is the standard output-projection-with-input-residual pattern, and here it is arrived at from a measured cosine rather than assumed.

**Gate for that change, written now:** the cell harness must reach **above 5% next-token accuracy** (it is at 0.00% now) with the state's consecutive cosine **below 0.90**, before any memory claim is
made on it. **Falsifier:** if accuracy stays at 0.00% with the input present in the readout, the fault is not the state's flatness and the harness is abandoned rather than re-scaled again.

Composition is certified, so whatever this change does, the gradient path will not be the excuse.
