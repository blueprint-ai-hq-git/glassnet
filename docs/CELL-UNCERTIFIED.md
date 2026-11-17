
## The gated cell is written, uncertified, and its own precondition fails - so no arm is built (2026-11-17)

`lens/cell.py` implements the structure the skip probe pointed at: a **separate linear cell state** that never passes through `tanh`,
`c_{t+1} = f * c_t + i * v`, with learned forget/input gates and a tanh candidate. Forward and backward are explicit, plain NumPy.

`tools/cell_selftest.py` ran the two checks that must pass before any training. **Both fail.**

## 1. Gradients: fail at T >= 2

```
T=1  worst abs diff 2.429e-12 at Wxi(2,1)   numeric -0.002988  analytic -0.002988   <- exact
T=2  worst abs diff 2.903e-02 at bv(0,)     numeric +1.475758  analytic +1.504791
T=3  worst abs diff 1.004e-01 at bv(0,)     numeric +2.805122  analytic +2.905538
```

The **single-step case is exact to 2.4e-12**, so the per-gate derivatives are right. The error appears at T >= 2 and grows with T, which means the **carry path through time** is
mismanaged - the analytic answer is consistently *larger* than the numeric one, i.e. a term is being counted that should not be. My first version of `backward` took a single
final-state gradient and could not represent a loss that touches every step; the rewrite to per-step upstream gradients fixed the shape error but not this one. **The module is not
certified and nothing may be trained on it.**

## 2. Memory: the structure does not clear its own gate either

Same probe that killed the tanh carry, on random weights:

```
   variant               k      shift
   random init           1     50.17%
   random init           4     18.71%
   random init           8     10.50%
   random init          16      3.10%
   random init          24      1.07%
```

**k=16 gives 3.10%, against the 10% gate.** Better than the tanh carry's 1.719% - the linear state does help - but not enough, and the decay is still roughly geometric.

## What this means, stated plainly

The falsifier written before this work was: *if the separate linear state also fails to lift the k=16 shift above 10%, then no recurrence of this size holds 16-step memory on this
corpus and the memory question is answered negatively for the whole family.*

That falsifier has **not** legitimately fired, and I will not pretend otherwise. It was written for a **trained** arm, and these are untrained weights: a gate whose job is to hold
information has not been given any opportunity to learn to hold it. **An untrained gate failing to remember is not evidence about a trained one.**

But the adjacent fact does matter: with gates initialised open (forget bias 1.0) and random weights, the carry still decays 30x from k=1 to k=16. If the mechanism is to be worth an
arm, that decay needs a real explanation, and it has not got one yet.

## What happens next, in order

1. **Fix the backward** so the gradient check passes at T=2, 3, 8, 32. No exceptions - this project does not train on an unverified gradient. The T=1-exact / T>=2-wrong split
   localises the fault to the carry-through-time path, which is a short and checkable region.
2. **Only then** ask whether the memory probe on *trained* gates clears the gate, with the spectral clamp off - the arm, as pre-registered.
3. If both hold, the arm runs under the existing gate (k=16 above 10%, accuracy at or above 16%). If the gradient check cannot be made to pass, the cell is abandoned rather than
   patched around.

**No arm launched, no number claimed, and the module is documented as uncertified** so it cannot be picked up later as if it had passed.
