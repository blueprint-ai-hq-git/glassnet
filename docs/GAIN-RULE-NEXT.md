

---

# THE OWED RETRY FAILED TOO: GRADIENT-NORMALISATION IS CLOSED, AND THE GAIN RULE IS THE ANSWER THE EVIDENCE POINTS AT (2026-11-20 18:20)

## The full table, all at emb_rank 32, seed 0, 600 steps

```
optimiser                     lr / start    held-out acc   verdict
momentum (one lr)                0.05          12.51%      reference
momentum (one lr)                0.50           0.00%      DIVERGED, loss nan by step 300
adaptive per-tensor              0.50           0.03%      sank, loss 9.41 -> 27.57
adaptive per-ELEMENT             0.50           0.26%      sank, loss ~27.6
adaptive per-ELEMENT             0.05           0.00%      sank  <- broken at the lr that WORKS for plain momentum
adaptive per-ELEMENT             5.00           0.01%      sank
```

**The pre-registered falsifier fired, in the strongest possible form.** Per-element was supposed to be the variant that rescued the
idea. It is broken at **every** learning rate tested, including **0.05, the value at which plain momentum reaches 12.51%**. That is not
insensitivity to a knob - it is an optimiser that cannot train this model at any setting.

## Why, precisely

Both variants divide the gradient by a measured magnitude:

- **per-tensor:** every element inside a tensor keeps its relative size, but the tensor's overall scale is removed.
- **per-element:** every element gets a step of nearly **equal size**, regardless of how much the loss cares about it.

The second one is the fatal case: normalising per element turns the update into sign-like steps of uniform magnitude, so unimportant
coordinates move as far as critical ones. All three per-element arms converge to the same terrible place (loss ~27.6), which is the
signature of a step rule that is overshooting in most coordinates simultaneously. **Gradient-magnitude normalisation is closed** - no
third variant, as pre-registered.

## The mechanism the evidence actually demands

A learning-rate knob exists because a step's *effect on the loss* is not known in advance. So the step should be chosen by
**measuring that effect**, not by rescaling the gradient:

```
one global direction   d = -g / ||g||
try the step           p <- p + s*d , measure the loss on the same batch
accept if the loss fell; otherwise halve s and retry (up to 4 times)
adapt for next step    first-try accept -> s *= 1.3 ; late accept -> s *= 0.7
```

The initial value becomes a guess rather than a setting. Smoke-tested before launch: 20 steps, loss **9.38 -> 8.10**, step size
self-adjusted **0.5 -> 2.0 -> 1.25**, and it beat its constant. The real test is running on the two initial values that defeated
everything else - **0.50** and **0.005** - against the 12.51% reference.

**Pre-registered:** both must land within 1.0 point of 12.51%. **Falsifier:** either arm 2.0 points below -> loss-response control
also fails here, step-size automation is **abandoned** for this model (hand-set lr documented as the shipping choice), and the effort
moves to capacity allocation.
