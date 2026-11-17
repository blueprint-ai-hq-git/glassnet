
## A naive carry path cannot buy memory, and the reason sets the architecture (2026-11-17)

Before building a gated-memory arm, the cheapest possible version was tested: add an identity carry to the recurrence, `h_{t+1} = tanh(x Wx + h Wh + b) + beta * h_t`, and see whether
memory comes back without the saturation that destroyed radius 3.4. These weights were trained with `beta = 0`, so **no accuracy claim is made here** - this is a feasibility probe.

```
  beta     k=16 shift       acc%    entropy state norm
   0.0         0.788%     14.69%     6.6466       2.93
   0.2         1.719%     10.86%     7.9453       6.12
   0.4         0.948%      3.24%     8.0835      14.06
   0.6         1.176%      3.98%     8.2160      27.67
   0.8         1.161%      3.28%     8.2078      63.24
   0.9         0.443%      4.26%     8.1123     126.20
```

## It fails, and the failure is the useful part

**The k=16 shift never exceeds 1.719% at any carry strength** - it does not approach the 10% gate at all - while the **state norm explodes from 2.93 to 126.20** and entropy rises to ~8.2 nats.

The mechanism is visible in those two columns together: adding a carry makes the state **grow**, growth pushes `tanh` into its **saturated** region, and a saturated `tanh` has near-zero derivative, so a perturbation is **squashed on the way through** no matter how much of the old state is
added back. **The carry raises the state's magnitude and destroys its sensitivity in the same move.**

That is the same failure mode as radius 3.4, reached from a different direction. It kills the whole "add the old state back" family on this architecture.

## The design constraint this establishes

Memory cannot live **inside** a saturating nonlinearity. Keeping information to 16+ steps without losing sensitivity requires a channel that is **linear and untanh'd**, carried separately from the
nonlinear hidden state, with a learned gate deciding what enters and what is erased.

That is exactly the LSTM-cell / SSM structure, arrived at here from measurement rather than by copying a design: the probe shows that the two requirements - *hold information* and *stay sensitive* -
are in direct conflict within a single `tanh` recurrence, and only separate channels resolve it.

## What the arm must therefore be, pre-registered read

- A **separate linear state** `c_t` (no `tanh` on the carry path) alongside the existing nonlinear `h_t`.
- **Learned per-dimension gates** for write and erase.
- **The spectral clamp must be off or raised** for this arm, or the transition is normalised back and the mechanism reports no effect.
- **Gate, unchanged:** k=16 perturbation shift above **10%** while held-out accuracy holds at or above **16%**.
- **Falsifier, sharpened by this probe:** if the separate linear state also fails to lift the k=16 shift above 10%, then no recurrence of this size holds 16-step memory on this corpus and the
  memory question is answered negatively for the whole family - not just for the clamp.
- **Accuracy is a pass/fail co-condition, not a target:** a memory fix that costs accuracy is not a fix, and this probe shows how easily that happens.

Nothing is launched yet: the arm needs its own gradient verification for the gate parameters before any training, per this project's rule that a new term gets an analytic gradient checked
against finite differences before it is trusted.
