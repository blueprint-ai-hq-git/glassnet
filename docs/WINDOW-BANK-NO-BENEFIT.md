

---

# THE READABLE WINDOW BANK: READABLE, AND USELESS AT THIS BUDGET (2026-11-19 11:45)

Predictions and the falsifier were written before the run. Result:

```
arm            params      HELD-OUT              train ema   gap
winoff (K=0)   1,684,224   6.07%  (loss 6.5013)   7.0492    -0.5479
winon  (K=8)   1,848,320   6.04%  (loss 6.5862)   7.0475    -0.4613

window at the end: ` | class | for | a | library | management | ` | book   (8 distinct real words)
```

- **P1** (both clear the 14.70% trivial constant): **FAILED** - both are less than half of it.
- **P2** (window at least 0.5 points above control): **FAILED** - it is 0.03 points *below*.
- **P3** (window holds K distinct real words): **MET** - eight distinct tokens, no collapse. The readability problem is solved.
- **FALSIFIER FIRED**: "if the window is not at least 0.5 points above the control, it buys nothing measurable at this budget and the memory
  line stops being built and starts being diagnosed."

## The diagnosis this forces

Both arms sit at **~6%** while the per-window constant predictor scores **14.70%**. The model is not merely unimproved by memory - it is
**worse than guessing the most frequent token in the window**. A mechanism that supplies context cannot show a benefit to a model that
has not yet learned to use the context it already has.

Which means the experiment was mis-designed, by me: at 1,200 steps this trainer produces a model that is below trivial, so no memory
mechanism - working or not - could have shown anything. The base models that reach 21.74% needed **6,000** steps. Every memory arm I
have run today used 600-1,500 steps. **The memory question has never been tested at a budget where the model can answer it.**

## What is established vs not established

- **Established:** the window bank is readable (eight real words), its gradient is exact (4.334e-10), its batched path is equivalent to the
  certified one (1e-16), and it does not collapse the way the previous bank did (slot-to-slot cosine -0.081 against 0.864).
- **Not established:** that it helps. At this budget it is indistinguishable from no memory at all, and the honest reading is that the
  test was too weak to tell.

## The next experiment that can actually answer it

Window vs no-window at **3,000-4,000 steps**, where this trainer's models clear the constant - and with the delayed-recall probe as a
second, sharper instrument, rather than next-token accuracy alone. If the window cannot beat its control once the model is above trivial,
then the memory line is genuinely dead and I will say so with a number. If it can, the earlier arms were measuring nothing but noise.
