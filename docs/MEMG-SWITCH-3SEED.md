

---

# MEMORISE VERSUS GENERALISE: THE SWITCH WORKS ON ALL THREE SEEDS, AND THE GENERALISATION HALF IS STILL WEAK (2026-11-20 20:31)

The third and last thing asked for, built and measured. Both halves are in the same model, the same training run, and the trap dataset
makes them unconfusable.

## Measured, 3 seeds, 1500 steps, same data protocol per seed

```
seed   rule acc (held-out)   model only -> with switch   fact acc (unique keys)   collisions
 0          23.33%                 23.33%                13.50% -> 100.00%          0.00%
 1          26.33%                 26.33%                11.75% -> 100.00%          0.00%
 2          25.33%                 25.33%                 8.00% -> 100.00%          0.00%
```

- **P1 memorisation MET on all three seeds:** facts go from **8.00-13.50%** (the model cannot generalise a unique key, as the trap
  intends) to **100.00%** with the switch.
- **P2 generalisation not hurt MET on all three:** held-out rule accuracy changes by exactly **+0.00** points.
- **P3 no collision MET on all three:** **0 of 600** held-out rule answer tokens matched a stored fact context.
- **Falsifier not fired.**

## What P2 does and does not prove - stated because it matters

**P2 is trivially true by construction.** The store only ever answers contexts it stored; held-out rule contexts are distinct from
fact contexts, so an exact-match store *cannot* collide with them. So "generalisation is not hurt" is a **structural property of exact
matching**, not evidence that the switch is clever about sharing capacity. It does prove the store is not swallowing rule questions,
which is what P3 checks independently.

## The weak half, before anyone reads a win into this

**Held-out rule accuracy is only 23-26%.** Train accuracy at the same moment is **49.7-52.5%**, so the model is roughly halfway to
memorising its own training pairs and has **not learned two-digit addition**. That means:

- the **memorise** half is demonstrated: **11.08% -> 100.00%** mean, no interference;
- the **generalise** half is **not** demonstrated at this budget. A 25% held-out score on a rule task proves the model is not
  memorising the held-out pairs, but it does not show it can apply the rule.

So the honest sentence is: **the switch separates the two correctly; the model's ability to generalise is still unproven at 1500
steps.** The next run trains the same trap longer to see whether the rule half can actually be learned - and if it cannot, that gets
reported as a failure of the generalisation side rather than hidden behind the switch's success.

## Scope, unchanged and load-bearing

This is the **decision rule at inference** with an exact external store. It does not change the weights. Weight-integrated memory was
already falsified in this project by measurement (+0.49 against a +1.0 bar, falsifier fired). What is proven here is the **switch**,
not gradient memory.
