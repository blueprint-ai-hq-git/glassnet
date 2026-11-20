

---

# FACTORED RANK 32: EQUAL ACCURACY AT 1.84x FEWER PARAMETERS (2026-11-20 16:32)

The owed retry, and this time both predictions are met.

```
embedding          params     table share   held-out acc   vs full table
full table E      836,352        91.8%         12.33%         -
factored r=16     261,280        73.9%         10.86%      -1.47
factored r=32     454,336        85.0%         12.51%      +0.18
```

- **P1 MET:** 12.51% against a bar of >= 11.33%. It is also **0.18 above** the full table.
- **P2 MET:** 454,336 parameters, **1.84x smaller** than the full table's 836,352.
- **Falsifier not triggered**, so factored embeddings are a working mechanism at rank 32.

## What may and may not be said

**May be said:** at matched budget, a factored output table reaches the same held-out accuracy as a full table with **1.84x fewer
total parameters**. That is the "smaller, equally good" claim, and it is now measured rather than hoped for.

**May NOT be said:** that it is more accurate. **+0.18 points on a single seed at 600 steps is inside noise** - this project's own
measured draw noise on a neighbouring task was sd 0.93, and no seed sweep has run here. The claim it supports is "equal within noise",
not "better". A 3-seed confirmation is the next run, and it is what the project's own rule requires before any accuracy statement.

## The capacity question is still unanswered, exactly as predicted

Table share went **73.9% -> 85.0%**, which is the wrong direction for the real goal. The model got **smaller**, not **more layer-heavy**.
So:
- "smaller at equal accuracy" - **achieved**.
- "each layer 10x smarter / capacity in the layers" - **not advanced at all** by this mechanism.

Those were separated in the pre-registration before the run, so this is not a reinterpretation after the fact: rank 32 was only ever
able to rescue accuracy, and it did.
