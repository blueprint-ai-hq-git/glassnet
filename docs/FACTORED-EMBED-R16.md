

---

# FACTORED EMBEDDING: 3.2x FEWER PARAMETERS FOR 1.47 POINTS - P1 MISSED, FALSIFIER NOT FIRED, ONE RETRY OWED (2026-11-20 16:23)

## Measured, both arms on the current code, same seed, same 20k windows, same 600 steps

```
embedding              params     table share   held-out acc   delta vs constant 4.08%
full table E          836,352        91.8%         12.33%            +8.25
factored A @ B r=16   261,280        73.9%         10.86%            +6.78
```

- **P2 MET:** total parameters fall **3.20x** (836,352 -> 261,280).
- **P1 NOT MET:** the bar was >= 11.32% (within 1.0 point of the full table); factored landed at **10.86%**, missing by
  **0.46 points** and sitting **1.47 points** below the full table.
- **FALSIFIER DID NOT FIRE:** it was set at below 10.32%, and 10.86% clears that. So by the rule written before the run, **one
  retry at rank 32 is owed** - not a new idea, and not a sweep.

## The number that matters more than either accuracy

Even factored, the output table is **73.9% of the model**. Cutting the rank moved the share from 91.8% to 73.9%, which is the right
direction but nowhere near the goal: the layers still hold a quarter of the capacity, and the "10x smarter per layer" ambition needs
the layers to hold most of it. A rank-32 retry will **not** fix that (it would raise the share back toward 85%), so it can only rescue
the accuracy question, not the capacity question. Both are recorded here so the retry cannot be mistaken for progress on the real goal.

## What rank 32 can and cannot win

- **Can win:** whether factored embeddings reach full-table accuracy at fewer parameters - the honest "smaller, equally good" claim.
- **Cannot win:** shifting capacity into the layers. That needs a different mechanism (smaller vocabulary, character-level modelling,
  or allocating capacity where the data demands it), which is the piece after this.
