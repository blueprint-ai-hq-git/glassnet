

---

# CORRECTION: BOTH TERNARY RESULTS ARE WITHDRAWN - MY CARRY TARGET WAS NOT THE ARITHMETIC (2026-11-21 20:45)

The three-addend experiments published an hour ago - "80.5% on held-out triples" and then "seed 0 reached 100.0%" - are **withdrawn as
invalid**. The defect is mine and it is in the thing the model was being scored against.

## The defect

`expected_nary` recorded the carry with `1 if carry else 0` - a **flag**, not the carry. With three addends a column sums to at most
9+9+9+2 = 29, so **the carry can be 2**. The written trace therefore did not represent the arithmetic:

```
999 + 999 + 999 = 2997      true carries: 2, 2, 2
what my code recorded:      sums [7, 9, 9]  carries [1, 1, 1]   ->  decodes as 997
```

`123+456+789` happened to match (carries 1,1,1) which is why nothing looked wrong at a glance; `999+999+999` does not. **The model was
being trained and scored on a trace that was not the computation**, so both numbers measured the wrong thing.

## Why this is the same trap as three earlier ones today

This project has a rule: never construct the phenomenon you are testing for. Four times today a number looked like a result and the number
itself was the fault - the padding mask that produced a fake 0.387 loss, the decode offset that produced a fake 0.0%, the 4,000-step run
that was a budget artefact rather than a boundary, and now a ternary target that was not the arithmetic. In every case **the second
measurement caught it**, never code review. The useful habit is the one that caught this: seed 0 reaching exactly 100.0% while two seeds
sat at ~60% is *too clean* - a task where two seeds collapse while one is perfect deserves a check of the target, not applause.

## Fixed and verified

`carries.append(carry)` - the true carry, 0, 1 or 2. Reconstruction checked on four cases:

```
999+999+999 -> sums [7,9,9] carries [2,2,2] -> 2997  MATCH
123+456+789 -> sums [8,6,3] carries [1,1,1] -> 1368  MATCH
500+400+300 -> sums [0,0,2] carries [0,0,1] -> 1200  MATCH
1+2+3       -> sums [6,0,0] carries [0,0,0] ->    6  MATCH
```

Binary paths are unaffected because a two-addend carry is naturally 0 or 1: `73+4 -> [7,7] [0,0] = 77`, `99+1 -> [0,0] [1,1] = 100`.
**Every binary result in the record stands** - 2, 3, 4, 5 and 8 digits at 100.0%, and subtraction with borrow at 3 and 5 digits at 100.0%.

## What is and is not established right now

- **Established:** binary addition (1-7 carry chain) and subtraction with borrow generalise to unseen operands at 100.0%, decoded greedily,
  with a fully readable trace.
- **WITHDRAWN:** everything about three addends. Whether the mechanism extends past binary is **open again**.
- The corrected ternary test is re-running with the same pre-registered bars; its numbers will be reported as they come, and if it does not
  generalise the boundary of the claim is exactly "two numbers".
