

---

# THE FALSIFIER FIRED: THE CONCEPT LINE CLOSES AT THIS SCALE (2026-11-20 13:12)

`narrowdir` - 9 concepts WITH the direct path - was the last configuration that could have been both honest and accurate. It was not.

## Verdict against its own pre-registration, written before the run

```
P1 required: acc >= 13.18% (12.18 + 1.0), OR within 1.0 of 12.18 with audit >= 8/9
   actual: acc 9.35% = 2.83 BELOW 12.18, audit 5 of 9        -> P1 FAILED
P2 required: positive supply below 8.476x
   actual: 80.896x, i.e. 9.5x HIGHER                          -> P2 FAILED
FALSIFIER: acc more than 2.0 below 12.18 AND audit below 6 of 9
   2.83 > 2.0 and 5 < 6                                       -> FIRED
```

By its own words: "narrowing does not combine with the direct path and the honest+accurate configuration does not exist at this
scale - the concept line closes for good."

## The four configurations measured at 100k windows

```
configuration                          acc      ppl      audit     supply
36 concepts + direct (decorr 0)     12.18%   712.92  25/36 (69%)    8.476x   <- most accurate
36 concepts, no direct (decorr 0)    7.31%   953.87  25/36 (69%)  236.039x
9 concepts, no direct (decorr 0.5)   7.37%  1275.33   9/9 (100%)   19.839x   <- most honest
9 concepts + direct (decorr 0.5)     9.35%  1269.90    5/9 (56%)   80.896x   <- neither
```

The two good properties refuse to combine. Narrowing buys honesty **only** in the bottlenecked shape (9/9) and **destroys** it once
the direct path is present (5/9, worse than the 36-concept layer's 69%). The direct path buys accuracy (12.18% vs 7.31%) and the
least cancellation (8.476x) but leaves concepts a passenger.

## What closes, stated plainly

- **Concept count is not the lever.** 36 -> 9 moves accuracy and honesty in opposite directions depending on the direct path.
- **No configuration at 100k windows is both honest and accurate.** The honest one scores 7.31-7.37%; the accurate one has 69% honest
  concepts and a bypass that carries most of the decision.
- **Concentration never moved** (top-half 74.2-81.6%, entropy 0.906-0.950 across all nine measurements).
- This is the honest answer to "is our readable architecture better than a regular model": **at this scale it is not, and there is
  still no plain matched control for the chat architecture to even measure that comparison** - that gap stays recorded in QUEUE.txt.

## The one arm this closure still leaves open

The most accurate configuration (36 concepts + direct) was trained at `lam_decorr 0`; the decorrelation weights were only ever
measured on the **bottlenecked** shape. So one last arm asks whether the winner improves at 0.5. If accuracy drops more than 1.0 point
below 12.18%, decorrelation hurts the direct-path configuration and the best-known setting stays 0 - and then **no further concept
arm of any kind is run.**
