

---

# SERIALISATION SCALES: THREE-DIGIT ADDITION WITH A TWO-CARRY CHAIN, 100.0% ON 2,000 HELD-OUT PAIRS (2026-11-21 14:39)

One carry was not the limit.

## Measured, 2 seeds, 1,000,000 possible pairs, 2,000 held out

```
seed   TRAIN pairs      HELD-OUT pairs    no-carry cases     WITH-carry cases
 0     100.0% exact        100.0% exact      100.0% (n=48)      100.0% (n=152)
 1     100.0% exact        100.0% exact      100.0% (n=57)      100.0% (n=143)
```

- **P1 MET**, **P2 MET**, **P3 MET**, falsifier not fired.
- The chain is **six written tokens**: `sum0 carry0 sum1 carry1 sum2 carry2`, each depending on the carry before it.
- Example of what the model wrote, decoded greedily from its own outputs:
  `928 + 367` -> `(5, 1, 9, 0, 2, 1)` = units 5, carry 1, tens 9, carry 0, hundreds 2, final carry 1 -> **1295**. Correct.
- **1,000,000 possible pairs** and every evaluated pair was withheld, so a lookup table is not a possible explanation at any point.

## The full arithmetic picture, measured today

| formulation | result |
|---|---|
| answer in ONE token | **2.0%** (sum2fixed) / **6.0%** (sum2) |
| loop R=1 / R=4 / R=8 (zero extra params) | **7.0% / 7.0% / 7.0%** |
| serialised as counting | **100.0%** (bounded by taught range on unseen sums) |
| serialised, 2 digits, ONE carry | **100.0% on 2,000 held-out pairs** |
| serialised, 3 digits, TWO carries | **100.0% on 2,000 held-out pairs** |

**The single change that made this work is where the intermediate state lives.** Not capacity (no reliable gain across 3 seeds), not data
(+6.11 points, bar missed), not depth (0.0 points at any loop count), not memory (not needed at this context size) - a place **for the
carry to exist between steps**. In this architecture that place is a token, which is why every step of the computation is a token a human
can read.

## What this licenses, precisely

**Licensed:** this architecture, at **38,444 parameters**, learns to perform multi-digit addition as a **serial procedure that generalises
to unseen operands**, and the procedure is fully readable token by token.

**Not licensed:** that it can add in one step (it cannot - 2.0%); that it generalises to a range wider than it was trained on (the counting
experiment measured 0.0%/14.0% on unseen sums); or that the same trick works for operations with a different kind of intermediate, such as
subtraction with borrowing, which has not been tested.

## Next: where does it stop?

Digit width 4 and 5, same task, same model shape. If 5 digits fails where 4 succeeds, that boundary is a real property of the architecture
and belongs in the record as such.
