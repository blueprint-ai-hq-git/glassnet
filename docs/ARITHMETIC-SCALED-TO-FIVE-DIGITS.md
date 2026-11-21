

---

# SERIALISED ARITHMETIC: 100.0% ON HELD-OUT OPERANDS FROM 2 TO 5 DIGITS, NO BOUNDARY FOUND (2026-11-21 15:01)

Four widths, four held-out evaluations, one mechanism, 38-68 thousand parameters.

```
width   chain length   possible pairs   TRAIN     HELD-OUT (2000 unseen)   with-carry cases
2       1 carry          10,000        100.0%          100.0%             100.0% (n=76 / n=109)
3       2 carries      1,000,000        100.0%          100.0%             100.0% (n=152 / n=143)
4       3 carries    100,000,000        100.0%          100.0%             100.0% (n=169)
5       4 carries 10,000,000,000        100.0%          100.0%             100.0% (n=179)
```

Every result above was **decoded greedily from the model's own outputs** - no answer key is supplied at evaluation - and every evaluated
pair was withheld from training. At 5 digits a lookup table would need ten billion entries, so memorisation is not a conceivable
explanation by the fourth width.

## The one sentence this project now has

**The architecture could not add because an intermediate value had nowhere to live. Give it a token to live in, and it adds five-digit
numbers it has never seen, correctly, with every step readable.**

## Two honest defects found on the way, both mine, both caught by instruments

1. The **4- and 5-digit runs first reported 0.0%.** The printed example gave it away: `got = (11, 20, 5, 1, 7, 0, 3, 0)` for
   `want = (5, 1, 7, 0, 3, 0, 6, 0)` - `11` is the NOISE id and `20` is DELIM, followed by the **correct answer shifted by two**. I had
   widened the input but left `OUT_START` at its 2-digit value, so supervision *and* decoding pointed two tokens early. **The model was
   right; the ruler was wrong.** Fixed, gated, re-run: 100.0%.
2. `carryN.py` breaks at **2 digits** with a broadcast error (11 vs 12 positions) - my generalisation is off-by-one at the original width.
   The 2-digit result stands on `carry.py`, which is unchanged and was gated before its run.

Neither defect touched a published number, because each was caught before the number was believed.

## What remains unproven, stated plainly

- **No boundary has been found**, but none was searched beyond 5 digits; "scales indefinitely" is not claimed.
- **Subtraction with borrowing is untested** - a different intermediate, so success there, not addition, is what would make the mechanism
  general. Running now at 3 and 5 digits, with the falsifier written first.
- The **bounded-range** finding from the counting experiment stands: this is a procedure over the taught input range, not an unbounded
  operator. Nothing here contradicts it.
