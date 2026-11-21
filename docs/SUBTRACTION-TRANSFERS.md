

---

# SUBTRACTION WITH BORROW TRANSFERS: 100.0% ON HELD-OUT OPERANDS AT 3 AND 5 DIGITS (2026-11-21 19:30)

The mechanism is not tuned to addition.

## Measured, both arms on 2,000 held-out pairs

```
arm                    TRAIN pairs    HELD-OUT pairs     no-borrow        WITH-borrow
3-digit subtraction       100.0%         100.0%         100.0% (n=63)    100.0% (n=137)
5-digit subtraction       100.0%         100.0%         100.0% (n=16)    100.0% (n=184)
```

Borrowing is a **different intermediate** from carrying: `d = da - db - borrow`, and if it goes negative the next column must be told. The
model writes `[diff0][borrow0][diff1][borrow1]...` and the borrow chain propagates exactly as the carry chain did. Example it decoded
itself: `741 - 269` -> diff 2, borrow 1, diff 7, borrow 1, diff 4, borrow 0 -> **472**. Correct.

So the finding is now:

> **Give an intermediate value a token to live in, and this architecture performs multi-digit arithmetic - addition and
> subtraction - on operands it has never seen, with every step readable by a human.**

## The 8-digit arm failed, and it is MY mistake, not a boundary

```
8-digit addition, 4000 steps:  TRAIN 0.0% | HELD-OUT 0.0%
training token accuracy:  58.1% @1600   57.0% @2400   57.7% @3200   67.5% @4000   loss 0.79
example: want (4,0,0,1,7,0,6,1,3,1,4,1,9,0,0,1) got (5,0,1,1,2,1,6,1,7,1,6,1,7,0,1,1)
```

Three facts decide how to read this, and all three come from the run itself:

1. **The model was never trained enough.** Its own training token accuracy peaked at **67.5%**, against **100.0%** in every arm that
   succeeded. A model that cannot reproduce its own training data has not failed at generalisation; it has failed to converge.
2. **I gave it less budget than the arms it is being compared with:** **4,000 steps** against **6,000** for the 3- and 5-digit runs, on a
   task with **16 written tokens, a chain of 8 intermediates and a 36-token context** (roughly triple the compute per step).
3. **It is not a decode-offset bug** (the mistake I made earlier today). The output is structurally right - alternating digit/carry, most
   carries correct - and simply has the wrong digits. No shift, no NOISE or DELIM tokens.

**So the 8-digit result is recorded as INVALID-AS-A-BOUNDARY**, and the reason is a budget choice of mine. The falsifier "held-out <= 20% ->
serialisation does not cover a real intermediate" fired on a run that never converged, which is not what the falsifier was written to
detect.

## The fair test, pre-registered before it runs

Re-run 8 digits with the budget equalised or raised (**10,000 steps**) and nothing else changed.

- **P1** 8-digit addition reaches >= 80% exact on TRAIN pairs, i.e. it actually converges this time.
- **P2** if P1 holds, held-out >= 50% means serialisation scales to 8 digits and a 7-carry chain.
- **HONEST FALSIFIER:** if training accuracy again plateaus below 90% at 10,000 steps, the limit is **capacity per written token** at this
  model size, not the serialisation idea - and that gets stated as the boundary, with the training curve as the evidence.
