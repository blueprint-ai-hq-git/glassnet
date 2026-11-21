

---

# 8-DIGIT ADDITION SOLVED AT A FAIR BUDGET: THE "BOUNDARY" WAS MY BUDGET, CONFIRMED (2026-11-21 19:52)

The re-test was pre-registered with one change - **10,000 steps instead of 4,000** - and it converged.

```
8-digit addition, 10,000 steps, 16 written tokens, 7-intermediate chain, 36-token context
training token accuracy:   100.0% at step 8,800   (vs a 67.5% plateau at 4,000 steps)
TRAIN pairs     100.0% exact
HELD-OUT pairs  100.0% exact   (2,000 unseen, of which 195 required carries)
P1 MET   P2 MET   falsifier not fired
```

**10^16 possible pairs.** A lookup table would need ten quadrillion entries, so this is a procedure by any measure.

## Why the earlier failure is now a *useful* data point rather than an embarrassment

The 4,000-step run was recorded as **INVALID-AS-A-BOUNDARY** before this result existed, on the strength of its own training curve: 67.5%
training token accuracy where every successful arm reached 100.0%. That call was right, and it establishes a rule worth keeping:

> **Training-set accuracy is the tell.** If the model cannot reproduce what it was trained on, the experiment says nothing about
> generalisation or architecture - it says the run has not converged. A held-out number from an unconverged run must not be read as a
> boundary, and the training curve is what distinguishes the two.

That is the same distinction as today's earlier defects (the padding mask that produced a fake 0.387 loss, the decode offset that produced a
fake 0.0%): **in every case the number was testable against a second measurement, and the second measurement won.**

## The complete arithmetic line, measured end to end today

| task | result on 2,000 held-out pairs |
|---|---|
| addition, 2 digits, 1 carry | **100.0%** |
| addition, 3 digits, 2 carries | **100.0%** |
| addition, 4 digits, 3 carries | **100.0%** |
| addition, 5 digits, 4 carries | **100.0%** |
| addition, 8 digits, 7 carries | **100.0%** |
| subtraction with borrow, 3 digits | **100.0%** |
| subtraction with borrow, 5 digits | **100.0%** |
| the same tasks asked for in ONE token | **2.0% - 7.0%** |
| the same tasks with a LOOP (R=1/4/8) | **7.0%, unchanged at any depth** |

One mechanism, 38-69 thousand parameters, greedy decoding from the model's own outputs, and every step of every computation is a token a
human can read.
