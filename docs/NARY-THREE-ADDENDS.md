

---

# THREE ADDENDS: 80.5% ON HELD-OUT TRIPLES, AND ONE SEED THAT HAS NO VERDICT (2026-11-21 20:14)

A ternary column (three inputs plus the carry, so the carry out can be 2) is harder than binary, and it shows - but the two seeds tell
different stories, and the training curve says which one counts.

## Measured, 6,000 steps, 3-digit addends, held-out triples

```
seed   training token acc (final)   TRAIN exact   HELD-OUT exact   with-carry
 0            96.9%                   77.5%           80.5%        80.1% (n=196)
 1            61.6%                    0.0%            0.0%         0.0% (n=196)
```

- **seed 0 is a fair run:** training token accuracy reached **96.9%**, so the model very nearly reproduces its own training data; held-out
  **80.5%** exact, with **80.1%** on the carry cases.
- **seed 1 has NO VERDICT.** Training token accuracy stalled at **61.6%** - the same signature as this evening's 8-digit failure, where the
  model never converged and its held-out 0.0% said nothing about the architecture. By the rule established then, an unconverged run is
  **invalid**, not a boundary.
- **P1** (train >= 80%) NOT MET (77.5% on the valid seed, and the invalid one drags the mean to nothing).
- **P2** (held-out >= 60%) **MET on the valid seed at 80.5%**.
- **Falsifier** technically fired on seed 1, but on an invalid run.

## What this means, precisely

**The written-out mechanism does extend past binary - to 80.5% on triples - but not yet at the standard the binary task set (100.0%).** Two
honest explanations remain open, and this run cannot separate them:

1. **budget:** seed 1 never converged at 6,000 steps, and seed 0 only reached 96.9% token accuracy (not 100%), so the ternary task may
   simply need more steps;
2. **capacity per written token:** three inputs plus a carry that can reach 2 is a harder column rule, and 67,740 parameters may not cover
   it as comfortably as the binary case.

## The one pre-registered way to separate them

Re-run the ternary task with the budget raised (**12,000 steps**) and **three seeds**, nothing else changed.

- **P1 convergence:** all three seeds reach >= 99% training token accuracy. If they do, the earlier zeros were budget, full stop.
- **P2 accuracy:** held-out >= 80% on the mean of the three seeds, matching the best valid run today.
- **THE FALSIFIER THAT ACTUALLY DISCRIMINATES:** if training token accuracy again plateaus below 95% at 12,000 steps, then the ternary column
  is at **this model size's capacity limit**, and the boundary of the claim is stated as "binary addition and subtraction, plus partial
  ternary" - with the training curves as the evidence.

Either outcome is a real answer, and neither is obtained by quietly widening the model inside the same run.
