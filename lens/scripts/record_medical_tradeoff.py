"""record_medical_tradeoff.py — the concept loss is a TRADE-OFF, not a free win.
Buys readability (13 -> 20-30 concepts) at the cost of ~3.4 pts accuracy and 3.7x the variance.
Also draws the distinction between two different comparisons that must not be conflated.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-09 — MEDICAL: the concept loss is a TRADE-OFF, not a free win (seeds corrected it)

The low-weight sweep showed accuracy flat at 66.03-67.35% while the concept audit climbed 13 -> 30/35.
Single-seed runs made 0.05 (67.35%) and 0.10 (67.33%) look like the best of both worlds. **Three seeds
each say otherwise.**

```
  lam_concept   mean acc   std     concept audit        (3 seeds each)
  0.00           65.39     1.33    13-16 / 35    accurate AND stable, poorly readable
  0.05           61.87     4.98    20-23 / 35
  0.10           62.15     4.99    21-28 / 35
  0.15           62.02     4.51    22-30 / 35

  g1  best weighted arm beats 0.00's 65.39 -> FAIL (nothing does; all land ~62)
  g2  variance not worse than 0.15's 4.51 -> FAIL (0.05 and 0.10 are ~5.0)
  g3  concepts >= 20/35                    -> PASS (all weighted arms)
  g4 APPLIES: no weight dominates. The choice is a STATED TRADE-OFF, not a winner.
```

### THE FINDING
**Turning on the concept loss buys readability (13-16 -> 20-30 of 35 named concepts firing) and pays
for it with about 3.4 points of mean accuracy and 3.7x the run-to-run variance.** Weight 0.00 is both
the most accurate AND the most stable arm; every weighted arm is less accurate and much more variable.
The variance increase is itself a real finding: the concept loss makes training less reliable, not
just slightly worse on average.

### THE DISTINCTION THAT MUST BE KEPT STRAIGHT (this matters for prior claims)
Two DIFFERENT comparisons, easy to conflate:
  (a) **readable architecture vs parameter-matched PLAIN control** — this is the +24.62 pt medical
      and +6.74 pt Fashion-MNIST result. It compares two ARCHITECTURES. **Unaffected by this finding,
      and it still stands.**
  (b) **concept loss on vs off WITHIN the readable architecture** — this is what changed today.
      Turning the concept loss on costs ~3.4 pts and 3.7x variance.
So the honest statement is: *the readable architecture is better than the plain control; and within
the readable architecture, the concept loss trades accuracy and stability for interpretability.*
Those are compatible, and (a) was never a claim about (b).

### WHAT THIS MEANS FOR SHIPPING
The shippable medical artifact depends on the requirement:
  * maximum accuracy + stability  -> lam_concept 0.00 (65.39 +- 1.33), but only 13-16 concepts fire
  * maximum interpretability      -> lam_concept 0.15 (62.02 +- 4.51), 22-30 concepts fire
There is no single "best" without naming which property is wanted, so both are recorded with their
numbers rather than one being declared the winner.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended medical trade-off record")
print("  0.00: 65.39 +- 1.33, 13-16 concepts  |  0.15: 62.02 +- 4.51, 22-30 concepts")
print("  g4 applies: stated trade-off, no winner")
