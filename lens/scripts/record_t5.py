"""record_t5.py — R5 (epochs 192) GATE PASS, promoted; R6 pre-registered + endpoint rule.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
LOG = f"""
---

## 2026-10-10 — TALK LADDER, ROUND 5 (epochs 96 -> 192): GATE PASS. PROMOTED to shipping talk.

Twin = t4_ep96 (val 28.38%). ONE change: 192 epochs. Corrected engine, 43/43 concepts.

```
  metric          R4 (ep96)      R5 (ep192)     delta
  val_acc         28.38%         30.88%          +2.50   (bar 30.38 -> clears by 0.50)
  train_acc       42.60%         46.06%          +3.46   gap 15.2 — NEVER closes, just grows slower
  val_ppl         21778          35946           WORSE   (ppl/acc disagree for the FOURTH time)
  unk (probes)    0.00%          0.00%           unchanged
  gates           -              a PASS b PASS c PASS
```
PROMOTED: variants.CHAT_FINAL = t5_ep192_ep192.npz.

DIMINISHING RETURNS, RECORDED HONESTLY: round gains +5.69 -> +4.87 -> +2.50. The +2.0-margin bar
will get harder each round while gains shrink. Samples remain word-soup; zero sign of real
answering yet — the next-token metric improves without the skill improving proportionally.

### R6 pre-registered: window 32 -> 64 (the LAST named lever), d256, 192 epochs. bar = 32.88%.
### ENDPOINT RULE (pre-registered NOW): when R6 lands, regardless of pass/fail, STOP the ladder
### and deliver the endpoint report (full table + what each lever bought). No further single
### levers are pre-registered; continuing to scale epochs has hit diminishing returns, and
### choosing the next step (data expansion, architecture change) belongs to the owner.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("R5 recorded PASS, promoted; R6 (W64, bar 32.88) + endpoint rule pre-registered")