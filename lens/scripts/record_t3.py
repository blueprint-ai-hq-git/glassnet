"""record_t3.py — R3 (epochs 48) GATE PASS, promoted; R4 pre-registered (epochs 96).
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — TALK LADDER, ROUND 3 (epochs 24 -> 48): GATE PASS. PROMOTED to shipping talk.

Twin = t2_d256 (W32, d256, ep24, val 17.82%). ONE change: 48 epochs. Corrected engine, 43/43
concepts passing (AUROCs mostly 0.9+, coverage 100%).

```
  metric          R2 (ep24)      R3 (ep48)      delta
  val_acc         17.82%         23.51%          +5.69   (bar was 19.82 -> clear)
  train_acc       26.87%         35.70%          +8.83   still underfit (gap 12.2, growing)
  val_ppl         10180          14081           WORSE   (ppl and acc disagree again — the known
                                                          "ppl lies" lesson; gates are acc-based)
  unk (probes)    0.00%          0.00%           unchanged
  gates           -              a PASS b PASS c PASS
```
PROMOTED: variants.CHAT_FINAL = t3_ep48_ep48.npz. selfcheck re-run after promotion (must pass;
the _ctx fix from R2 covers the W=32 ship).

Underfit is the live signal (train 35.7% vs val 23.5%, both still rising) -> the honest next single
change is the same lever, one step: epochs 48 -> 96 (R4). bar = 23.51 + 2.0 = 25.51%.

### THE RULES OF THIS LADDER (pre-registered at R1)
  * every round is ONE change from the last PASSING ship
  * gate = twin val_acc + 2.0 pts AND unk <= 1% AND fluent prose; BINARY, gate never moved
  * PASS -> promote to shipping and continue; FAIL -> STOP the ladder, ship the last passer,
    report the full ladder table as the endpoint ("best one on evidence", margins exhausted)
  * val_acc (next-token) is the gate metric; ppl is reported but deliberately not gating, because
    ppl has lied twice in this project's history (ranked codedoc_full best while exec rate said 0/25)
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("R3 recorded PASS, promoted; R4 pre-registered (epochs 48->96, bar 25.51)")