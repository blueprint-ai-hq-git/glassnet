"""record_t4.py — R4 (epochs 96) GATE PASS, promoted; R5 pre-registered (epochs 192).
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
LOG = f"""
---

## 2026-10-10 — TALK LADDER, ROUND 4 (epochs 48 -> 96): GATE PASS. PROMOTED to shipping talk.

Twin = t3_ep48 (val 23.51%). ONE change: 96 epochs. Corrected engine, 43/43 concepts.

```
  metric          R3 (ep48)      R4 (ep96)      delta
  val_acc         23.51%         28.38%          +4.87   (bar 25.51 -> clear)
  train_acc       35.70%         42.60%          +6.90   still underfit (gap 14.2, growing)
  val_ppl         14081          21778           WORSE   (ppl/acc disagree AGAIN — ppl is not gating)
  unk (probes)    0.00%          0.00%           unchanged
  gates           -              a PASS b PASS c PASS
```
PROMOTED: variants.CHAT_FINAL = t4_ep96_ep96.npz. selfcheck re-run required.

HONESTY NOTE FOR THE LEDGER: the ladder metric is NEXT-TOKEN val_acc + basic fluency. The samples
are still word-soup ("names are just just already preventing just match me all v") — the gates say
"fluent filler with better token odds", NOT "answers questions". The gap to real answering is the
reason the ladder exists; the moment a round FAILS the margin, we stop and report the endpoint.

### R5 pre-registered: epochs 96 -> 192 (the same single lever; still underfit). bar = 30.38%.
Runtime ~90 min GPU. If R5 also clears, R6 = window 32 -> 64 (different lever, one change).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("R4 recorded PASS and promoted; R5 pre-registered (epochs 192, bar 30.38)")