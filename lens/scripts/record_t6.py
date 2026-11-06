"""record_t6.py — R6 FAIL + TALK LADDER ENDPOINT report (pre-registered stop).
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
LOG = f"""
---

## 2026-10-10 — TALK LADDER, ROUND 6 (window 32 -> 64): GATE FAIL. LADDER STOPPED (endpoint rule).

Twin = t5_ep192 (val 30.88%). ONE change: W=64. d256, 192 epochs, same data/lr.

```
  metric          R5 (W32)      R6 (W64)     delta
  val_acc         30.88%        29.71%        -1.17   (bar 32.88 -> FAIL)
  val_ppl         35946         9175          better  (ppl improved while acc fell — ppl lies, 5th time)
  unk (probes)    0.00%         0.00%
  concepts        43/43         46/46         coverage 100%
  samples         word soup     MORE repetitive ("by...by...by", "generally generally")
```
The wider window HURT accuracy and visible quality. Clean negative result, recorded as measured.

## ============ TALK LADDER — ENDPOINT REPORT (pre-registered stop) ============
```
  lever             twin -> round    delta    verdict
  W 16->32          13.38 -> 14.99   +1.61    FAIL  (bar +2.0; helps but not the blocker)
  d 128->256        14.99 -> 17.82   +2.83    PASS  (shipped)
  epochs 24->48     17.82 -> 23.51   +5.69    PASS  (shipped)
  epochs 48->96     23.51 -> 28.38   +4.87    PASS  (shipped)
  epochs 96->192    28.38 -> 30.88   +2.50    PASS  (shipped; gains shrinking)
  W 32->64          30.88 -> 29.71   -1.17    FAIL  (ladder stop)
```
  SHIPPED TALK (unchanged by R6): t5_ep192_ep192.npz — val_acc 30.88%, 0 unk, 43/43 concept audit.
  Total: 13.38 -> 30.88 next-token val_acc across five single changes on the corrected engine.
  HONEST ENDPOINT: the talk model is a fluent, low-unk word-soup generator, NOT an answerer.
  Next-token accuracy rose 2.3x but answering quality did not improve proportionally — the soup
  just got statistically more native. ppl disagreed with acc in 5 of 6 rounds (never gated on it).
  Model still underfits (train 46.06% vs val 30.88%; tokens/param ~0.30).

  NEXT MOVE = OWNER'S CALL (no more pre-registered levers):
   1) DATA EXPANSION (the natural lever — underfit with limited local data; more/better
      conversations or a bigger vocab, still zero-download license-safe only)
   2) ARCHITECTURE CHANGE (deeper stack / attention — bigger change, same from-scratch rules)
   3) PARK talk at the endpoint; resume the code or image/edit line
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("R6 recorded FAIL; endpoint report appended; shipping talk stays t5_ep192 (30.88%)")