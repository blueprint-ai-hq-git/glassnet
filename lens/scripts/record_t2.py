"""record_t2.py — T2 (width d128->256) GATE PASS + promotion + R3 pre-registration.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — TALK LADDER, ROUND 2 (width d 128 -> 256): GATE PASS. PROMOTED to shipping talk.

Twin = t1_w32 (W=32, d=128, val 14.99%). ONE change: width 256. Cross-checked on the corrected
gradient engine, 24 epochs, 26s/epoch (cheap).

```
  metric          R1 (W32,d128)   R2 (W32,d256)   delta
  val_acc         14.99%          17.82%          +2.83   (bar was 16.99)
  val_ppl         11304           10180           better
  train_acc       23.18%          26.87%          +3.69
  unk (probes)    0.00%           0.00%           unchanged
  gates           -               a PASS b PASS c PASS
```
**Width WAS part of the blocker.** R2 promoted: variants.CHAT_FINAL now = t2_d256_ep24.npz
(also NOTE: the previous shipping talk was the PRE-FIX asonly_v12000_ep24 — the new ship is on the
corrected engine AND better: 17.82 vs 9.86 val_acc).

### ALSO FIXED BY THE PROMOTION (the stale-test bug pattern, third time caught)
selfcheck.py hardcoded `[0]*16` context for the talk checks. The new ship is W=32, so the check
crashed with "windows must be 32 tokens wide". Same class as verify_explain_fidelity pinning the old
K22 and the eval_all_code shared-vocab issue: **tests must follow the shipping default, not a
hardcoded constant**. Now _ctx(v) builds the context from the loaded model's W. SELFCHECK ALL PASS.
Also: the persistent-shell PYTHONPATH pollution (hermes 3.14 numpy shadowing the venv's 3.11) is
neutralized by running selfcheck via its own script with env -u PYTHONPATH when needed.

### R3 pre-registered (ONE change from R2): epochs 24 -> 48. d256, W32, lr 0.4 decay 0.5, lam 0.3,
seed 0, gpu, assistant-only. Model is UNDERFIT (train 26.9% vs val 17.8%; tokens/param 0.30) so
more of the same data is the textbook next step and a single change.
  bar = R2 17.82 + 2.0 = 19.82%  (same margin rule)
  a unk<=1%  b val>=19.82%  c fluent  -> ALL: promote R3 to shipping; else STOP the ladder:
  width + window already bought 13.38 -> 17.82 (the +margin rule would demand +2.0 each time and
  running the ladder until the margin breaks is the honest "best one on evidence" endpoint).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("R2 recorded: PASS, promoted to shipping (17.82%); R3 pre-registered (epochs 24->48, bar 19.82)")