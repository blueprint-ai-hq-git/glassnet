"""record_t1.py — T1 (window 16->32) verdict and T2 pre-registration.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — T1 (talk, window 16 -> 32): GATE FAIL — window is NOT the blocker, but it helps.

Twin = asonly_v12000_fixedgrad (W=16, d=128). ONE change: window 32. Same data/vocab/lr/epochs.

```
  metric            W=16 twin      W=32 (T1)      delta
  last val_acc      13.38%         14.99%          +1.61
  val_ppl           13064          11304           better
  train_acc         19.65%         23.18%          +3.53
  unk (probes)      0.00%          0.00%           unchanged
  gates             -              t1a PASS, t1b FAIL (needed >=15.4), t1c PASS
```
The extra context helps (accuracy up, perplexity down, still fluent) but the improvement is under
the pre-registered bar: 14.99 < 15.4. **Verdict recorded as FAIL, gate not moved.** The talk model
gains +1.6 pts from a longer window — that is a real but small effect; window is not the blocker.

### T2 pre-registered (run next — ONE change from the new best baseline)
Baseline: t1_w32_ep24 (W=32, d=128, val 14.99%). Change: width d 128 -> 256. Keep W=32, data,
vocab, lr 0.4 decay 0.5, lam 0.3, 24 epochs, seed 0, gpu, assistant-only.
```
  t2a  unk <= 1% on the shared probes      (no collapse)
  t2b  last-epoch val_acc >= 17.0%         (2+ pts over the 14.99 twin — same margin rule as T1)
  t2c  generation still fluent multi-word prose
  ALL THREE -> width was part of the blocker; candidate for shipping talk.
  Any gate FAIL -> width is not the blocker either; next lever is DATA (scale), then stop-and-report.
```
If T2 also fails, the tiny-model talk line is capacity/data-limited, and that conclusion will be
reported with the full ladder of numbers (16/32 windows x 128/256 widths), not asserted.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("T1 recorded as FAIL (+1.61 real but < bar); T2 pre-registered (d 128->256)")