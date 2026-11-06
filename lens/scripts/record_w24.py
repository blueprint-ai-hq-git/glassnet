"""record_w24.py — context-length experiment: both gates FAIL, recorded honestly per pre-registration.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
v = json.load(open(os.path.join(P, "lens", "runs", "chat", "w24_verdict.json")))

LOG = f"""
---

## 2026-10-09 — EXPERIMENT w24 (context W=16 -> 24, V=12000): BOTH GATES FAIL

One change from v12k: context window 16 -> 24, same recipe (d=128, V=12000, 24 epochs, lr 0.4
decay 0.5, 1.5M windows). Gates were fixed BEFORE the run in run_w24.sh.

```
  model           V      W   val_ppl   bits/token   <unk>
  v12k          12,000  16    98.00       6.61        3.2%
  w24_v12000    12,000  24    99.24       6.63        8.5%

  w1  ppl improvement -1.3%   (bar >3%)          FAIL
  w2  bits/token delta +0.018 (bar -0.05)        FAIL
```

### VERDICT: W=24 does NOT help this model at this data size. Recorded as FAIL per the
pre-registered rule, which says "report and STOP adding context — do not keep stacking changes
hoping something moved." Context length is CLOSED as a direction for now.

The <unk> rate ROSE from 3.2% to 8.5% (-5.3 pts), so longer context made output worse on the
quality axis too, not just the ppl axis. Sample output also shows more repetition ("- - - - -") and
more artifacts ("/ / ha"). Three independent measurements agree: W=16 is better than W=24 here.

### WHY (hypothesis, stated as hypothesis, not asserted as fact)
W=24 means each prediction conditions on twice as much context per window; with the same 1.5M
windows and 24 epochs, the model effectively sees the same amount of training signal but must
learn a harder task (longer-range dependencies). It may simply be under-trained relative to W=16
at this budget. This is a plausible reason, NOT an excuse — the gates were binary and they failed.
If context is revisited, it needs either more epochs or a larger window budget, not another
incremental bump.

### WHAT THIS MEANS FOR THE TALK MODEL
Current best talk artifact: **v12k** (V=12000, W=16, ppl 98.00, bpt 6.61, unk 3.2%).
best1 remains the perplexity-optimal model (bpt 6.04, unk 14.5%).
Next direction is NOT more vocab and NOT more context — it needs a different mechanism entirely.
Candidate: constrained decoding for the code model (same mechanism, applied to talk), or a
quality-focused loss. Each will be pre-registered and gated the same way.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended w24 record (both gates FAIL)")
print(f"  w1 FAIL: ppl {v['base']['ppl']:.2f} -> {v['new']['ppl']:.2f} (-1.3%, bar >3%)")
print(f"  w2 FAIL: bpt {v['base']['bpt']:.2f} -> {v['new']['bpt']:.2f} (+0.018, bar -0.05)")
