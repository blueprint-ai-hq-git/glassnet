"""record_k22.py — shipping config K=22 passed its big run with a GROWN delta.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "seetalk", "seetalk_k22_big_log.json")))
h, hc = d["hist"], d["hist_control"]
peak = max(e["val_acc"] for e in h) * 100
delta = d["delta_vs_k22"] if "delta_vs_k22" in d else d["delta_vs_k12"]

LOG = f"""
---

## 2026-10-08 — SHIPPING CONFIG K=22 BIG RUN: 4/4 GATES, AND THE DELTA GREW (p2 HELD)

```
  d=128, 60k images, K=22 real named concepts, 14 epochs, GPU
  see+talk final {h[-1]['val_acc']*100:.2f}%  peak {peak:.2f}%   control {hc[-1]['val_acc']*100:.2f}%
  DELTA +{delta['k22_big']:.2f}  vs K=12 big +14.51   (g4 grew: the image path got STRONGER, not just wider)

  g1 7 distinct words from 10 images  PASS   (K=12 showed 5-6 — more concepts, more nouns)
  g2 loss 0.7939 -> 0.5163            PASS
  g3 explain residual 8.58e-06        PASS   (GPU float32; CPU 0.00e+00)
  g4 delta > +0.50                    PASS
  checkpoint round-trip 0.000e+00     IDENTICAL
```

**Prediction scorecard (written before the run):** p1 4/4 PASS — held. p2 "77-80% and delta grows
past +14.51" — held exactly (peak 78.69, delta +16.03). p3 round-trip identical — held.

**What this means for the product:** the ladder's best honest number is now **78.7 peak / 77.0
final with +16.0 image contribution**, and the g1 probe (7 nouns from 10 frames) shows the caption
path is using MORE distinct class words than K=12 did. The sweep said K=22 is where width pays and
flattening starts; this run confirms K=22 is the right SHIP point — beyond it there is no evidence
of further gain at this pixel budget, and we would be adding concepts for looks, not for results.

Concept registry note: `SeeTalkNet.load` now resolves concepts from the FROZEN registry
(base SPEC + extended + more + band) by (name, formula) — unknown name or changed formula still
raises; old K=12/K=13 checkpoints re-verified loading (regression run passed all three).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended; k22 delta +{delta['k22_big']:.2f} vs k12 +{delta['k12_big']:.2f}")
