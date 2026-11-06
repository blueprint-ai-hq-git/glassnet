"""record_seetalk_big.py — see+talk at the big rung: 4/4, but prediction p1 missed.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "seetalk", "seetalk_big_log.json")))
st, ct = d["see_talk"], d["text_only"]
peak = max(e["val_acc"] for e in st) * 100

LOG = f"""
---

## 2026-10-08 — SEE+TALK BIG RUNG: 4/4 GATES. ONE PREDICTION MISSED.

```
  60,000 images, d=128, 14 epochs, GPU, ~{int(sum(e['seconds'] for e in st))}s train arm
  see_talk  final val {st[-1]['val_acc']*100:.2f}%  peak {peak:.2f}%   params {d['params']['see_talk']:,}
  control   final val {ct[-1]['val_acc']*100:.2f}%               params {d['params']['control']:,}
  delta     +{st[-1]['val_acc']*100-ct[-1]['val_acc']*100:.2f}

  g1 6 distinct nouns from 10 images            PASS
  g2 loss fell 1.2032 -> 0.5681                 PASS
  g3 explain residual 6.68e-06 (GPU float32)    PASS (bar 1e-5; CPU runs were 0.00e+00)
  g4 beats control by >+0.50                    PASS (+{st[-1]['val_acc']*100-ct[-1]['val_acc']*100:.2f})
```

### THE LADDER (see+talk val accuracy vs its own control)

```
  tiny     72.53%  vs 61.06%   +11.47
  small    77.29%  vs 61.72%   +15.58
  medium   77.49%  vs 61.19%   +16.30
  big      75.51% (peak 78.08) vs 61.00%   +14.51
```

### PREDICTION SCORECARD (written before the run)

```
  p1  "val accuracy lands 80-92%"                     MISSED — measured 78.08 peak, 75.51 final
  p2  "control stays 61-62% (template ceiling)"       HELD — 61.00% exactly
  p3  "g1 shows >=6 distinct nouns"                   HELD — 6 nouns
```

**p1's miss is the interesting result: 6x the data did NOT raise the accuracy band.** small/medium/big
all sit in 75-78% and the control never moves off 61%. The ceiling is NOT data and NOT capacity — it
is the TASK: predicting the class noun is capped by how well 12 named concepts can separate 10 classes
of clothing from a 28x28 image, and the val accuracy already includes template words the model gets
right blind. The class-word accuracy the variants smoke measured (77.2% at the noun position) matches
the earlier image-model line (readable 82.5-86.4% full-class, 12-concept layer). **To break the band
the concept layer needs more/better concepts — which is the same finding the image and medical lines
already made, now confirmed from the language side.** No rewording: p1 failed, logged.

### CHECKPOINT

`seetalk_big.npz` saved; round-trip max diff 0.000e+00 (bit-identical); vocab words + concept
names + formulas + supports inside the checkpoint; teacher sha for future distillation runs.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 90)
print(f"  big: see_talk {st[-1]['val_acc']*100:.2f}% (peak {peak:.2f}%) vs control "
      f"{ct[-1]['val_acc']*100:.2f}%  gates 4/4")
print("  p1 MISSED (80-92 predicted) — ceiling is the task/concept layer, not data/capacity")
print("=" * 90)
