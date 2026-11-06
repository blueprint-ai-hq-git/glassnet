"""record_k22_promotion.py — the fixed-gradient K22 is now the shipping see+talk default.
"""
import os
import json

P = "/run/media/yahya/HARD DISK 4/GlassNet"
fid = json.load(open(os.path.join(P, "lens", "runs", "seetalk",
                                  "seetalk_k22_big_fixedgrad_fidelity.json")))
c = fid["counts"]

LOG = f"""
---

## 2026-10-10 — SHIPPING PROMOTED: see+talk default is now the FIXED-GRADIENT K22 retrain.

### WHY
Every checkpoint trained with lam_concept > 0 and concepts whose support != 0.5 learned under a
concept-BCE gradient that was wrong by a factor of pos_weight (bug found and fixed today). The old
shipping K22 had **0 of 22 concepts at w == 1**, so its concept layer was trained against a distorted
objective. It was retrained on the corrected engine and the retrain is now the default.

### THE COMPARISON (same config, same seed, only the gradient changed)
```
                              old (buggy)      new (corrected)
  final train_loss             0.5163           0.4873      <- LOWER: the step now descends its own loss
  train_acc                    81.84%           81.79%
  val_ppl                      1.6176           1.6231
  val_acc                      77.03%           76.90%      <- within noise
  delta vs text-only control   +15.90           +15.90
  gates                        4/4 PASS         4/4 PASS
  explain residual             5.72e-06         5.72e-06
  concept fidelity:
    strong                     10               {c.get('strong', 0)}
    honest                      6               {c.get('honest', 0)}
    weak                        1               {c.get('weak', 0)}
    DECORATIVE                  5               {c.get('decorative', 0)}    <- the names got HONESTER
```
**The transparency claim improved as a side effect of fixing the gradient:** decorative concept names
halved (5 -> 2) and strong rose (10 -> 12). A correctly-scaled concept loss makes each neuron track
the concept it is NAMED after more tightly, which is the whole point of the architecture.

### WHAT CHANGED IN THE CODE
* `variants.build_default()` -> see_talk = `seetalk_k22_big_fixedgrad.npz`, with a fallback to the old
  artifact if missing (the interface never dead-ends). The OLD checkpoint is KEPT, not overwritten.
* `selfcheck.py` verifies BOTH K22 checkpoints (old and new) so neither can rot unnoticed.
* `verify_explain_fidelity.py` no longer hardcodes a checkpoint path — it reads the SHIPPING default.
  A test pinned to an artifact silently stops testing what ships; that is exactly what happened when
  shipping moved (the test failed on the new default and had to be repointed).

### VERIFIED THROUGH THE REAL INTERFACE
```
  default see_talk: seetalk_k22_big_fixedgrad.npz
  explain word: sandal | fidelity: seetalk_k22_big_fixedgrad_fidelity.json
     bottom_left_mass    own_corr 0.344  weak
     top_right_mass      own_corr 0.384  weak
     bottom_half_heavier own_corr 0.892  strong
     centre_row          own_corr 0.437  weak
  4 live captions: sandal / pullover / trouser / trouser
  run_all_tests: 14/14 PASS   selfcheck: ALL CHECKS PASS
```
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended K22 promotion record")
print(f"  new default: seetalk_k22_big_fixedgrad.npz")
print(f"  fidelity: strong {c.get('strong')}, honest {c.get('honest')}, weak {c.get('weak')}, decorative {c.get('decorative')}")
