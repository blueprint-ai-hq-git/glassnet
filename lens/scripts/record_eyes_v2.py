"""record_eyes_v2.py — eyes experiment final: resolution is NOT the bottleneck, measured twice.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "eyes", "eyes56.json")))
r = d["res"]

LOG = f"""
---

## 2026-10-08 — EYES v2 FINAL: SHARPER EYES DID NOT HELP ON THIS TASK FAMILY. ALL GATES FAIL.

Honest design this time: ONE 56px source; the 28 arm sees the PHYSICAL 2x2 mean pool. Same images,
two resolutions, same architecture, gates fixed before running.

```
  28px: overall {r['28']['overall']:.2f}±{r['28']['overall_std']:.2f}  detail {r['28']['detail']:.2f}  size-indep {r['28']['size_indep']:.2f}
  56px: overall {r['56']['overall']:.2f}±{r['56']['overall_std']:.2f}  detail {r['56']['detail']:.2f}  size-indep {r['56']['size_indep']:.2f}

  e1 sharper better overall (+5):   {r['56']['overall']-r['28']['overall']:+.2f}  FAIL
  e2 sharper better on detail(+10): {r['56']['detail']-r['28']['detail']:+.2f}  FAIL
  e3 sharper costs nothing:                    {r['56']['size_indep']-r['28']['size_indep']:+.2f}  FAIL (inside noise)
```

And the 56 arm is the BIGGER model (patch projections 196->64 vs 49->64; 25,583 vs ~16k params).
Bigger, sharper, and not better.

### WHAT THIS ACTUALLY SHOWS

* Binary shapes survive 2x2 pooling: a 2px ring at 56 is a visible 1px ring at 28. The "detail"
  classes were never detail-critical against a 28px camera. v1's 100% at 28 was not an accident of
  native generation — pooling genuinely preserves these distinctions.
* All three deltas (±3 points) sit INSIDE the ±8-10 seed noise. Both single-seed arms would have
  "proven" whatever the seed lottery said. 3 seeds called it: **no measured resolution effect here.**
* This matches the OTHER two independent findings today (p1 miss at see+talk big rung; concept-layer
  ceiling at image/medical lines): **the binding constraint on this task family is the concept layer
  and the task's label structure, not pixels, not data volume, not parameter count.**

### WHAT IT MEANS FOR THE VIDEO COMPLAINT

The blocky look is real (784 pixels stretched 15x) and the frame now LABELS that honestly. But
"sharper eyes" was bought as a fix and the measurement says it would not have bought accuracy on
tasks like this one. Where resolution genuinely matters is REAL photographs with small objects and
open-vocabulary captions — which is exactly the next dataset step (COCO), not more synthetic eyes.

Artifacts kept: eyes56_v1_native.json (design flaw), eyes56.json (honest run, all gates FAIL).
No rewording anywhere: this experiment's answer is "no effect measured", written before any spin.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
print("=" * 90)
print(f"  28: {r['28']['overall']:.2f}±{r['28']['overall_std']:.2f}  56: {r['56']['overall']:.2f}"
      f"±{r['56']['overall_std']:.2f}  e1/e2/e3 all FAIL — no resolution effect measured")
print("=" * 90)
