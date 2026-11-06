"""dbg_concept_shapes.py — why do concept label shapes disagree?
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import vl_data                                            # noqa: E402
import vl_data3                                           # noqa: E402
import image_concepts                                     # noqa: E402
import colour_concepts                                    # noqa: E402
from lens.vision import load_vision                       # noqa: E402

base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0, color=True)
d = vl_data3.prepare(base, "medical", seed=0)
print("  Xtr", np.asarray(d["Xtr"]).shape, " Xte", np.asarray(d["Xte"]).shape)

ckept, cspec = colour_concepts.build(np.asarray(d["Xtr"], dtype=np.float32),
                                     min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in ckept]

R = os.path.join(PROJ, "lens", "runs", "vl")
cands = sorted([p for p in os.listdir(R) if p.startswith("mlow_lc0.00_ep")],
               key=lambda s: int(s.split("ep")[1].split(".")[0]))
net, meta = load_vision(os.path.join(R, cands[-1]), device="cpu")

X = np.asarray(d["Xte"][:64], dtype=np.float32)
print(f"\n  model has {len(net.concepts)} concepts; input slice {X.shape}")
print(f"  {'concept':<28}{'side':<7}{'label shape':<14}")
bad = 0
for c in net.concepts:
    try:
        lab = c.labels(X)
        shp = str(np.asarray(lab).shape)
        flag = "" if np.asarray(lab).shape == (len(X),) else "  <-- WRONG SHAPE"
        if flag:
            bad += 1
    except Exception as e:
        shp = f"ERROR {type(e).__name__}"
        flag = f"  <-- {e}"
        bad += 1
    side = "text" if hasattr(c, "ids") and getattr(c, "ids", None) else "image"
    print(f"  {c.name[:27]:<28}{side:<7}{shp:<14}{flag}")
print(f"\n  {bad} concept(s) do not return shape ({len(X)},)")
