#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY' 2>&1 | tail -12
import sys, os
sys.path.insert(0, "."); sys.path.insert(0, "lens/scripts")
import numpy as np
import lens.tensor as T; T.set_device("cpu")
from lens.vision import load_vision

# 1) old checkpoints still load (base SPEC only)
net, meta = load_vision("lens/runs/image/lad_image_cls_ep24.npz")
print("  ladder ep24:", type(net).__name__, "K", len(net.concepts), "loads OK")

# 2) extended K=23 real checkpoint loads through the frozen registry
net2, meta2 = load_vision("lens/runs/image/ktest_real_s0_ep12.npz")
print("  ktest_real s0:", "K", len(net2.concepts), "loads OK")

# 3) forward + explain on one test image through the extended model
import vl_data
d = vl_data.prepare_vqa("fashion", seed=0)
out = net2.forward_arrays(d["Xte"][:8])
h = out["concepts"][0]
fired = [net2.concepts[i].name for i in range(len(net2.concepts)) if h[i] > 0.5]
print("  forward OK; fired:", fired[:5])

# 4) hand-recompute last layer == engine (explain identity on the see-only model)
X = d["Xte"][:32]
o = net2.forward_arrays(X)
Pn = net2.front.patches(X.astype(np.float32))
from lens import backend as B
zd = (Pn.reshape(-1, Pn.shape[-1]) @ B.to_cpu(net2.front.proj.W.data)
      + B.to_cpu(net2.front.proj.b.data)).reshape(len(X), -1)
cl = zd @ B.to_cpu(net2.concept.W.data) + B.to_cpu(net2.concept.b.data)
h3 = 1/(1+np.exp(-np.clip(cl, -60, 60)))
manual = h3 @ B.to_cpu(net2.answer.W.data) + B.to_cpu(net2.answer.b.data)
print(f"  explain identity residual {float(np.abs(manual - o['logits']).max()):.2e} (bar 1e-5)")

# 5) dummy checkpoints must REFUSE: random projections have no meaning to resolve
try:
    load_vision("lens/runs/image/ktest_dummy_s0_ep12.npz")
    print("  dummy load: LOADED (registry too permissive?)")
except (KeyError, ValueError) as e:
    print("  dummy load REFUSED as expected:", str(e)[:70])
PY
bash lens/scripts/run_selfcheck.sh 2>&1 | tail -2
