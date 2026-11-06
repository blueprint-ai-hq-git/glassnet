"""dbg_gradcheck_vl.py — find WHERE the VisionNet gradient check disagrees and WHY.

The full check reports worst |analytic - numeric| = 7e-3 with worst relative = 1.0 exactly.
rel == 1.0 means |a - n| == |a| + |n|: either the two are OPPOSITE signs or one is zero. abs 7e-3
is large for float64. This script computes the analytic gradient for EVERY parameter entry (no
sampling), then numeric for the WORST few, printing the actual values so the bug is visible.
"""
import os, sys
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P); sys.path.insert(0, P+"/lens/scripts")
import lens.tensor as T
from lens import backend as B
from lens.vision import ImageConcept, VisionNet

T.set_device("cpu"); T.set_dtype("float64")
rel = lambda a, b: abs(a-b) / (abs(a) + abs(b) + 1e-12)
rng = np.random.default_rng(3)
eps = 1e-6
imgs = rng.random((8, 784)).astype(np.float64)

def _c(name, fn):
    return ImageConcept(name, fn, support=0.5, formula="test")

concepts = [
    _c("bright", lambda X: (np.asarray(X).mean(axis=1) > 0.5).astype(np.float64)),
    _c("dark", lambda X: (np.asarray(X).mean(axis=1) < 0.5).astype(np.float64)),
    _c("big", lambda X: (np.asarray(X).std(axis=1) > 0.2).astype(np.float64)),
]
y = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
net = VisionNet(concepts, n_classes=8, d=8, patch=7, seed=0, lam_concept=0.3)
for p in net.parameters():
    p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)

loss, parts = net.loss(imgs, y)
loss.backward()

# full analytic grads for every entry
worst = []  # (rel, abs, name, flat_idx, a, num)
all_grads = []
for p in net.parameters():
    an = np.asarray(B.to_cpu(p.grad), dtype=np.float64).ravel()
    all_grads.append((p, an))
    print(f"  {p.name:<20} grad absmax {np.abs(an).max():.3e}  n={an.size}")

# numeric for the entries with the LARGEST analytic magnitude is not the point;
# the rel==1.0 ones are entries where analytic ~= 0 or mixed sign. Check ALL entries of the
# worst param tensors but cap at ~120 perturbations to stay fast.
ntot = 0
for p, an in all_grads:
    base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
    flat = base.reshape(-1)
    take = min(120, flat.size)
    sel = np.linspace(0, flat.size - 1, take).astype(int)
    for j in sel:
        o = flat[j]
        flat[j] = o + eps; p.data = base.copy()
        lp = float(net.loss(imgs, y)[0].data)
        flat[j] = o - eps; p.data = base.copy()
        lm = float(net.loss(imgs, y)[0].data)
        flat[j] = o
        num = (lp - lm) / (2 * eps)
        a = float(an[j])
        worst.append((rel(a, num), abs(a - num), p.name, j, a, num))
        ntot += 1
    p.data = base

worst.sort(key=lambda t: -t[0])
print(f"\n  checked {ntot} entries")
print(f"  worst 15 by relative difference:")
print(f"  {'rel':>10} {'abs':>10} {'param':<14} {'idx':>5} {'analytic':>14} {'numeric':>14}")
for rel_v, abs_v, name, j, a, num in worst[:15]:
    print(f"  {rel_v:>10.3e} {abs_v:>10.3e} {name:<14} {j:>5} {a:>14.3e} {num:>14.3e}")
w = worst[0]
print(f"\n  worst: {w[2]}[{w[3]}]  analytic {w[4]:.6e}  numeric {w[5]:.6e}")
print(f"  -> opposite signs? {np.sign(w[4]) != np.sign(w[5])}")