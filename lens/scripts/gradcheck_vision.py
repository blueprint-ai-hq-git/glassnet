"""gradcheck_vision.py — numerical gradient check for the NEW reshape op and the whole image net.

The project rule: "Check every gradient with a numerical gradient test before trusting it."

`reshape` was added to the tensor engine for the image front-end, so it is checked here on its own
AND as part of the full VisionNet, against central differences:

    dL/dx ~= (L(x + eps) - L(x - eps)) / (2 eps)

Run in float64 so the finite difference is not swamped by float32 rounding.
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import lens.tensor as T                                   # noqa: E402
from lens import backend as B                             # noqa: E402
from lens.model import Dense                              # noqa: E402
from lens.vision import ImageConcept, VisionNet           # noqa: E402


def rel(a, b):
    d = abs(a - b)
    return d / max(abs(a) + abs(b), 1e-12)


def numeric_grad(fn, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        o = x[i]
        x[i] = o + eps
        lp = fn()
        x[i] = o - eps
        lm = fn()
        x[i] = o
        g[i] = (lp - lm) / (2 * eps)
    return g


print("=" * 100)
print("GRADIENT CHECK — the NEW reshape op, on its own")
print("=" * 100)
T.set_device("cpu")
# THE FIX (2026-10-10): the engine used to cast every input to float32 no matter what, so the
# float64 numeric check below could never pass — eps=1e-6 is below float32's resolution and the
# "numeric" gradient came back quantised (steps of 0.119), differing from the analytic gradient by
# up to 1.0. set_dtype("float64") makes this script's float64 claim TRUE. set_dtype(None) after the
# run restores the float32 training default.
T.set_dtype("float64")
rng = np.random.default_rng(0)
x0 = rng.standard_normal((4, 6)).astype(np.float64)
Wc = rng.standard_normal((2, 12)).astype(np.float64)


def f_reshape(xx):
    """A scalar function of the (4,6) input: reshape it, weight it, sum it."""
    xt = T.Tensor(xx, requires_grad=True)
    yt = T.reshape(xt, (2, 12))
    return float(T.sum_all(T.mul(yt, T.Tensor(Wc))).data)


xt = T.Tensor(x0.copy(), requires_grad=True)
yt = T.reshape(xt, (2, 12))
T.sum_all(T.mul(yt, T.Tensor(Wc))).backward()
an = np.asarray(xt.grad)
num = numeric_grad(lambda: f_reshape(x0), x0)
print(f"  forward shape (4,6) -> (2,12): {yt.shape}")
print(f"  max |analytic - numeric| = {np.abs(an - num).max():.3e}")
print(f"  max relative difference  = {max(rel(a, b) for a, b in zip(an.ravel(), num.ravel())):.3e}")
ok1 = np.abs(an - num).max() < 1e-6
print(f"  VERDICT: {'PASS' if ok1 else 'FAIL'}")
print()

print("=" * 100)
print("GRADIENT CHECK — the whole VisionNet (patch projection -> concepts -> answer)")
print("=" * 100)
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
# force float64 params so the finite difference is meaningful
for p in net.parameters():
    p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)

loss, parts = net.loss(imgs, y)
loss.backward()

worst_abs, worst_rel, n_checked, bad = 0.0, 0.0, 0, []
for p in net.parameters():
    if p.grad is None:
        bad.append(f"{p.name}: NO GRADIENT")
        continue
    base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
    an = np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy()
    flat = base.reshape(-1)
    take = min(25, flat.size)                       # sample 25 entries per parameter tensor
    sel = np.linspace(0, flat.size - 1, take).astype(int)
    for j in sel:
        o = flat[j]
        flat[j] = o + eps
        p.data = base.copy()
        lp = float(net.loss(imgs, y)[0].data)
        flat[j] = o - eps
        p.data = base.copy()
        lm = float(net.loss(imgs, y)[0].data)
        flat[j] = o
        num = (lp - lm) / (2 * eps)
        a = an.reshape(-1)[j]
        worst_abs = max(worst_abs, abs(a - num))
        worst_rel = max(worst_rel, rel(a, num))
        n_checked += 1
    p.data = base

print(f"  checked {n_checked} parameter entries across {len(net.parameters())} tensors")
print(f"  max |analytic - numeric| = {worst_abs:.3e}")
print(f"  max relative difference  = {worst_rel:.3e}")
print(f"  concept auxiliary loss active: {'concept_aux' in parts}  value {parts.get('concept_aux')}")
if bad:
    print("  PROBLEMS:")
    for b in bad:
        print(f"    {b}")
ok2 = worst_abs < 1e-5 and not bad
print(f"  VERDICT: {'PASS' if ok2 else 'FAIL'}")
print()
print("=" * 100)
print(f"  OVERALL: {'PASS — the new op and the image network are gradient-correct' if (ok1 and ok2) else 'FAIL'}")
print("=" * 100)
