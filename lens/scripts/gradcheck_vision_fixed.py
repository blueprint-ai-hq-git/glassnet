"""gradcheck_vision_fixed.py — numerical gradient check for the new reshape op + image network.

PRECISION, STATED HONESTLY
--------------------------
The `lens` kernels are float32-only (the Tensor constructor casts everything to float32). In
float32 the precision floor near 1.0 is ~1e-7, so:

    * eps = 1e-6 is BELOW the noise floor -- the finite difference reads rounding, not gradient.
    * an absolute tolerance of 1e-6 is not achievable by any float32 computation.

My first version of this check used eps=1e-6 with an absolute 1e-6 bar and reported FAIL on a
correct gradient. That was my TEST being wrong, not the code -- the fifth time in this project that
a claim was attached to a measurement that could not support it.

The project's own L1 gate already documents the right convention for this engine:
    eps = 1e-2, relative tolerance 1e-2
This file uses it, and shows WHY rather than asserting it.
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import lens.tensor as T                                   # noqa: E402
from lens import backend as B                             # noqa: E402
from lens.vision import ImageConcept, VisionNet           # noqa: E402

EPS = 1e-2          # matches lens/tests/test_l1.py -- float32 engine
TOL = 1e-2          # relative
RESULTS = []


def rel(a, b):
    return abs(a - b) / max(abs(a) + abs(b), 1e-12)


def report(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} {detail}")


print("=" * 100)
print(f"GRADIENT CHECK — float32 engine, eps {EPS:g}, relative tolerance {TOL:g}")
print("  (the same convention as the project's own L1 gate)")
print("=" * 100)
T.set_device("cpu")
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. the NEW reshape op
x0 = rng.standard_normal((4, 6)).astype(np.float32)
Wc = rng.standard_normal((2, 12)).astype(np.float32)


def f_reshape(xx):
    xt = T.Tensor(xx, requires_grad=True)
    return float(T.sum_all(T.mul(T.reshape(xt, (2, 12)), T.Tensor(Wc))).data)


xt = T.Tensor(x0.copy(), requires_grad=True)
T.sum_all(T.mul(T.reshape(xt, (2, 12)), T.Tensor(Wc))).backward()
an = np.asarray(B.to_cpu(xt.grad)).astype(np.float64).reshape(-1)
shape = np.asarray(xt.data).shape
flat = np.asarray(B.to_cpu(xt.data)).astype(np.float32).reshape(-1).copy()

worst = 0.0
for k in range(flat.size):
    o = flat[k]
    flat[k] = np.float32(o + EPS)
    lp = f_reshape(flat.reshape(shape))
    flat[k] = np.float32(o - EPS)
    lm = f_reshape(flat.reshape(shape))
    flat[k] = o
    worst = max(worst, rel(an[k], (lp - lm) / (2 * EPS)))
print(f"    forward (4,6) -> (2,12); checked all {flat.size} entries")
report("reshape gradient", worst <= TOL, f"worst rel diff {worst:.3e}")

# ---------------------------------------------------------------- 2. the whole VisionNet


def _c(name, fn):
    return ImageConcept(name, fn, support=0.5, formula="test")


concepts = [
    _c("bright", lambda X: (np.asarray(X).mean(axis=1) > 0.5).astype(np.float32)),
    _c("dark", lambda X: (np.asarray(X).mean(axis=1) < 0.5).astype(np.float32)),
    _c("spread", lambda X: (np.asarray(X).std(axis=1) > 0.2).astype(np.float32)),
]
imgs = rng.random((8, 784)).astype(np.float32)
y = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

net = VisionNet(concepts, n_classes=8, d=8, patch=7, seed=0, lam_concept=0.3)
for p in net.parameters():
    p.grad = None
loss, parts = net.loss(imgs, y)
loss.backward()

worst2 = 0.0
n_checked = 0
missing = []
for p in net.parameters():
    if p.grad is None:
        missing.append(p.name)
        continue
    an_p = np.asarray(B.to_cpu(p.grad)).astype(np.float64)
    shp = np.asarray(B.to_cpu(p.data)).shape
    saved = np.asarray(B.to_cpu(p.data)).astype(np.float32).copy()
    flatp = saved.reshape(-1).copy()
    take = min(24, flatp.size)
    sel = np.linspace(0, flatp.size - 1, take).astype(int)
    for k in sel:
        o = flatp[k]
        flatp[k] = np.float32(o + EPS)
        p.data = flatp.reshape(shp)
        lp = float(net.loss(imgs, y)[0].data)
        flatp[k] = np.float32(o - EPS)
        p.data = flatp.reshape(shp)
        lm = float(net.loss(imgs, y)[0].data)
        flatp[k] = o
        worst2 = max(worst2, rel(an_p.reshape(-1)[k], (lp - lm) / (2 * EPS)))
        n_checked += 1
    p.data = saved

print(f"    checked {n_checked} entries across {len(net.parameters())} parameter tensors")
print(f"    concept auxiliary loss active: {'concept_aux' in parts}  "
      f"(value {parts.get('concept_aux', 0):.6f})")
if missing:
    report("VisionNet gradients", False, f"NO GRADIENT for {missing}")
else:
    report("VisionNet gradients", worst2 <= TOL, f"worst rel diff {worst2:.3e}")

# ---------------------------------------------------------------- 3. why the bar is 1e-2
print()
print("  --- why the tolerance is 1e-2, shown instead of asserted ---")
print("  the engine casts every tensor to float32, so a 1e-6 step is swallowed by rounding:")
print(f"    float32(1.0) + 1e-6 == float32(1.0) ?   {np.float32(1.0 + 1e-6) == np.float32(1.0)}")
print(f"    smallest step that changes a float32 near 1.0: {np.spacing(np.float32(1.0)):.3e}")
print(f"    so eps must be well above {np.spacing(np.float32(1.0)):.0e}; this check uses {EPS:g}")

print()
alls = all(ok for _, ok, _ in RESULTS)
print("=" * 100)
print(f"  VERDICT: {'PASS — the new op and the image network are gradient-correct' if alls else 'FAIL'}")
print("=" * 100)
