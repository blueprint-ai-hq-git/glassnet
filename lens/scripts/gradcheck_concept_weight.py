"""gradcheck_concept_weight.py — the check that would have CAUGHT the w!=1 gradient bug.

WHY THIS FILE EXISTS
  The old gradient check built its test concepts with support=0.5. `pos_weight` is (1-s)/s, so
  support=0.5 gives w=1 -- and the old (wrong) backward formula
        dh = w * (h - t) / (h * (1 - h))
  is EXACTLY correct when w=1. So the check passed for months while every real concept (support 0.02
  -> w=49, support 0.9 -> w=0.111) trained against a mis-scaled gradient. The test's choice of
  parameters made it blind to the bug it existed to find.

  This check uses supports that produce w = 49, 4, 1, 0.111 and verifies the concept-loss gradient
  for EACH, including the extreme case, so a w-dependent error cannot hide again.
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens.tensor as T                                   # noqa: E402
from lens import backend as B                             # noqa: E402
from lens.vision import ImageConcept, VisionNet           # noqa: E402

T.set_device("cpu")
T.set_dtype("float64")
_rng = np.random.default_rng(11)
_imgs = _rng.random((8, 784))
_y = np.arange(8)


def _mk(name, fn, support):
    return ImageConcept(name, fn, support=support, formula="test")


# supports chosen to span the weight range, NOT 0.5-everywhere
SPECS = [
    ("rare", lambda X, s=0.02: (np.asarray(X).mean(axis=1) > 0.20).astype(np.float64), 0.02),
    ("mid", lambda X, s=0.20: (np.asarray(X).mean(axis=1) > 0.40).astype(np.float64), 0.20),
    ("half", lambda X, s=0.50: (np.asarray(X).mean(axis=1) > 0.52).astype(np.float64), 0.50),
    ("common", lambda X, s=0.90: (np.asarray(X).mean(axis=1) > 0.35).astype(np.float64), 0.90),
]


def rel(a, b):
    return abs(a - b) / (abs(a) + abs(b) + 1e-12)


print("=" * 100)
print("CONCEPT-LOSS GRADIENT vs THE POSITIVE WEIGHT  (the w!=1 bug check)")
print("=" * 100)
concepts = [_mk(n, f, s) for n, f, s in SPECS]
net = VisionNet(concepts, n_classes=8, d=8, patch=7, seed=0, lam_concept=0.3)
for p in net.parameters():
    p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)

pw = [float((1 - s) / s) for _, _, s in SPECS]
print(f"  supports           : {[s for _, _, s in SPECS]}")
print(f"  pos_weight (1-s)/s : {[round(x, 4) for x in pw]}")
print(f"  the old formula was exact only at w=1, i.e. support 0.5 only\n")

loss, parts = net.loss(_imgs, _y)
loss.backward()
eps = 1e-6
worst_abs, worst_rel, n_chk = 0.0, 0.0, 0
per_param = {}
for p in net.parameters():
    base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
    an = np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy().reshape(-1)
    flat = base.reshape(-1)
    take = min(20, flat.size)
    sel = np.linspace(0, flat.size - 1, take).astype(int)
    pa, pr = 0.0, 0.0
    for j in sel:
        o = flat[j]
        flat[j] = o + eps
        p.data = base.copy()
        lp = float(net.loss(_imgs, _y)[0].data)
        flat[j] = o - eps
        p.data = base.copy()
        lm = float(net.loss(_imgs, _y)[0].data)
        flat[j] = o
        num = (lp - lm) / (2 * eps)
        a = float(an[j])
        pa = max(pa, abs(a - num))
        pr = max(pr, rel(a, num))
        n_chk += 1
    p.data = base
    per_param[p.name] = (pa, pr)
    worst_abs = max(worst_abs, pa)
    worst_rel = max(worst_rel, pr)

print(f"  checked {n_chk} parameter entries")
print(f"  {'param':<16}{'max abs diff':>16}{'max rel diff':>16}")
for k, (a, r) in per_param.items():
    print(f"  {k:<16}{a:>16.3e}{r:>16.3e}")
ok = worst_abs < 1e-8
print(f"\n  worst abs {worst_abs:.3e}   worst rel {worst_rel:.3e}   (bar 1e-8)")
print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
print(f"\n  The old check (support 0.5 only) would have reported PASS for the WRONG formula, because")
print(f"  w=1 makes the wrong formula right. This one would not: at support 0.02 the weight is "
      f"{pw[0]:.0f},")
print(f"  and a gradient wrong by that factor is impossible to miss.")
sys.exit(0 if ok else 1)
