"""gradcheck_wce.py — numerical gradient check for the NEW class-weighted cross-entropy.

The project rule: check every gradient numerically before trusting it. `softmax_ce` gained a
`class_weights` argument so the medical run cannot collapse onto the majority answer, and a new
gradient path is a new thing to break.

The engine is float32, so the check follows the project's own convention (lens/tests/test_l1.py):
    eps = 1e-2, relative tolerance 1e-2.

It also verifies the weighted loss against an INDEPENDENT NumPy reference computed from the formula,
because a finite difference on a strongly curved function has a truncation floor -- the same reason
the L1 gate checks softmax against a reference rather than only against central differences.
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)

import lens.tensor as T                                   # noqa: E402
from lens import backend as B                             # noqa: E402

EPS, TOL = 1e-2, 1e-2
RESULTS = []


def rel(a, b):
    return abs(a - b) / max(abs(a) + abs(b), 1e-12)


def report(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<38} {detail}")


T.set_device("cpu")
rng = np.random.default_rng(0)
C = 5
N = 12
logits0 = rng.standard_normal((N, C)).astype(np.float32)
labels = rng.integers(0, C, size=N).astype(np.int32)
# deliberately UNBALANCED labels, so the weights actually differ
labels[:7] = 0
W = np.asarray([0.3, 1.0, 2.0, 3.0, 0.5], dtype=np.float32)

print("=" * 100)
print(f"WEIGHTED CROSS-ENTROPY — float32 engine, eps {EPS:g}, relative tolerance {TOL:g}")
print("=" * 100)
print(f"  labels {labels.tolist()}")
print(f"  class weights {W.tolist()}")

# ---------------------------------------------------------------- 1. loss vs an independent formula
xt = T.Tensor(logits0.copy(), requires_grad=True)
loss = T.softmax_ce(xt, labels, class_weights=W)
z = logits0 - logits0.max(axis=1, keepdims=True)
p = np.exp(z)
p = p / p.sum(axis=1, keepdims=True)
w_s = W[labels]
w_s = w_s / w_s.mean()
ref = float((-(w_s * np.log(p[np.arange(N), labels] + 1e-12))).mean())
report("loss vs independent numpy reference", abs(float(loss.data) - ref) < 1e-5,
       f"engine {float(loss.data):.7f}  reference {ref:.7f}  diff {abs(float(loss.data)-ref):.2e}")

# ---------------------------------------------------------------- 2. gradient vs the formula
loss.backward()
an = np.asarray(B.to_cpu(xt.grad)).astype(np.float64)
ref_d = p.copy()
ref_d[np.arange(N), labels] -= 1.0
ref_d = ref_d * w_s[:, None] / N
d = np.abs(an - ref_d).max()
report("gradient vs independent numpy reference", d < 1e-6, f"max abs diff {d:.3e}")

# ---------------------------------------------------------------- 3. gradient vs finite difference
worst = 0.0
flat = logits0.reshape(-1).copy()
shape = logits0.shape
for k in range(flat.size):
    o = flat[k]
    flat[k] = np.float32(o + EPS)
    lp = float(T.softmax_ce(T.Tensor(flat.reshape(shape)), labels, class_weights=W).data)
    flat[k] = np.float32(o - EPS)
    lm = float(T.softmax_ce(T.Tensor(flat.reshape(shape)), labels, class_weights=W).data)
    flat[k] = o
    worst = max(worst, rel(an.reshape(-1)[k], (lp - lm) / (2 * EPS)))
report("gradient vs central difference", worst <= TOL, f"worst rel diff {worst:.3e}")

# ---------------------------------------------------------------- 4. unweighted path unchanged
xt2 = T.Tensor(logits0.copy(), requires_grad=True)
l2 = T.softmax_ce(xt2, labels)
l2.backward()
p2 = p
g2 = np.asarray(B.to_cpu(xt2.grad)).astype(np.float64)
r2 = p2.copy()
r2[np.arange(N), labels] -= 1.0
r2 /= N
d2 = np.abs(g2 - r2).max()
report("unweighted path unchanged", d2 < 1e-6, f"max abs diff {d2:.3e}")

# ---------------------------------------------------------------- 5. the point of the change
print()
print("  --- does the weighting actually change the balance? ---")
print(f"    labels: {np.bincount(labels, minlength=C).tolist()}  (class 0 holds "
      f"{(labels == 0).mean()*100:.0f}% of samples)")
unw = float(T.softmax_ce(T.Tensor(logits0), labels).data)
wgt = float(T.softmax_ce(T.Tensor(logits0), labels, class_weights=W).data)
print(f"    unweighted loss {unw:.6f}   weighted loss {wgt:.6f}")
print(f"    -> the rare classes now carry more of the loss, which is the point.")

print()
alls = all(ok for _, ok, _ in RESULTS)
print("=" * 100)
print(f"  VERDICT: {'PASS — weighted CE is gradient-correct and the unweighted path is untouched' if alls else 'FAIL'}")
print("=" * 100)
