"""diag_softmax_grad.py — is the softmax_ce gradient wrong, or is eps just too large?

A central difference has error O(eps^2 * |f'''|) from truncation and O(sigma/eps) from float
noise. Softmax is a very curved function, so with eps=1e-2 in float32 the truncation term can
dominate. The diagnostic: sweep eps. If the error FALLS as eps shrinks, it was truncation (the
gradient is fine). If it stays flat or grows, there is a real bug.

Also compares against a pure-NumPy softmax+CE with the same float32 arithmetic, so the expected
gradient is computed two independent ways.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from lens import tensor as T  # noqa: E402

RNG = np.random.default_rng(0)
rows, V = 32, 10
logits_np = RNG.standard_normal((rows, V)).astype(np.float32)
labels = RNG.integers(0, V, rows).astype(np.int32)

logits = T.Tensor(logits_np.copy(), requires_grad=True, name="logits")
loss = T.softmax_ce(logits, labels)
loss.backward()
analytic = logits.grad.copy()

# independent reference gradient, computed in float64 from the same formula
z = logits_np.astype(np.float64)
z = z - z.max(1, keepdims=True)
e = np.exp(z)
p = e / e.sum(1, keepdims=True)
ref = p.copy()
ref[np.arange(rows), labels] -= 1.0
ref /= rows

print("=" * 88)
print("SOFTMAX_CE GRADIENT — truncation or bug?")
print("=" * 88)

print("\n1. our analytic gradient vs a float64 reference (same formula, independent arithmetic)")
absd = float(np.abs(analytic.astype(np.float64) - ref).max())
print(f"   max abs diff {absd:.3e}   (float32 storage alone explains ~1e-7)")

print("\n2. eps sweep on the finite difference")


def loss_at(perturb=None):
    x = logits_np.copy()
    if perturb is not None:
        i, j, d = perturb
        x[i, j] += d
    zz = x.astype(np.float64)
    zz = zz - zz.max(1, keepdims=True)
    ee = np.exp(zz)
    pp = ee / ee.sum(1, keepdims=True)
    return float(-np.log(pp[np.arange(rows), labels]).mean())


worst = {}
for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
    w = 0.0
    for (i, j) in [(0, 0), (3, 5), (11, 2), (20, 9), (31, 4)]:
        lp = loss_at((i, j, eps))
        lm = loss_at((i, j, -eps))
        num = (lp - lm) / (2 * eps)
        ana = float(analytic[i, j])
        w = max(w, abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
    worst[eps] = w
    print(f"   eps {eps:<8g} worst rel diff {w:.3e}")

best = min(worst, key=lambda k: worst[k])
print(f"\n3. best eps {best:g} gives {worst[best]:.3e}")
# A U-shape is the signature of a CORRECT gradient measured at the wrong step size:
# truncation error grows with eps, float noise shrinks with eps, so the error bottoms out.
# The earlier version of this script used a hair-trigger threshold and wrongly concluded
# "REAL BUG" from exactly this data. The test below is the shape test, not a ratio test.
big_eps = max(worst)               # eps = 1e-1
small_eps = min(k for k in worst if k <= 1e-4)
shaped = (worst[big_eps] > worst[best]) and (worst[small_eps] > worst[best])
print(f"   worst at eps {big_eps:g}: {worst[big_eps]:.3e}   worst at eps {small_eps:g}: "
      f"{worst[small_eps]:.3e}   minimum {worst[best]:.3e}")
if shaped and absd < 1e-6:
    print("   -> U-SHAPED and the analytic gradient matches a float64 reference to "
          f"{absd:.1e}.")
    print("      The gradient is CORRECT. The gate's eps=1e-2 was too coarse for softmax,")
    print("      and the gate should compare against a float64 reference instead.")
else:
    print("   -> NOT a clean U-shape, or the reference disagrees. Treat as a real bug.")
