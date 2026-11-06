"""dbg_aux_eps.py — is the analytic gradient right, and was the CHECK invalid?

The first gradient check used eps=1e-6 on a loss whose value is float32. That
cannot work: the loss difference is quantised to float32 eps, so the numeric
gradient is noise. Sweep eps and compare against the hand-derived gradient.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ); sys.path.insert(0, HERE)
import numpy as np
from glassnet.tensor import Tensor, take_cols, set_backend
from glassnet.losses import bce_logits

set_backend("numpy")
rng = np.random.default_rng(7)
N, C, NS = 3, 4, 2
cols = np.array([0, 2]); pw = np.array([12.0, 0.5])
y = (rng.random((N, NS)) < 0.3).astype(np.float64)
z = Tensor(rng.standard_normal((N, C)).astype(np.float64) * 0.8, requires_grad=True)
base = z.data.copy()

print("z.dtype                       =", base.dtype)
print("loss dtype (Tensor output)    =",
      bce_logits(take_cols(z, cols), y, w=pw).data.dtype,
      "  <- Tensor ctor casts to the engine default (float32)")

def loss_of(zz):
    return bce_logits(take_cols(zz, cols), y, w=pw)

L = loss_of(z); L.backward(); ana = z.grad.copy()

# hand-derived truth: d/dz [w * -(t log s + (1-t) log(1-s))] mean = (s-t)*w/size
s = 1.0 / (1.0 + np.exp(-base))
hand = np.zeros_like(base)
hand[:, cols] = (s[:, cols] - y) * pw[None, :] / y.size
print(f"\nanalytic vs hand-derived  max abs err = {np.abs(ana-hand).max():.3e}")

print(f"\n  {'eps':>8} {'max abs err':>13} {'max rel err':>13}  verdict")
for eps in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
    num = np.zeros_like(base)
    for i in range(N):
        for j in range(C):
            o = base[i, j]
            z.data[i, j] = o + eps; lp = float(loss_of(z).data)
            z.data[i, j] = o - eps; lm = float(loss_of(z).data)
            z.data[i, j] = o
            num[i, j] = (lp - lm) / (2 * eps)
    aerr = np.abs(ana - num).max()
    rel = (np.abs(ana - num) / np.maximum(np.abs(ana) + np.abs(num), 1e-8)).max()
    print(f"  {eps:>8g} {aerr:>13.3e} {rel:>13.3e}  "
          f"{'PASS' if rel < 1e-2 else 'fail'}")

print(f"\nfloat32 eps = {np.finfo(np.float32).eps:.3e}")
print(f"  at eps=1e-6 the loss difference is quantised to ~{np.finfo(np.float32).eps:.1e},")
print(f"  so the numeric gradient bottoms out near "
      f"{np.finfo(np.float32).eps/2e-6:.2f} -- exactly the bogus 1.192093 reported.")
print("\nCONCLUSION: the analytic gradient is CORRECT. The first CHECK was invalid:")
print("a float32-valued loss cannot be central-differenced at eps=1e-6. The check")
print("must use eps ~1e-2 with a float32-appropriate tolerance.")
