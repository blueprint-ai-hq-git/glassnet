"""dbg_aux_grad.py — print analytic vs numerical gradients side by side."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ); sys.path.insert(0, HERE)
import numpy as np
from glassnet.tensor import Tensor, take_cols, set_backend
from glassnet.losses import bce_logits

set_backend("numpy")
np.set_printoptions(precision=6, suppress=True, linewidth=150)

rng = np.random.default_rng(7)
N, C, NS = 3, 4, 2
cols = np.array([0, 2], dtype=np.int64)
pw = np.array([12.0, 0.5], dtype=np.float64)
y = (rng.random((N, NS)) < 0.3).astype(np.float64)
z = Tensor(rng.standard_normal((N, C)).astype(np.float64) * 0.8, requires_grad=True)
base = z.data.copy()

def loss_of(zz):
    return bce_logits(take_cols(zz, cols), y, w=pw)

L = loss_of(z)
print("loss =", float(L.data))
L.backward()
ana = z.grad.copy()

eps = 1e-6
num = np.zeros_like(base)
for i in range(N):
    for j in range(C):
        o = base[i, j]
        z.data[i, j] = o + eps; lp = float(loss_of(z).data)
        z.data[i, j] = o - eps; lm = float(loss_of(z).data)
        z.data[i, j] = o
        num[i, j] = (lp - lm) / (2 * eps)

print("\nz =\n", base)
print("\nANALYTIC =\n", ana)
print("\nNUMERICAL =\n", num)
print("\nDIFF =\n", ana - num)

# what the gradient SHOULD be, computed by hand
s = 1.0 / (1.0 + np.exp(-base))
hand = np.zeros_like(base)
d = (s[:, cols] - y) * pw[None, :] / y.size
hand[:, cols] = d
print("\nBY HAND (s-t)*w/size =\n", hand)
print("\nhand vs analytic max diff:", np.abs(hand - ana).max())
print("hand vs numeric  max diff:", np.abs(hand - num).max())
print("\nclipping active?", bool((s < 1e-7).any() or (s > 1 - 1e-7).any()))
print("y.size =", y.size, " (bce mean divides by this)")
print("sigmoid range:", s.min(), s.max())
