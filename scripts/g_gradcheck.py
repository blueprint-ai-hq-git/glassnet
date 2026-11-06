"""g_gradcheck.py — numerical gradient check for ArithNet, including the embedding scatter path
that was just fixed. Run before trusting any Target G number.
"""
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)

import numpy as np  # noqa: E402

from glassnet import arith                       # noqa: E402
from glassnet.arith_model import ArithNet        # noqa: E402
from glassnet.tensor import set_backend          # noqa: E402

set_backend("numpy")
EPS = 1e-6

print("=" * 92)
print("ARITHNET GRADIENT CHECK (float64, central differences)")
print("=" * 92)

rng = np.random.default_rng(0)
probs = arith.problems()[:12]
X = np.array([[arith.TOKEN[arith.PAD], arith.TOKEN[arith.BOS], arith.TOKEN[str(a)],
               arith.TOKEN[arith.PLUS], arith.TOKEN[str(b)], arith.TOKEN[arith.EQ]]
              for a, b in probs])
y = np.array([arith.encode(a, b)[4] for a, b in probs])
lab = np.stack([[fn(a, b) for _, fn in arith.CONCEPTS] for a, b in probs], axis=1)

net = ArithNet(k=8, d=16, seed=1, readable=True, lam_concept=0.1)
for p in net.parameters():
    p.data = p.data.astype(np.float64)


def total_loss():
    out = net.forward(X)
    logits = out["logits"].data
    z = logits - logits.max(1, keepdims=True)
    ex = np.exp(z)
    probs_ = ex / ex.sum(1, keepdims=True)
    lm = float(-np.log(np.maximum(probs_[np.arange(len(y)), y], 1e-12)).mean())
    h = np.asarray(out["concepts"].data)
    eps = 1e-7
    aux = float(-(lab * np.log(h + eps) + (1 - lab) * np.log(1 - h + eps)).mean())
    return lm + 0.1 * aux


# analytic
out = net.forward(X)
loss = net.graph_loss(X, y, lab)
for p in net.parameters():
    p.grad = None
loss.backward()
analytic = {p.name: p.grad.copy() for p in net.parameters()}

worst = {}
for p in net.parameters():
    ng = np.zeros_like(p.data)
    it = np.nditer(p.data, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        orig = p.data[i]
        p.data[i] = orig + EPS
        lp = total_loss()
        p.data[i] = orig - EPS
        lm_ = total_loss()
        p.data[i] = orig
        ng[i] = (lp - lm_) / (2 * EPS)
    a = analytic[p.name]
    denom = np.maximum(np.abs(a) + np.abs(ng), 1e-12)
    rel = float(np.max(np.abs(a - ng) / denom))
    worst[p.name] = (float(np.max(np.abs(a - ng))), rel)

ok = True
for name, (absd, rel) in worst.items():
    flag = "PASS" if rel < 1e-5 else "FAIL"
    if flag == "FAIL":
        ok = False
    print(f"  {flag}  {name:<12} max abs diff {absd:.3e}   max rel diff {rel:.3e}")

print(f"\nOVERALL: {'PASS' if ok else 'FAIL'} — every parameter's gradient matches central "
      f"differences (rel tol 1e-5)")
sys.exit(0 if ok else 1)
