"""dbg_aux_term.py — test the concept_bce aux term in ISOLATION.

Hypothesis: CE gradient is correct, aux (concept_bce) gradient has the wrong sign or magnitude,
and the full-net check saw ce + aux_wrong vs numeric ce + aux_right.
"""
import os, sys
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P); sys.path.insert(0, P+"/lens/scripts")
import lens.tensor as TT
from lens import backend as B
from lens.vision import ImageConcept, VisionNet

T = TT
T.set_device("cpu"); T.set_dtype("float64")
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

def make_net(lam):
    n = VisionNet(concepts, n_classes=8, d=8, patch=7, seed=0, lam_concept=lam)
    for p in n.parameters():
        p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)
    return n

# analytic grad with lam=0 (CE only) and lam=0.3 (CE + aux)
net0 = make_net(0.0)
loss0, parts0 = net0.loss(imgs, y)
loss0.backward()
g0 = {p.name: np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy() for p in net0.parameters()}

net3 = make_net(0.3)
loss3, parts3 = net3.loss(imgs, y)
loss3.backward()
g3 = {p.name: np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy() for p in net3.parameters()}

print("parts lam0:", parts0)
print("parts lam3:", parts3)

# analytic AUX gradient = g3 - g0
print("\n  param           |g0|max (CE)   |g3|max (CE+aux)   |g3-g0|max (aux)")
for nm in g0:
    print(f"  {nm:<16} {np.abs(g0[nm]).max():>12.3e} {np.abs(g3[nm]).max():>14.3e}  {np.abs(g3[nm]-g0[nm]).max():>14.3e}")

# numeric AUX gradient: (L3(x+eps)-L3(x-eps)) - (L0(x+eps)-L0(x-eps)) over 2eps
# pick patch_proj.W[13] — the worst entry from the full check
pname = "patch_proj.W"
p = [p for p in net3.parameters() if p.name == pname][0]
base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
j = 13
o = base.reshape(-1)[j]
base.reshape(-1)[j] = o + eps; p.data = base.copy()
lp0 = float(net0.loss(imgs, y)[0].data); lp3 = float(net3.loss(imgs, y)[0].data)
base.reshape(-1)[j] = o - eps; p.data = base.copy()
lm0 = float(net0.loss(imgs, y)[0].data); lm3 = float(net3.loss(imgs, y)[0].data)
base.reshape(-1)[j] = o
p.data = base
num_ce = (lp0 - lm0) / (2 * eps)
num_aux = (lp3 - lm3) / (2 * eps) - num_ce

print(f"\n  patch_proj.W[{j}]:")
print(f"    analytic CE  grad : {g0[pname].reshape(-1)[j]:+.6e}")
print(f"    numeric   CE  grad : {num_ce:+.6e}")
print(f"    analytic AUX grad  : {float((g3[pname]-g0[pname]).reshape(-1)[j]):+.6e}")
print(f"    numeric   AUX grad  : {num_aux:+.6e}")
print(f"    aux sign match     : {np.sign((g3[pname]-g0[pname]).reshape(-1)[j]) == np.sign(num_aux)}")
print(f"\n  -> if analytic aux and numeric aux DISAGREE (sign or magnitude), the bug is in")
print(f"     `concept_bce` backward (lens/model.py _bw_concept_bce).")