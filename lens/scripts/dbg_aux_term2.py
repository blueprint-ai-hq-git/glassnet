"""dbg_aux_term2.py — isolation, done right. Two same-seed nets (lam 0 and lam 0.3); perturb EACH
net's own parameter and evaluate EACH net's own loss. The CE numeric and AUX numeric gradients are
then meaningful, and can be compared against the analytic split (g3 - g0).
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

net0 = make_net(0.0)
net3 = make_net(0.3)

def an_grad(net, name):
    for p in net.parameters():
        p.grad = None
    loss, _ = net.loss(imgs, y)
    loss.backward()
    for p in net.parameters():
        if p.name == name:
            return np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy()

def num_grad(net, name, j):
    p = [p for p in net.parameters() if p.name == name][0]
    base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
    flat = base.reshape(-1)
    o = flat[j]
    flat[j] = o + eps
    p.data = base.copy()
    lp = float(net.loss(imgs, y)[0].data)
    flat[j] = o - eps
    p.data = base.copy()
    lm = float(net.loss(imgs, y)[0].data)
    flat[j] = o
    p.data = base
    return (lp - lm) / (2 * eps)

for name, j in [("patch_proj.W", 13), ("patch_proj.W", 118), ("concept.W", 212), ("patch_proj.b", 2)]:
    an_ce = an_grad(net0, name)          # CE analytic
    an_tot = an_grad(net3, name)         # CE+aux analytic
    an_aux = an_tot - an_ce              # aux analytic
    nm_ce = num_grad(net0, name, j)      # CE numeric
    nm_tot = num_grad(net3, name, j)     # CE+aux numeric
    nm_aux = nm_tot - nm_ce              # aux numeric
    print(f"  {name}[{j}]:")
    print(f"    CE  an {an_ce.reshape(-1)[j]:+.6e}  num {nm_ce:+.6e}  match {np.isclose(an_ce.reshape(-1)[j], nm_ce, rtol=1e-5, atol=1e-9)}")
    print(f"    AUX an {an_aux.reshape(-1)[j]:+.6e}  num {nm_aux:+.6e}  match {np.isclose(an_aux.reshape(-1)[j], nm_aux, rtol=1e-4, atol=1e-8)}")
    print(f"    TOT an {an_tot.reshape(-1)[j]:+.6e}  num {nm_tot:+.6e}  match {np.isclose(an_tot.reshape(-1)[j], nm_tot, rtol=1e-4, atol=1e-8)}")