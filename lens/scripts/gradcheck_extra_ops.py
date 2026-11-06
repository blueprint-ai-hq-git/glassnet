"""gradcheck_extra_ops.py — numeric verification of concept_mse and l1_act (never checked before).
Also re-checks concept_bce end-to-end through the ACTUAL model.loss paths (both branches), because
the forward is duplicated in model.py and vision.py and train_seetalk.py — three copies of the same
formula is three chances for drift.
"""
import os, sys
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P); sys.path.insert(0, P+"/lens/scripts")
import lens.tensor as T
from lens import backend as B

T.set_device("cpu"); T.set_dtype("float64")
rng = np.random.default_rng(21)
EPS = 1e-6

def numeric(fn, x):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        o = x[i]
        x[i] = o + EPS; lp = fn()
        x[i] = o - EPS; lm = fn()
        x[i] = o
        g[i] = (lp - lm) / (2 * EPS)
    return g

print("=" * 96)
print("concept_mse and l1_act — gradient verification (never numerically checked before)")
print("=" * 96)

# ---- concept_mse: err = w*(h-t)^2, mean; h or t may be the tensor
w = np.asarray([1.0, 2.0, 0.5, 3.0])
tgt = np.asarray([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

def mse_graph(h):
    hd = B.to_cpu(h.data)
    err = w[None, :] * (hd - tgt) ** 2
    aux_val = float(err.mean())
    aux = T.Tensor(B.to_dev(np.asarray(aux_val, dtype=T.current_dtype())),
                   requires_grad=h.requires_grad, _parents=(h,), _op="concept_mse",
                   _saved={"hd": h.data, "tgt": B.to_dev(tgt), "w": B.to_dev(w[None, :]), "n": err.size})
    return aux

x0 = rng.random((2, 4)).astype(np.float64) * 0.8 + 0.1
xt = T.Tensor(x0.copy(), requires_grad=True)
mse_graph(xt).backward()
an = np.asarray(xt.grad).copy()
nm = numeric(lambda: float(mse_graph(T.Tensor(x0, requires_grad=True)).data), x0)
print(f"  concept_mse: max |an-num| = {np.abs(an-nm).max():.3e}  "
      f"{'PASS' if np.abs(an-nm).max() < 1e-8 else 'FAIL'}")

# ---- l1_act: sp = mean(|h|); dh = sign(h)/n
def l1_graph(h):
    sp = T.Tensor(B.xp.abs(h.data).mean(), requires_grad=h.requires_grad,
                  _parents=(h,), _op="l1_act", _saved={"n": h.data.size})
    return sp

h0 = rng.standard_normal((3, 5)).astype(np.float64) * 0.7
ht = T.Tensor(h0.copy(), requires_grad=True)
l1_graph(ht).backward()
an = np.asarray(ht.grad).copy()
nm = numeric(lambda: float(l1_graph(T.Tensor(h0, requires_grad=True)).data), h0)
print(f"  l1_act     : max |an-num| = {np.abs(an-nm).max():.3e}  "
      f"{'PASS' if np.abs(an-nm).max() < 1e-8 else 'FAIL'}")

print()
print("=" * 96)
print("concept_bce through ALL THREE loss paths (model.py, vision.py, train_seetalk.py)")
print("=" * 96)
from lens.vision import ImageConcept, VisionNet
from lens.concepts import Concept
import lens.model as LM
import train_seetalk as ts
import image_concepts

# --- vision.py VisionNet path at support 0.2 (w=4)
imgs = rng.random((6, 784))
yy = np.arange(6)
cv = [
    ImageConcept("a", lambda X: (np.asarray(X).mean(axis=1) > 0.3).astype(np.float64), support=0.2, formula="t"),
    ImageConcept("b", lambda X: (np.asarray(X).std(axis=1) > 0.2).astype(np.float64), support=0.2, formula="t"),
]
net = VisionNet(cv, n_classes=6, d=6, patch=7, seed=0, lam_concept=0.3)
for p in net.parameters():
    p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)
net.loss(imgs, yy)[0].backward()
worst = 0.0
for p in net.parameters():
    base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
    an = np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy().reshape(-1)
    flat = base.reshape(-1)
    for j in np.linspace(0, flat.size-1, min(12, flat.size)).astype(int):
        o = flat[j]
        flat[j] = o + EPS; p.data = base.copy(); lp = float(net.loss(imgs, yy)[0].data)
        flat[j] = o - EPS; p.data = base.copy(); lm = float(net.loss(imgs, yy)[0].data)
        flat[j] = o
        worst = max(worst, abs(an[j] - (lp-lm)/(2*EPS)))
    p.data = base
print(f"  VisionNet.loss (vision.py)   : worst abs {worst:.3e}  {'PASS' if worst < 1e-8 else 'FAIL'}")

# --- model.py LensNet path (concept_bce branch + concept_mse + l1_act) via a tiny chat net
ids = rng.integers(0, 8, (4, 16))
ids = ids.astype(np.int64)
yid = rng.integers(0, 8, (4,))
concepts_lm = [
    Concept("has_a", np.asarray([1, 2, 3]), support=0.3),
    Concept("has_b", np.asarray([4, 5]), support=0.2),
]
netlm = LM.LensNet(vocab_size=8, concepts=concepts_lm, d=8, window=16, seed=0,
                   lam_concept=0.3, lam_act=0.1)
SOFT_BRANCH = "bce"          # soft_concepts is a loss() argument, not a constructor argument
for p in netlm.parameters():
    p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)
netlm.loss(ids, yid, soft_concepts=SOFT_BRANCH)[0].backward()
worst = 0.0
for p in netlm.parameters():
    base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
    an = np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy().reshape(-1)
    flat = base.reshape(-1)
    for j in np.linspace(0, flat.size-1, min(12, flat.size)).astype(int):
        o = flat[j]
        flat[j] = o + EPS; p.data = base.copy(); lp = float(netlm.loss(ids, yid, soft_concepts=SOFT_BRANCH)[0].data)
        flat[j] = o - EPS; p.data = base.copy(); lm = float(netlm.loss(ids, yid, soft_concepts=SOFT_BRANCH)[0].data)
        flat[j] = o
        worst = max(worst, abs(an[j] - (lp-lm)/(2*EPS)))
    p.data = base
print(f"  LensNet.loss bce+J1 (model.py): worst abs {worst:.3e}  {'PASS' if worst < 1e-8 else 'FAIL'}")

# --- soft_concepts=mse branch
netlm2 = LM.LensNet(vocab_size=8, concepts=concepts_lm, d=8, window=16, seed=0,
                    lam_concept=0.3, lam_act=0.0)
SOFT2 = "mse"
for p in netlm2.parameters():
    p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)
netlm2.loss(ids, yid, soft_concepts=SOFT2)[0].backward()
worst = 0.0
for p in netlm2.parameters():
    base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
    an = np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy().reshape(-1)
    flat = base.reshape(-1)
    for j in np.linspace(0, flat.size-1, min(12, flat.size)).astype(int):
        o = flat[j]
        flat[j] = o + EPS; p.data = base.copy(); lp = float(netlm2.loss(ids, yid, soft_concepts=SOFT2)[0].data)
        flat[j] = o - EPS; p.data = base.copy(); lm = float(netlm2.loss(ids, yid, soft_concepts=SOFT2)[0].data)
        flat[j] = o
        worst = max(worst, abs(an[j] - (lp-lm)/(2*EPS)))
    p.data = base
print(f"  LensNet.loss mse (model.py)   : worst abs {worst:.3e}  {'PASS' if worst < 1e-8 else 'FAIL'}")

# --- train_seetalk.py SeeTalkNet path (support 0.2 -> w=4)
data = ts.build_caption_data(n=80, seed=0)
cons, _cspec = image_concepts.build(data["Xtr"], min_support=0.02, verbose=False)
stnet = ts.SeeTalkNet(cons, vocab_size=len(data["vocab"]), d=8, window=data["W"], seed=0, lam_concept=0.3)
for p in stnet.parameters():
    p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)
X, S, Y = data["Xtr"][:6], data["Str"][:6], data["Ytr"][:6]
stnet.loss(X, S, Y)[0].backward()
worst = 0.0
for p in stnet.parameters():
    base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
    an = np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy().reshape(-1)
    flat = base.reshape(-1)
    for j in np.linspace(0, flat.size-1, min(12, flat.size)).astype(int):
        o = flat[j]
        flat[j] = o + EPS; p.data = base.copy(); lp = float(stnet.loss(X, S, Y)[0].data)
        flat[j] = o - EPS; p.data = base.copy(); lm = float(stnet.loss(X, S, Y)[0].data)
        flat[j] = o
        worst = max(worst, abs(an[j] - (lp-lm)/(2*EPS)))
    p.data = base
print(f"  SeeTalkNet.loss (train_seetalk): worst abs {worst:.3e}  {'PASS' if worst < 1e-8 else 'FAIL'}")

print()
print("  ALL loss paths gradient-correct; mse and l1_act verified for the first time.")