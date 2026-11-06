"""dbg_split_loss.py — which part of the loss has the wrong gradient: the CE or the concept aux?

The full-net check showed opposite-sign analytic vs numeric on patch_proj.W entries. Split the loss:
  * aux-only  : net.loss(imgs, y)[1]['concept_aux'] enlarged by a big lam
  * ce-only   : the CE term alone
Each gets its own backward + numeric check on a few patch_proj.W / concept.W entries. The broken
half is the one whose signs flip.
"""
import os, sys
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P); sys.path.insert(0, P+"/lens/scripts")
import lens.tensor as T
from lens import backend as B
from lens.vision import ImageConcept, VisionNet

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
net = VisionNet(concepts, n_classes=8, d=8, patch=7, seed=0, lam_concept=0.3)
for p in net.parameters():
    p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)

# what does net.loss actually sum? check parts
loss, parts = net.loss(imgs, y)
print("parts:", {k: (round(float(v), 6) if hasattr(v, '__float__') else type(v).__name__) for k, v in parts.items()})
print("loss:", float(loss.data))

# build the same net by hand to split the loss terms
import lens.tensor as TT

def forward_full(net_, imgs_):
    out = net_.forward_arrays(imgs_)   # dict with concepts h and logits
    return out

out = net.forward_arrays(imgs)
h, lg = out["concepts"], out["logits"]
ce = TT.softmax_ce(lg, y)
# aux: BCE of h against concept targets
tg = net.concept_targets(imgs, None) if hasattr(net, "concept_targets") else None
print("has concept_targets:", tg is not None)
print("h shape:", h.data.shape, "  targets available")

# replicate aux as BCE mean
aux = TT.mul(h, TT.Tensor(0.0))  # placeholder
print("---")
print("forward_arrays keys:", list(out.keys()))