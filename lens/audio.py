"""lens/audio.py — the AUDIO line: spectral columns as patches onto the ONE named-concept layer.

Mirrors LensNet/VisionNet structurally on purpose (the "patch is a token" rule): a column of the
spectrogram is a patch, exactly like a text token or an image patch. The layer after the
front-end is the SAME: sigmoid concept layer (named, trained to match concept labels), answer
layer, exact explain(). Nothing is hidden: forward, backward, loss, and optimizer all run on
lens/tensor.py ops; every gradient is covered by the numeric audit in gradcheck_full_ops.
"""
import json
import os

import numpy as np

from lens import tensor as T
from lens import backend
from lens.model import Dense


class AudioNet:
    """audio window (N, W, F) -> shared column projection F->d -> concept layer -> answer.

    Classes of the answer layer ARE the concept names (self-describing output: "this 0.4 s is
    has_voiced because of concept has_voiced (0.87)"). C = K by design for the audio line; a
    standalone audio QA head can be swapped in the joint model without touching the layer.
    """

    def __init__(self, concepts, d=64, window=50, f_bins=65, seed=0,
                 lam_concept=0.3, lam_act=0.0, class_order=None):
        if not concepts:
            raise ValueError("AudioNet needs named concepts; the concept layer IS the architecture.")
        self.concepts = list(concepts)
        self.K = len(concepts)
        self.C = len(concepts)          # answer classes == number of concepts
        self.class_order = list(class_order) if class_order is not None else list(range(self.K))
        # class c in the answer layer means "concept class_order[c]" — the semantic mapping,
        # so explain() must print through it (never assume class index == concept index)
        self.answer_classes = [self.concepts[i].name for i in self.class_order]
        self.D = int(d)
        self.W = int(window)
        self.F = int(f_bins)
        self.seed = int(seed)
        self.lam_concept = float(lam_concept)
        self.lam_act = float(lam_act)
        self.lam_w = 0.0
        self.device_name = "cpu"

        rng = np.random.default_rng(seed)
        self.front = Dense(self.F, self.D, rng, name="front")
        self.concept = Dense(self.W * self.D, self.K, rng, name="concept")
        self.answer = Dense(self.K, self.C, rng, name="answer")

    @property
    def attention(self):
        return False

    def parameters(self):
        return self.front.parameters() + self.concept.parameters() + self.answer.parameters()

    def forward(self, x):
        """x: (N, W, F) float32 log-scaled spectral windows -> logits (N, C)."""
        N, W, F = x.shape
        assert F == self.F and W == self.W, f"expected (N,{self.W},{self.F}), got {x.shape}"
        cols = T.reshape(x, (N * W, F))
        proj = T.reshape(T.add(T.matmul(cols, self.front.W), self.front.b), (N, W * self.D))
        h = T.sigmoid(T.add(T.matmul(proj, self.concept.W), self.concept.b))
        logits = T.add(T.matmul(h, self.answer.W), self.answer.b)
        return logits, h

    def concept_targets(self, x_np):
        return np.stack([c.labels(x_np) for c in self.concepts], axis=1).astype(np.float32)

    def loss(self, x_np, y_idx, tgt_np, class_weights=None):
        """Classification CE + concept BCE (registered op, same as LensNet) + sparsity.

        tgt_np: the concept-label matrix (N, K) — for the audio line this comes from the
        audio_data cache; audio concepts carry name+formula text only. class_weights balances
        the CE when one class dominates (the majority-collapse that killed the first 2-concept
        run: with 86% voiced windows, plain CE optimised to constant concepts and never fired).
        """
        x = T.Tensor(backend.to_dev(np.asarray(x_np, dtype=np.float32)), requires_grad=False)
        logits, h = self.forward(x)
        ce = T.softmax_ce(logits, np.asarray(y_idx, dtype=np.int64),
                          class_weights=class_weights)
        n, k = h.data.shape
        hd = backend.to_cpu(h.data)
        t = np.asarray(tgt_np, dtype=T.current_dtype())
        pos_w = (1.0 - t.sum(axis=0) / n) / (t.sum(axis=0) / n + 1e-12)
        w = pos_w[None, :]
        eps = 1e-7
        bce = -(w * t * np.log(hd + eps) + (1.0 - t) * np.log(1.0 - hd + eps))
        aux = T.Tensor(backend.to_dev(np.asarray(bce.mean(), dtype=T.current_dtype())),
                       requires_grad=h.requires_grad, _parents=(h,), _op="concept_bce",
                       _saved={"hd": h.data, "tgt": backend.to_dev(t),
                               "w": backend.to_dev(w), "n": bce.size})
        sp = T.Tensor(backend.xp.abs(h.data).mean(), requires_grad=h.requires_grad,
                      _parents=(h,), _op="l1_act", _saved={"n": h.data.size})
        lam_c = T.Tensor(np.asarray(self.lam_concept, dtype=T.current_dtype()))
        lam_a = T.Tensor(np.asarray(self.lam_act, dtype=T.current_dtype()))
        total = T.add(T.add(ce, T.mul(aux, lam_c)), T.mul(sp, lam_a))
        return total, {"ce": float(backend.to_cpu(ce.data)),
                       "concept": float(backend.to_cpu(aux.data)),
                       "sparsity": float(backend.to_cpu(sp.data))}

    def concept_loss(self, x_np, tgt_np, lam=1.0):
        """CONCEPT-ONLY loss (no CE): the video line's answer layer reads argmax(h), so each
        neuron trains independently against its own formula — per-neuron BCE cannot fight like a
        shared CE did (91/9 classes killed whichever concept the majority-optimised CE ignored).
        """
        x = T.Tensor(backend.to_dev(np.asarray(x_np, dtype=np.float32)), requires_grad=False)
        _, h = self.forward(x)
        n, k = h.data.shape
        hd = backend.to_cpu(h.data)
        t = np.asarray(tgt_np, dtype=T.current_dtype())
        pos_w = (1.0 - t.sum(axis=0) / n) / (t.sum(axis=0) / n + 1e-12)
        w = pos_w[None, :]
        eps = 1e-7
        bce = -(w * t * np.log(hd + eps) + (1.0 - t) * np.log(1.0 - hd + eps))
        aux = T.Tensor(backend.to_dev(np.asarray(bce.mean(), dtype=T.current_dtype())),
                       requires_grad=h.requires_grad, _parents=(h,), _op="concept_bce",
                       _saved={"hd": h.data, "tgt": backend.to_dev(t),
                               "w": backend.to_dev(w), "n": bce.size})
        out = T.mul(aux, T.Tensor(np.asarray(lam, dtype=T.current_dtype())))
        return out, {"concept": float(backend.to_cpu(aux.data))}

    def n_params(self):
        return sum(int(p.data.size) for p in self.parameters())


def save(net, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {p.name: backend.to_cpu(p.data) for p in net.parameters()}
    meta = {"D": net.D, "W": net.W, "F": net.F, "K": net.K, "C": net.C, "seed": net.seed,
            "lam_concept": net.lam_concept, "lam_act": net.lam_act,
            "concepts": [{"name": c.name, "formula": c.formula} for c in net.concepts]}
    np.savez(path, **data, __meta__=json.dumps(meta))


def load(path, device="cpu"):
    z = np.load(path)
    meta = json.loads(str(z["__meta__"]))
    # labels() for audio concepts come from the audio_data cache (the Y matrix), never from the
    # net itself: a concept object here carries name + formula text only (no lying zero-labels)
    concepts = [type("C", (), {"name": c["name"], "formula": c["formula"], "labels": None})()
                for c in meta["concepts"]]
    net = AudioNet(concepts, d=meta["D"], window=meta["W"], f_bins=meta["F"],
                   seed=meta["seed"], lam_concept=meta["lam_concept"], lam_act=meta["lam_act"])
    T.set_device(device)
    for p in net.parameters():
        if p.name not in z:
            raise KeyError(f"{path} has no parameter {p.name!r}")
        p.data = backend.to_dev(np.asarray(z[p.name], dtype=np.float32))
        p.grad = None
    return net


def explain(net, x_np, top_k=3):
    """I said <class> because of concepts... — EXACT arithmetic of the net, nothing hidden."""
    logits, h = net.forward(T.Tensor(backend.to_dev(np.asarray(x_np, dtype=np.float32))))
    l = backend.to_cpu(logits.data)[0]
    hh = backend.to_cpu(h.data)[0]
    cls = int(np.argmax(l))
    order = np.argsort(-hh)
    reasons = [(net.concepts[i].name, float(hh[i])) for i in order[:top_k]]
    return {"class": cls, "class_name": net.answer_classes[cls],
            "scores": {net.answer_classes[cls]: float(l[cls])},
            "reasons": reasons}