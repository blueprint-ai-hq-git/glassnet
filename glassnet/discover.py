"""glassnet.discover — Step 1: the model DISCOVERS its own concepts.

A concept layer of K learned sigmoid neurons with a SPARSITY penalty so each
neuron fires rarely and on few words; neurons are AUTO-NAMED afterwards from
the words that activate them most (top weights + example sentences); every
name is AUDITED on held-out sentences (does it fire on its named words and
stay quiet on others?); classes read the discovered concepts directly — no
rule layer, no unnamed gate. Lie control = pinned probe (one neuron's
concept weights frozen at init -> provably random -> audit must fail it).

Loss = CE(classes) + lam_act * mean(activations)   [neurons mostly OFF]
             + lam_w * mean(|concept.W|)           [few words per neuron]
"""
import os
import numpy as np

from .tensor import Tensor, sigmoid, abs_, sum_all
from .layers import Dense
from .losses import cross_entropy
from .optim import SGD
from . import text_data as TD


def _mean(x):                        # scalar mean via sum_all (0-d * scalar)
    return sum_all(x) * (1.0 / x.data.size)


class DiscoverNet:
    def __init__(self, K, classes, input_dim, rng=None, lam_act=1.0,
                 lam_w=1.0):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.K = int(K)
        self.classes = int(classes)
        self.input_dim = int(input_dim)
        self.lam_act = float(lam_act)
        self.lam_w = float(lam_w)
        self.concept = Dense(input_dim, K, rng=rng, name="concept")
        self.answer = Dense(K, classes, rng=rng, name="answer")
        self.frozen_cols = []

    def freeze(self, cols):
        """Pinned-probe (lie control): zero the gradient of these concept
        columns every step, so they stay at random init — provably not
        connected to anything they might be named after."""
        self.frozen_cols = list(cols)

    def forward(self, x):
        z = self.concept(x)
        h = sigmoid(z)
        return {"concepts": h, "concept_pre": z, "logits": self.answer(h)}

    def loss(self, x, y, f=None):
        f = f or self.forward(x)
        ce = cross_entropy(f["logits"], y)
        sa = self.lam_act * _mean(f["concepts"])
        sw = self.lam_w * _mean(abs_(self.concept.W))
        return ce + sa + sw, {"ce": ce, "sp_act": sa, "sp_w": sw}

    def parameters(self):
        return self.concept.parameters() + self.answer.parameters()

    def count_parameters(self):
        return sum(p.data.size for p in self.parameters())

    def save(self, path):
        out = {"concept.W": self.concept.W.data,
               "concept.b": self.concept.b.data,
               "answer.W": self.answer.W.data,
               "answer.b": self.answer.b.data,
               "K": np.int32(self.K), "classes": np.int32(self.classes),
               "input_dim": np.int32(self.input_dim),
               "lam_act": np.float32(self.lam_act),
               "lam_w": np.float32(self.lam_w)}
        np.savez(path, **out)

    @staticmethod
    def load(path):
        z = np.load(path)
        net = DiscoverNet(int(z["K"]), int(z["classes"]),
                          int(z["input_dim"]),
                          lam_act=float(z["lam_act"]),
                          lam_w=float(z["lam_w"]))
        net.concept.W.data[:] = z["concept.W"]
        net.concept.b.data[:] = z["concept.b"]
        net.answer.W.data[:] = z["answer.W"]
        net.answer.b.data[:] = z["answer.b"]
        return net


def val_acc(net, Xval, yval):
    pred = net.forward(Tensor(Xval))["logits"].data.argmax(1).astype(np.int64)
    return float(np.mean(pred == np.asarray(yval)))


def train_discover(Xtr, ytr, Xval, yval, K, classes, seed, epochs=48,
                   batch=128, lr=0.1, lam_act=1.0, lam_w=1.0, frozen=None):
    """3-seed-friendly single run; returns (net, best_val_acc)."""
    net = DiscoverNet(K, classes, Xtr.shape[1],
                      rng=np.random.default_rng(seed),
                      lam_act=lam_act, lam_w=lam_w)
    if frozen:
        net.freeze(frozen)
    opt = SGD(net.parameters(), lr=lr)
    rngd = np.random.default_rng(seed * 777 + 1)
    best_v, best_ep, best_ws = -1.0, -1, None
    for ep in range(epochs):
        perm = rngd.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss, _ = net.loss(Tensor(Xtr[idx]), ytr[idx])
            loss.backward()
            if net.frozen_cols and net.concept.W.grad is not None:
                for k in net.frozen_cols:
                    net.concept.W.grad[:, k] = 0.0
                    net.concept.b.grad[k] = 0.0
            opt.step()
        v = val_acc(net, Xval, yval)
        if v > best_v:
            best_v, best_ep = v, ep
            best_ws = {p.name: p.data.copy() for p in net.parameters()}
    for p in net.parameters():
        p.data[:] = best_ws[p.name]
    return net, best_v


def auto_name(net, feat_names, Xtr, texts, topn=8, n_examples=2,
              max_ex_scan=6000):
    """Name each neuron from the words that DRIVE it ON (top POSITIVE
    weights — a sigmoid neuron fires when w*x+b > 0, so only positive
    weights') + fires-on examples. Stopwords dropped (they name nothing).
    feat_names covers ALL input columns: plain tokens and bigram tokens
    ("card_lost" — the audit splits these back into their words)."""
    STOP = {"i", "my", "me", "we", "you", "your", "the", "a", "an", "to",
            "up", "can", "is", "are", "am", "was", "were", "for", "why",
            "how", "what", "where", "when", "which", "who", "do", "does",
            "did", "it", "on", "in", "at", "and", "or", "of", "with",
            "about", "want", "need", "get", "got", "please", "would",
            "could", "should", "this", "that", "have", "has", "had", "be",
            "been", "so", "but", "not", "no", "yes", "ok", "thanks"}
    W = net.concept.W.data
    names, examples = [], []
    h = net.forward(Tensor(Xtr[:max_ex_scan]))["concepts"].data
    for k in range(net.K):
        order = np.argsort(-W[:, k])            # descending (positive first)
        words = []
        for j in order[:topn]:
            if W[order[j], k] <= 1e-4:
                continue
            tok = feat_names[j]
            parts = tok.split("_") if "_" in tok else [tok]
            if all(p in STOP for p in parts):
                continue                        # stopword-only feature
            words.append(tok)
        nm = "_".join(words[:6]) if words else f"neuron{k}"
        names.append(nm)
        top = np.argsort(-h[:, k])[:n_examples]
        examples.append([texts[i] for i in top])
    return names, examples


def _auroc(score, target01):
    pos = target01 > 0
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def audit_discover(net, names, Xval, texts_val):
    """Held-out audit. TWO criteria, both required for a name to PASS:
    (1) residualised AUROC >= 0.55 — the neuron fires on sentences
        containing its named words, beyond what utterance length predicts;
    (2) NAME-CONTRIBUTION >= 0.25 LOGITS — on sentences where the neuron
        fires (h > 0.5), the named words' logit contribution must average
        >= 0.25. Measured in LOGIT units (additive for a logistic unit) —
        the earlier activation-space threshold 0.15 was miscalibrated:
        sigmoid saturation compressed honest neurons' visible drops to
        ~0.08-0.30 while the random lie column sat at 0.000; logit-drop
        separates cleanly (measured on iter-022 models: honest 0.34-1.36,
        lie 0.000; threshold 0.25 pre-registered BEFORE iter 023).
    quiet/loud means are on the length-residual as before."""
    h = net.forward(Tensor(Xval))["concepts"].data
    toks = [TD.tokenize(t) for t in texts_val]
    L = np.log1p(np.array([len(t) for t in toks], dtype=np.float64))
    A = np.vstack([np.ones_like(L), L]).T
    W = net.concept.W.data
    vocab = TD.load_vocab()
    vmap = {w: i for i, w in enumerate(vocab)} if vocab else {}
    rows = []
    for k, nm in enumerate(names):
        words = set(nm.split("_")) - {"neuron"}
        if not words or nm.startswith("neuron"):
            rows.append({"neuron": k, "name": nm, "auroc": float("nan"),
                         "contrib": float("nan"), "pos_frac": float("nan"),
                         "quiet_mean": float("nan"), "loud_mean": float("nan"),
                         "flag": "NO-NAME"})
            continue
        hk = h[:, k]
        coef, *_ = np.linalg.lstsq(A, hk, rcond=None)
        resid = hk - A @ coef
        target = np.array([1.0 if any(w in tk for w in words) else 0.0
                           for tk in toks])
        a = _auroc(resid, target)
        # name-contribution: zero the named words' logit part
        named_idx = [vmap[w] for w in words if w in vmap]
        z = net.forward(Tensor(Xval))["concept_pre"].data     # logits (N,K)
        logit_named = np.zeros(len(Xval))
        if named_idx:
            Xv = np.asarray(Xval)
            logit_named = Xv[:, named_idx] @ W[named_idx, k]
        h_noname = 1.0 / (1.0 + np.exp(-(z[:, k] - logit_named)))
        fired = hk > 0.5
        contrib = float(logit_named[fired].mean()) if fired.any() else 0.0
        quiet = float(resid[target == 0].mean()) if (target == 0).any() else 0.0
        loud = float(resid[target == 1].mean()) if target.sum() > 0 else 0.0
        broken = not (a >= 0.55 and contrib >= 0.25)
        rows.append({"neuron": k, "name": nm, "auroc": a, "contrib": contrib,
                     "pos_frac": float(target.mean()), "quiet_mean": quiet,
                     "loud_mean": loud,
                     "flag": "BROKEN" if broken else "-"})
    n_pass = sum(1 for r in rows if not r["flag"].startswith("BROKEN")
                 and r["flag"] != "NO-NAME")
    return rows, n_pass


def discover_benchmark(Xtr, ytr, Xval, yval, K, seeds=(0, 1, 2), epochs=48,
                       batch=128, lr=0.1, lam_act=1.0, lam_w=1.0):
    """3-seed discovered-concept run (no frozen columns). Returns
    (vals, nets) — vals[s] = best-val accuracy %."""
    vals, nets = {}, {}
    for s in seeds:
        net, v = train_discover(Xtr, ytr, Xval, yval, K, len(np.unique(ytr)),
                                seed=s, epochs=epochs, batch=batch, lr=lr,
                                lam_act=lam_act, lam_w=lam_w)
        vals[s] = v * 100
        nets[s] = net
    return vals, nets


def discover_params(K, classes, input_dim):
    return input_dim * K + K + K * classes + classes