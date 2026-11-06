"""RUNG T2 iter 001 — tiny next-word prediction on tiny_shakespeare.

Pre-registered (written BEFORE the run):
HYPOTHESIS: a windowed word-bag (sum of learned embeddings over the last 8
tokens) feeding the rung-3 discovered-concept machinery (K=24 sigmoid
concepts, lam_act=0.2 lam_w=1.0, classes replaced by the next-word
softmax over V tokens) learns real next-word signal, and its concepts can
be named + audited exactly like rung 3 (names = words by concept-row ·
embedding dot; audit = residualized AUROC >= 0.55 AND logit
name-contribution >= 0.25, the standing meter; lie probe = frozen column).
SUCCESS TEST (fixed here):
  A) val next-word top-1 acc >= 2x the most-frequent-word prior
     (prior ~5% -> bar 10%);
  B) >= 50% of the 24 names pass the audit;
  C) lie probe caught.
Prediction (before run): acc 12-25% at 48 epochs (PPL ~30-80), audit
10-18/24, lie caught. TEST split untouched (quarantined).

One architecture note: the embed-sum bag is a LEARNED bag (differentiable
n-gram-free context) — this is the minimal Step-1 transfer to a LM, and
the naming/audit meter stays literally the rung-3 one, just computed by
word-embedding dot products (causal: removing a word removes its embedding
contribution to the concept's logit).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from glassnet.tensor import (Tensor, sigmoid, abs_, sum_all, sum_axis,
                             embedding, relu)
from glassnet.layers import Dense
from glassnet.losses import cross_entropy
from glassnet.optim import SGD
from glassnet import tiny_data

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runs", "t2")
ITERDIR = os.path.join(RUNS, "iter002")

K, D, WIN = 24, 64, 8
EPOCHS, LR, BATCH = 96, 0.1, 128
SKIP_NORMAL = os.environ.get("T2_SKIP_NORMAL", "0") == "1"  # iter 002 scope (2h cap)
LAM_A, LAM_W = 0.2, 1.0
SEEDS = (0, 1, 2)

STOP = {"i", "my", "me", "we", "you", "your", "the", "a", "an", "to", "up",
        "can", "is", "are", "am", "was", "were", "for", "why", "how", "what",
        "where", "when", "which", "who", "do", "does", "did", "it", "on",
        "in", "at", "and", "or", "of", "with", "about", "want", "need",
        "get", "got", "please", "would", "could", "should", "this", "that",
        "have", "has", "had", "be", "been", "so", "but", "not", "no", "yes",
        "ok", "thanks", "as", "if", "by", "from", "will", "shall", "may",
        "must", "now", "then", "yet", "all", "thy", "thou", "thee", "your",
        "his", "her", "our", "their"}


def _mean(x):
    return sum_all(x) * (1.0 / x.data.size)


class TinyLM:
    """Readable next-word model: bag-of-embeddings -> concepts -> logits."""

    def __init__(self, V, K, D, rng=None, lam_act=0.2, lam_w=1.0):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.V, self.K, self.D = V, K, D
        self.lam_act, self.lam_w = lam_act, lam_w
        s = D ** -0.5
        self.emb = Tensor(rng.standard_normal((V, D)).astype(np.float32) * s,
                          requires_grad=True, name="emb")
        self.concept = Dense(D, K, rng=rng, name="concept")
        self.answer = Dense(K, V, rng=rng, name="answer")
        self.frozen = []

    def freeze(self, cols):
        self.frozen = list(cols)

    def forward(self, ids):
        bag = sum_axis(embedding(self.emb, ids), 1)      # (N,D)
        z = self.concept(bag)
        h = sigmoid(z)
        logits = self.answer(h)
        return {"bag": bag, "concept_pre": z, "concepts": h,
                "logits": logits}

    def loss(self, ids, y):
        f = self.forward(ids)
        ce = cross_entropy(f["logits"], y)
        sa = self.lam_act * _mean(f["concepts"])
        sw = self.lam_w * _mean(abs_(self.concept.W))
        return ce + sa + sw, {"ce": ce, "sp_act": sa, "sp_w": sw}

    def parameters(self):
        return [self.emb] + self.concept.parameters() + self.answer.parameters()

    def n_params(self):
        return sum(p.data.size for p in self.parameters())

    def save(self, path):
        np.savez(path, emb=self.emb.data, cW=self.concept.W.data,
                 cb=self.concept.b.data, aW=self.answer.W.data,
                 ab=self.answer.b.data, V=np.int32(self.V),
                 K=np.int32(self.K), D=np.int32(self.D),
                 lam_act=np.float32(self.lam_act),
                 lam_w=np.float32(self.lam_w))


def tiny_normal(V, D, h, rng=None):
    """Same-size normal MLP on the same bag-of-embeddings input."""
    rng = rng if rng is not None else np.random.default_rng(0)
    s = D ** -0.5
    return (Tensor(rng.standard_normal((V, D)).astype(np.float32) * s,
                   requires_grad=True, name="emb"),
            Dense(D, h, rng=rng, name="n1"), Dense(h, V, rng=rng, name="n2"))


def normal_forward(net, ids):
    emb, l1, l2 = net
    bag = sum_axis(embedding(emb, ids), 1)
    return l2(relu(l1(bag)))


normal_params = lambda V, D, h: V * D + D * h + h + h * V + V
readable_params = lambda V, D, K: V * D + D * K + K + K * V + V


def match_h(V, D, K):
    rp = readable_params(V, D, K)
    return (rp - V * D - V) // (D + 1 + V)


def acc_of(logits, y):
    return float(np.mean(logits.data.argmax(1) == np.asarray(y)))


def train_readable(d, seed, iterdir):
    rngd = np.random.default_rng(seed * 131 + 7)
    net = TinyLM(d["V"], K, D, rng=np.random.default_rng(seed),
                 lam_act=LAM_A, lam_w=LAM_W)
    opt = SGD(net.parameters(), lr=LR)
    best_v, best_ep, best_ws = -1.0, -1, None
    X, y = d["Xtr"], d["ytr"]
    for ep in range(EPOCHS):
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss, _ = net.loss(X[idx], y[idx])
            loss.backward()
            if net.frozen:
                for k in net.frozen:
                    net.concept.W.grad[:, k] = 0.0
                    net.concept.b.grad[k] = 0.0
            opt.step()
        v = acc_of(net.forward(d["Xva"])["logits"], d["yva"])
        if v > best_v:
            best_v, best_ep = v, ep
            best_ws = {p.name: p.data.copy() for p in net.parameters()}
    for p in net.parameters():
        p.data[:] = best_ws[p.name]
    net.save(os.path.join(iterdir, f"final_seed{seed}.npz"))
    return net, best_v, best_ep


def train_normal(d, seed, hidden, iterdir=ITERDIR):
    rngd = np.random.default_rng(seed * 77 + 3)
    net = tiny_normal(d["V"], D, hidden, rng=np.random.default_rng(seed))
    params = [net[0]] + [p for lyr in net[1:] for p in lyr.parameters()]
    opt = SGD(params, lr=LR)
    best_v, best_ws = -1.0, None
    X, y = d["Xtr"], d["ytr"]
    for ep in range(EPOCHS):
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = cross_entropy(normal_forward(net, X[idx]), y[idx])
            loss.backward()
            opt.step()
        v = acc_of(normal_forward(net, d["Xva"]), d["yva"])
        if v > best_v:
            best_v, best_ws = v, {p.name: p.data.copy() for p in params}
    for p in params:
        p.data[:] = best_ws[p.name]
    np.savez(os.path.join(iterdir, f"normal_seed{seed}.npz"),
             **{p.name: p.data for p in params})
    return net, best_v


def auto_name_t2(net, d, topn=8):
    """Names = words whose embeddings point along each concept row."""
    row = net.concept.W.data.T            # (K,D)
    emb = net.emb.data                    # (V,D)
    scores = row @ emb.T                  # (K,V)
    names = []
    for k in range(K):
        order = np.argsort(-scores[k])
        ws = [d["vocab"][j] for j in order
              if j < len(d["vocab"])            # skip <unk> column
              and scores[k][order[0]] > 0
              and d["vocab"][j] not in STOP
              and scores[k][j] > 0][:topn]
        names.append("_".join(ws[:6]) if ws else f"neuron{k}")
    return names


def _auroc(score, target01):
    pos = target01 > 0
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def audit_t2(net, names, d, Xva):
    """Standing rung-3 meter on windows: residualized AUROC (window contains
    a named word, beyond length prediction) AND logit name-contribution
    (sum of row·emb over present named words, mean over fired windows)."""
    X = np.asarray(Xva)
    h = net.forward(X)["concepts"].data
    z = net.forward(X)["concept_pre"].data
    L = np.log1p(np.asarray([(x != d["V"] - 1).sum() for x in X]))
    A = np.vstack([np.ones_like(L), L]).T
    row = net.concept.W.data.T; emb = net.emb.data
    vmap = {w: i for i, w in enumerate(d["vocab"])}
    rows = []
    for k, nm in enumerate(names):
        words = set(nm.split("_")) - {"neuron"}
        if not words or nm.startswith("neuron"):
            rows.append({"name": nm, "auroc": float("nan"),
                         "contrib": float("nan"), "flag": "NO-NAME"})
            continue
        ids = [vmap[w] for w in words if w in vmap]
        contains = np.stack([np.isin(X, i).any(axis=1) for i in ids]).any(0)
        coef, *_ = np.linalg.lstsq(A, h[:, k], rcond=None)
        resid = h[:, k] - A @ coef
        a = _auroc(resid, contains)
        fired = h[:, k] > 0.5
        contrib = 0.0
        if fired.any() and ids:
            tot = 0.0
            for i in ids:
                wmask = (X == i).any(axis=1)
                take = fired & wmask
                if take.any():
                    tot += float((row[k] @ emb[i]) * take.sum())
            contrib = tot / float(fired.sum())
        broken = not (a >= 0.55 and contrib >= 0.25)
        rows.append({"name": nm, "auroc": a, "contrib": contrib,
                     "pos_frac": float(contains.mean()),
                     "flag": "BROKEN" if broken else "-"})
    n_pass = sum(1 for r in rows if r["flag"] == "-")
    return rows, n_pass


def main():
    os.makedirs(ITERDIR, exist_ok=True)
    d = tiny_data.build(win=WIN)
    t0 = time.time()
    prior = float(np.bincount(d["ytr"], minlength=d["V"]).max()
                  / len(d["ytr"]))
    print(f"[t2-001] prior (most-frequent-word) = {prior*100:.2f}% | "
          f"V={d['V']} windows tr/va/te {d['Xtr'].shape[0]}/"
          f"{d['Xva'].shape[0]}/{d['Xte'].shape[0]}")

    rvals, nets = {}, {}
    for s in SEEDS:
        net, v, bep = train_readable(d, s, ITERDIR)
        rvals[s] = v * 100
        nets[s] = net
        print(f"  readable seed{s}: val {v*100:.2f}% (best ep {bep})")
    rm, rs = np.mean(list(rvals.values())), np.std(list(rvals.values()))

    hidden = match_h(d["V"], D, K)
    nvals = {}
    if not SKIP_NORMAL:
        for s in SEEDS:
            net, v = train_normal(d, s, hidden, ITERDIR)
            nvals[s] = v * 100
    nm = ns = float("nan")
    if nvals:
        nm, ns = np.mean(list(nvals.values())), np.std(list(nvals.values()))
    rp = readable_params(d["V"], D, K)
    np_ = normal_params(d["V"], D, hidden)
    print(f"[t2-001] readable VAL {rm:.2f} ± {rs:.2f} ({rp} params) vs "
          f"normal VAL {nm:.2f} ± {ns:.2f} ({np_} params)")

    net0 = nets[0]
    names = auto_name_t2(net0, d)
    rows, n_pass = audit_t2(net0, names, d, d["Xva"])
    for r in rows:
        print(f"    {r['name'][:40]:40s} auroc={r['auroc']:.2f} "
              f"contrib={r['contrib']:.3f} {r['flag']}")
    print(f"[t2-001] audit pass {n_pass}/{K}")

    okA = bool(rm >= 2.0 * prior * 100)
    okB = bool(n_pass / K >= 0.5)
    if okA and okB:
        # probe allowed only when A and B pass (pre-registered iter 002)
        def run_probe():
            rngd = np.random.default_rng(1 * 131 + 7)
            np2 = TinyLM(d["V"], K, D, rng=np.random.default_rng(1),
                         lam_act=LAM_A, lam_w=LAM_W)
            np2.freeze([2])
            opt = SGD(np2.parameters(), lr=LR)
            X, y = d["Xtr"], d["ytr"]
            for ep in range(EPOCHS):
                perm = rngd.permutation(len(X))
                for i in range(0, len(X), BATCH):
                    idx = perm[i:i + BATCH]
                    opt.zero_grad()
                    loss, _ = np2.loss(X[idx], y[idx])
                    loss.backward()
                    opt.step()
            pnames = auto_name_t2(np2, d)
            prows, _ = audit_t2(np2, pnames, d, d["Xva"])
            return prows[2]
        lie = run_probe()
        lie_caught = lie["flag"] == "BROKEN" or lie["auroc"] < 0.55
        print(f"[t2-002] lie probe: auroc={lie['auroc']:.2f} "
              f"contrib={lie['contrib']:.3f} -> CAUGHT={lie_caught}")
    else:
        lie_caught = False
        print("[t2-002] lie probe DEFERRED (A and B did not both pass)")
    verdict = "KEEP" if (okA and okB and lie_caught) else "REVERT"
    print(f"[t2-002] A(acc>={2*prior*100:.1f})={okA} B(>={n_pass/24:+.0%}"
          f")={okB} C={lie_caught} -> {verdict} ({time.time()-t0:.0f}s)")
    json.dump({"iteration": "t2-002", "task": "tiny_shakespeare",
               "mode": "discovered-lm", "K": K, "D": D, "WIN": WIN,
               "epochs": EPOCHS, "lr": LR, "lam_act": LAM_A, "lam_w": LAM_W,
               "change": "epochs 48 -> 96; readable-only (normal deferred "
                        "to gate stage per 2h cap); probe conditional on A+B",
               "prior": prior, "rval_mean": rm, "rval_std": rs,
               "nval_mean": nm, "nval_std": ns,
               "readable_params": rp, "normal_params": np_,
               "normal_hidden": hidden, "audit_pass": n_pass,
               "audit_total": K, "lie_caught": bool(lie_caught),
               "okA": okA, "okB": okB, "okC": bool(lie_caught), "verdict": verdict,
               "names": names},
              open(os.path.join(ITERDIR, "results.json"), "w"), indent=1)
    print("saved", os.path.join(ITERDIR, "results.json"))


if __name__ == "__main__":
    main()