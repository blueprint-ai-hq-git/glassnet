"""glassnet.arith_model — the readable model for the verifiable task.

Same shape as the text model that produced Targets C/D/E: an embedding, a NAMED concept layer,
then an answer layer that sees ONLY the concepts. So every reason for every answer is a concept a
human can name, and coverage is 100% by construction.

The window is 6 tokens. For a problem a+b the model answers in two steps:
    [PAD] [BOS] a + b =        ->  tens digit
    [BOS]  a   + b =  tens     ->  units digit
"""
import numpy as np

from glassnet.tensor import Tensor, matmul, sigmoid, set_backend   # noqa: F401
from glassnet.layers import Dense
from glassnet.optim import SGD
from glassnet import arith


class ArithNet:
    def __init__(self, k=8, d=64, window=6, readable=True, seed=0, lam_concept=0.1):
        self.V, self.K, self.D, self.W = arith.V, k, d, window
        self.readable = bool(readable)
        self.lam_concept = lam_concept if readable else 0.0
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.emb = Tensor(rng.standard_normal((arith.V, d)).astype(np.float32) * d ** -0.5,
                          requires_grad=True, name="emb")
        self.concept = Dense(d * window, k, rng=rng, name="concept")
        self.answer = Dense(k, arith.V, rng=rng, name="answer")

    def parameters(self):
        return [self.emb] + self.concept.parameters() + self.answer.parameters()

    # ------------------------------------------------------------------ forward
    def forward(self, ids):
        ids = np.asarray(ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        E = self.emb.data[ids]
        x = Tensor(E.reshape(len(ids), -1), requires_grad=self.emb.requires_grad,
                   _prev=(self.emb,))

        def emb_bw():
            """Scatter the window gradient back into the embedding rows that were looked up.

            Without this the reshape node has no backward and the embedding never receives a
            gradient, which the optimiser catches as "no gradient for 'emb'".
            """
            if x.grad is None:
                return
            g = np.asarray(x.grad).reshape(len(ids), self.W, self.D)
            acc = np.zeros_like(self.emb.data)
            np.add.at(acc, ids, g)
            self.emb.grad = acc if self.emb.grad is None else self.emb.grad + acc
        x._backward = emb_bw

        h = sigmoid(self.concept(x))
        logits = self.answer(h)
        return {"concepts": h, "logits": logits}

    def graph_loss(self, ids, y, labels=None):
        """LM loss + (optionally) the concept loss. Real autograd path, used for training."""
        out = self.forward(ids)
        logits = out["logits"]
        z = logits.data - logits.data.max(1, keepdims=True)
        ex = np.exp(z)
        probs = ex / ex.sum(1, keepdims=True)
        lm = Tensor(float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)).mean()),
                    requires_grad=True, _prev=(logits,))

        def lm_bw():
            d = probs.copy()
            d[np.arange(len(y)), y] -= 1.0
            g = lm.grad * (d / len(y))
            logits.grad = g if logits.grad is None else logits.grad + g
        lm._backward = lm_bw

        if self.readable and labels is not None:
            h = out["concepts"]
            hd = np.asarray(h.data)
            tgt = np.asarray(labels, dtype=np.float32)
            eps = 1e-7
            aux = Tensor(float(-(tgt * np.log(hd + eps) + (1 - tgt) * np.log(1 - hd + eps)).mean()),
                         requires_grad=True, _prev=(h,))

            def aux_bw():
                dh = ((hd - tgt) / (hd * (1 - hd) + eps)) / hd.size
                g = aux.grad * self.lam_concept * dh
                h.grad = g if h.grad is None else h.grad + g
            aux._backward = aux_bw
            return lm + aux * self.lam_concept
        return lm

    # ------------------------------------------------------------------ inference
    def logits(self, ids):
        return np.asarray(self.forward(ids)["logits"].data)

    def answer_problem(self, a, b, greedy=True):
        """Run the two-step answer for one problem. Returns (tens, units) as token ids."""
        w1 = np.array([[arith.TOKEN[arith.PAD], arith.TOKEN[arith.BOS],
                        arith.TOKEN[str(a)], arith.TOKEN[arith.PLUS],
                        arith.TOKEN[str(b)], arith.TOKEN[arith.EQ]]])
        t = int(self.logits(w1)[0].argmax())
        w2 = np.array([[arith.TOKEN[arith.BOS], arith.TOKEN[str(a)], arith.TOKEN[arith.PLUS],
                        arith.TOKEN[str(b)], arith.TOKEN[arith.EQ], t]])
        u = int(self.logits(w2)[0].argmax())
        return t, u

    def sample_problem(self, a, b, rng, temperature=1.0):
        """Stochastic answer, so the teacher can be right sometimes and wrong other times."""
        w1 = np.array([[arith.TOKEN[arith.PAD], arith.TOKEN[arith.BOS],
                        arith.TOKEN[str(a)], arith.TOKEN[arith.PLUS],
                        arith.TOKEN[str(b)], arith.TOKEN[arith.EQ]]])
        t = _pick(self.logits(w1)[0], rng, temperature)
        w2 = np.array([[arith.TOKEN[arith.BOS], arith.TOKEN[str(a)], arith.TOKEN[arith.PLUS],
                        arith.TOKEN[str(b)], arith.TOKEN[arith.EQ], t]])
        u = _pick(self.logits(w2)[0], rng, temperature)
        return t, u

    def accuracy(self, pairs):
        """Exact-answer accuracy over (a, b) pairs. The only metric that matters here."""
        ok = 0
        for a, b in pairs:
            t, u = self.answer_problem(a, b)
            ok += int(t * 10 + u == a + b)
        return ok / max(len(pairs), 1)

    def perplexity(self, X, y):
        lg = self.logits(X)
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(1, keepdims=True)
        return float(np.exp(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()))

    def coverage(self, X):
        out = self.forward(X)
        h = np.asarray(out["concepts"].data)
        lg = np.asarray(out["logits"].data)
        W = self.answer.W.data
        pred = lg.argmax(1)
        mass = np.abs(h * W[:, pred].T)
        return float(mass.sum() / max(mass.sum(), 1e-12))     # every column is named by design

    def concept_auroc(self, X, labels):
        """Does each named concept fire on the thing its name claims? (the standing meter)"""
        h = np.asarray(self.forward(X)["concepts"].data)
        rows = []
        for c in range(self.K):
            rows.append(_auroc(h[:, c], np.asarray(labels)[:, c].astype(bool)))
        return rows


def _pick(logits, rng, temperature):
    z = logits / max(temperature, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def _auroc(score, target01):
    order = np.argsort(score)
    r = np.empty(len(score), dtype=float)
    r[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(target01.sum()), int((~target01).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[target01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train(net, X, y, labels=None, epochs=40, lr=1.0, batch=256, seed=0, verbose=False):
    opt = SGD(net.parameters(), lr)
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X))
        tot = 0.0
        nb = 0
        for s in range(0, len(X), batch):
            b = idx[s:s + batch]
            opt.zero_grad()
            lab = None if labels is None else np.asarray(labels)[b]
            loss = net.graph_loss(X[b], y[b], lab)
            loss.backward()
            opt.step()
            tot += float(loss.data)
            nb += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"    ep {ep:>3} loss {tot/max(nb,1):.4f}")
    return net
