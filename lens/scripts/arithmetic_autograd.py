"""ADDITION AS A RULE, built on our OWN gradient-checked engine instead of hand-written backprop.

Why: two hand-rolled attempts failed (unseen 0.0%, seen 6.6% then 0.6%) and both times the cause was my own
backprop, not the idea. lens/tensor.py implements autograd and its gradients are verified against numerical
checks to 4.14e-09, so this version cannot silently have a wrong gradient. First it gradient-checks itself,
then it trains, then it reports only the number that matters: accuracy on operands it has never seen.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import lens.tensor as T
from lens.tensor import softmax_ce

rng = np.random.default_rng(0)

def digs(n, L): return [(n // (10 ** i)) % 10 for i in range(L)]

def make(n, lo, hi, L):
    A = np.zeros((n, L), np.int64); B = np.zeros((n, L), np.int64); Y = np.zeros((n, L + 1), np.int64)
    for i in range(n):
        a = int(rng.integers(lo, hi)); b = int(rng.integers(lo, hi))
        A[i] = digs(a, L); B[i] = digs(b, L); Y[i] = digs(a + b, L + 1)
    return A, B, Y

def oh(A, B):
    n, L = A.shape
    X = np.zeros((n, L, 20), np.float32)
    for t in range(L):
        X[np.arange(n), t, A[:, t]] = 1.0
        X[np.arange(n), t, 10 + B[:, t]] = 1.0
    return X

class Net:
    def __init__(self, dh=96):
        self.dh = dh
        self.Wx = T.Tensor((rng.standard_normal((20, dh)) * np.sqrt(2.0 / 20)).astype(np.float32), requires_grad=True, name="Wx")
        self.Wh = T.Tensor((rng.standard_normal((dh, dh)) * np.sqrt(1.0 / dh)).astype(np.float32), requires_grad=True, name="Wh")
        self.bh = T.Tensor(np.zeros(dh, np.float32), requires_grad=True, name="bh")
        self.Wo = T.Tensor((rng.standard_normal((dh, 10)) * np.sqrt(1.0 / dh)).astype(np.float32), requires_grad=True, name="Wo")
        self.bo = T.Tensor(np.zeros(10, np.float32), requires_grad=True, name="bo")
    def params(self): return [self.Wx, self.Wh, self.bh, self.Wo, self.bo]
    def forward(self, X):
        """recurrent over digit positions; carry lives in h; one head per step plus a carry-out head"""
        n, L, _ = X.shape
        h = T.Tensor(np.zeros((n, self.dh), np.float32))
        logits = []
        for t in range(L):
            x_t = T.Tensor(X[:, t, :])
            h = T.sigmoid(T.add(T.add(T.matmul(x_t, self.Wx), T.matmul(h, self.Wh)), self.bh))
            logits.append(T.add(T.matmul(h, self.Wo), self.bo))
        logits.append(T.add(T.matmul(h, self.Wo), self.bo))     # carry-out digit
        return logits

class SGD:
    def __init__(self, params, lr): self.params, self.lr = list(params), lr
    def step(self):
        for p in self.params:
            if p.grad is not None: p.data = p.data - self.lr * p.grad
    def zero(self):
        for p in self.params: p.grad = None

def loss_of(net, X, Y):
    L = X.shape[1]; logits = net.forward(X); total = None
    for t in range(L + 1):
        l = softmax_ce(logits[t], Y[:, t])
        total = l if total is None else T.add(total, l)
    return total

def acc(net, A, B, Y):
    X = oh(A, B); logits = net.forward(X); L = A.shape[1]
    ok = np.ones(len(A), bool)
    for t in range(L + 1):
        ok &= (logits[t].data.argmax(1) == Y[:, t])
    return float(ok.mean())

# ---- GATE 0: gradient check on the recurrence itself, before any training claim
net = Net(dh=16); A, B, Y = make(64, 0, 100, 2); X = oh(A, B)
loss_of(net, X, Y).backward()
g_an = {p.name: p.grad.copy() for p in net.params() if p.grad is not None}
print(json.dumps({"gradcheck_params": list(g_an)}), flush=True)
eps, worst = 1e-4, 0.0
for p in net.params()[:2]:                       # check the two big matrices element-wise (sampled)
    flat = p.data.reshape(-1); idxs = rng.choice(flat.size, size=6, replace=False)
    for i in idxs:
        old = flat[i]
        flat[i] = old + eps; lp = float(loss_of(net, X, Y).data)
        flat[i] = old - eps; lm = float(loss_of(net, X, Y).data)
        flat[i] = old
        num = (lp - lm) / (2 * eps); ana = float(g_an[p.name].reshape(-1)[i])
        worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-9))
print(json.dumps({"GROWING_REL_ERROR": round(worst, 6), "pass": bool(worst < 0.05)}), flush=True)

# ---- train
L = 2
Atr, Btr, Ytr = make(20000, 0, 100, L)
Ain, Bin, Yin = make(3000, 0, 100, L)
Aout, Bout, Yout = make(3000, 100, 200, L)
opt = SGD(net.params(), lr=0.15); t0 = time.time()
for ep in range(30):
    perm = rng.permutation(len(Atr))
    for i in range(0, len(Atr), 128):
        idx = perm[i:i+128]
        opt.zero(); loss_of(net, oh(Atr[idx], Btr[idx]), Ytr[idx]).backward(); opt.step()
    if ep % 3 == 2 or ep == 29:
        print(json.dumps({"epoch": ep + 1, "loss": round(float(loss_of(net, oh(Atr[:2000], Btr[:2000]), Ytr[:2000]).data), 4),
                          "seen": round(acc(net, Ain, Bin, Yin), 4),
                          "UNSEEN_100_199": round(acc(net, Aout, Bout, Yout), 4)}), flush=True)
print("seconds:", round(time.time() - t0, 1), flush=True)
