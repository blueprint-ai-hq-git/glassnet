"""Per-digit diagnostic: is the RULE forming, or is nothing learning?

Exact-match over 3 digits hides progress (0.7^3 = 34%). This reports each digit separately and separates
digit 0 (no carry needed) from digit 1 and the carry-out (carry needed). If digit 0 is near perfect and
digit 1 lags, the rule is forming and the model needs capacity. If digit 0 is at chance, the design is wrong.
Trained with the verified autograd engine, more capacity, more steps.
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
    n, L = A.shape; X = np.zeros((n, L, 20), np.float32)
    for t in range(L):
        X[np.arange(n), t, A[:, t]] = 1.0; X[np.arange(n), t, 10 + B[:, t]] = 1.0
    return X
class Net:
    def __init__(self, dh=192):
        self.dh = dh
        self.Wx = T.Tensor((rng.standard_normal((20, dh)) * np.sqrt(2.0/20)).astype(np.float32), requires_grad=True, name="Wx")
        self.Wh = T.Tensor((rng.standard_normal((dh, dh)) * np.sqrt(1.0/dh)).astype(np.float32), requires_grad=True, name="Wh")
        self.bh = T.Tensor(np.zeros(dh, np.float32), requires_grad=True, name="bh")
        self.Wo = T.Tensor((rng.standard_normal((dh, 10)) * np.sqrt(1.0/dh)).astype(np.float32), requires_grad=True, name="Wo")
        self.bo = T.Tensor(np.zeros(10, np.float32), requires_grad=True, name="bo")
    def params(self): return [self.Wx, self.Wh, self.bh, self.Wo, self.bo]
    def forward(self, X):
        n, L, _ = X.shape
        h = T.Tensor(np.zeros((n, self.dh), np.float32)); out = []
        for t in range(L):
            h = T.sigmoid(T.add(T.add(T.matmul(T.Tensor(X[:, t, :]), self.Wx), T.matmul(h, self.Wh)), self.bh))
            out.append(T.add(T.matmul(h, self.Wo), self.bo))
        out.append(T.add(T.matmul(h, self.Wo), self.bo))
        return out
class SGD:
    def __init__(self, ps, lr): self.ps, self.lr = list(ps), lr
    def step(self):
        for p in self.ps:
            if p.grad is not None: p.data = p.data - self.lr * p.grad
    def zero(self):
        for p in self.ps: p.grad = None
def loss_of(net, X, Y):
    L = X.shape[1]; lg = net.forward(X); tot = None
    for t in range(L + 1):
        l = softmax_ce(lg[t], Y[:, t]); tot = l if tot is None else T.add(tot, l)
    return tot
def per_digit(net, A, B, Y):
    lg = net.forward(oh(A, B)); L = A.shape[1]
    accs = [float((lg[t].data.argmax(1) == Y[:, t]).mean()) for t in range(L + 1)]
    ok = np.ones(len(A), bool)
    for t in range(L + 1): ok &= (lg[t].data.argmax(1) == Y[:, t])
    return [round(a, 4) for a in accs], round(float(ok.mean()), 4)
L = 2
Atr, Btr, Ytr = make(20000, 0, 100, L)
Ain, Bin, Yin = make(3000, 0, 100, L)
Aout, Bout, Yout = make(3000, 100, 200, L)
net = Net(); opt = SGD(net.params(), lr=0.2); t0 = time.time()
for ep in range(45):
    perm = rng.permutation(len(Atr))
    for i in range(0, len(Atr), 128):
        idx = perm[i:i+128]
        opt.zero(); loss_of(net, oh(Atr[idx], Btr[idx]), Ytr[idx]).backward(); opt.step()
    if ep % 5 == 4 or ep == 44:
        pd, exact = per_digit(net, Ain, Bin, Yin)
        pdo, exacto = per_digit(net, Aout, Bout, Yout)
        print(json.dumps({"epoch": ep+1, "seen[d0,d1,carry]": pd, "seen_exact": exact,
                          "UNSEEN[d0,d1,carry]": pdo, "UNSEEN_exact": exacto,
                          "loss": round(float(loss_of(net, oh(Atr[:2000], Btr[:2000]), Ytr[:2000]).data), 3)}), flush=True)
print("seconds:", round(time.time()-t0, 1), flush=True)
