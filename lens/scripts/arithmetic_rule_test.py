"""Compute or memorise? Addition generalisation test. Written in PLAIN NUMPY with hand-written backprop,
because this project's Dense layer is not autograd-compatible (its weights are raw arrays used by a hand-written
backward pass), and the point here is the RESULT, not the plumbing.

Train: every operand < 50.  Test: operands 50-99, never seen.  A memoriser scores ~0 there.
"""
import os, json, time
import numpy as np

MAXLEN, ND, RES = 4, 11, 10
rng = np.random.default_rng(0)

def enc(a, b):
    s = f"{a}+{b}"; toks = [int(c) if c.isdigit() else 10 for c in s]
    x = np.zeros(MAXLEN, dtype=np.int64); x[:min(len(toks), MAXLEN)] = toks[:MAXLEN]
    return x

def make(n, lo, hi):
    X = np.zeros((n, MAXLEN), dtype=np.int64); Y = np.zeros((n, 3), dtype=np.int64); i = 0
    while i < n:
        a, b = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        s = str(a + b)
        if len(s) > 3: continue
        X[i] = enc(a, b); Y[i] = [int(c) for c in s.rjust(3, "0")]; i += 1
    return X, Y

def onehot(X):
    oh = np.zeros((len(X), MAXLEN * ND), dtype=np.float32)
    for j in range(MAXLEN): oh[np.arange(len(X)), j * ND + X[:, j]] = 1.0
    return oh

def init(fin, fh=128):
    return {"W1": (rng.standard_normal((fin, fh)) * np.sqrt(2.0 / fin)).astype(np.float32), "b1": np.zeros(fh, np.float32),
            "W2": (rng.standard_normal((fh, fh)) * np.sqrt(2.0 / fh)).astype(np.float32), "b2": np.zeros(fh, np.float32),
            "W3": (rng.standard_normal((fh, 3 * RES)) * np.sqrt(2.0 / fh)).astype(np.float32), "b3": np.zeros(3 * RES, np.float32)}

def forward(p, x):
    h1 = np.maximum(x @ p["W1"] + p["b1"], 0); h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
    return h1, h2, (h2 @ p["W3"] + p["b3"]).reshape(len(x), 3, RES)

def backward(p, x, h1, h2, z, Y, g):
    N = len(x); dz = np.zeros((N, 3, RES), np.float32)
    for pos in range(3):
        e = np.exp(z[:, pos, :] - z[:, pos, :].max(1, keepdims=True)); pr = e / e.sum(1, keepdims=True)
        pr[np.arange(N), Y[:, pos]] -= 1.0; dz[:, pos, :] = pr / N / 3
    dz = dz.reshape(N, -1)
    g["W3"] = h2.T @ dz; g["b3"] = dz.sum(0)
    d2 = (dz @ p["W3"].T) * (h2 > 0); g["W2"] = h1.T @ d2; g["b2"] = d2.sum(0)
    d1 = (d2 @ p["W2"].T) * (h1 > 0); g["W1"] = x.T @ d1; g["b1"] = d1.sum(0)

def acc(p, X, Y):
    _, _, z = forward(p, onehot(X)); pr = z.argmax(2)
    return float((pr == Y).all(1).mean()), float((pr == Y).mean())

def loss_of(p, X, Y):
    _, _, z = forward(p, onehot(X)); N = len(X); tot = 0.0
    for pos in range(3):
        e = np.exp(z[:, pos, :] - z[:, pos, :].max(1, keepdims=True)); pr = e / e.sum(1, keepdims=True)
        tot += -np.log(pr[np.arange(N), Y[:, pos]] + 1e-12).mean()
    return tot / 3

Xtr, Ytr = make(20000, 0, 50)
Xin, Yin = make(4000, 0, 50)
Xout, Yout = make(4000, 50, 100)
X3, Y3 = make(2000, 100, 500)
p = init(MAXLEN * ND); g = {k: np.zeros_like(v) for k, v in p.items()}
lr, log, t0 = 0.05, [], time.time()
for ep in range(60):
    perm = rng.permutation(len(Xtr))
    for i in range(0, len(Xtr), 256):
        idx = perm[i:i+256]; x = onehot(Xtr[idx])
        h1, h2, z = forward(p, x); backward(p, x, h1, h2, z, Ytr[idx], g)
        for k in p: p[k] -= lr * g[k]
    if ep % 5 == 4 or ep == 59:
        row = {"epoch": ep + 1, "loss": round(float(loss_of(p, Xtr[:4000], Ytr[:4000])), 4),
               "train_exact": round(float(acc(p, Xtr[:4000], Ytr[:4000])[0]), 4),
               "seen_range": round(float(acc(p, Xin, Yin)[0]), 4),
               "UNSEEN_50_99": round(float(acc(p, Xout, Yout)[0]), 4),
               "three_digit": round(float(acc(p, X3, Y3)[0]), 4)}
        log.append(row); print(json.dumps(row), flush=True)
print("seconds:", round(time.time() - t0, 1))
json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "ARITHMETIC.json"), "w"), indent=1)
