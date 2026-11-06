"""LEARN THE RULE, NOT THE TABLE - a compute-path model for addition, with actual carry propagation.

WHY THE OLD TEST FAILED (measured, 0.0% on unseen operands): a feed-forward net over the whole input has no
place to keep a carry and no way to process digit by digit. Addition is a LEFT-TO-RIGHT (least-significant
first) algorithm with one piece of state: the carry. So the model must do the same thing:
   step 1: look at the two least-significant digits
   step 2: predict digit 1 and update its carry state
   step 3: look at the next digits PLUS the carry, predict digit 2, update the carry
   ...
This is a recurrent net over digit positions. It can generalise to operands it never saw because the RULE is the
same regardless of which digits appear. Written in plain numpy with hand-written backprop through time.
"""
import os, json, time
import numpy as np

rng = np.random.default_rng(0)

def digits(n, L=3):
    """least-significant first, padded"""
    d = [(n // (10 ** i)) % 10 for i in range(L)]
    return d

def make(n, lo, hi, L=3):
    A = np.zeros((n, L), np.int64); B = np.zeros((n, L), np.int64); Y = np.zeros((n, L + 1), np.int64)
    for i in range(n):
        a = int(rng.integers(lo, hi)); b = int(rng.integers(lo, hi))
        A[i] = digits(a, L); B[i] = digits(b, L)
        s = digits(a + b, L + 1); Y[i] = s
    return A, B, Y

def onehot_digits(A, B):
    """per-position one-hot: [A digit, B digit] -> 20 features per step"""
    n, L = A.shape
    X = np.zeros((n, L, 20), np.float32)
    for i in range(n):
        for t in range(L):
            X[i, t, A[i, t]] = 1.0
            X[i, t, 10 + B[i, t]] = 1.0
    return X

def init(d_in=20, dh=64):
    s1, s2 = np.sqrt(2.0 / d_in), np.sqrt(2.0 / dh)
    return {"Wx": (rng.standard_normal((d_in, dh)) * s1).astype(np.float32), "Wh": (rng.standard_normal((dh, dh)) * s2).astype(np.float32),
            "bh": np.zeros(dh, np.float32), "Wo": (rng.standard_normal((dh, 10)) * s2).astype(np.float32), "bo": np.zeros(10, np.float32)}

def forward(p, X):
    """recurrent over digit positions; carry lives in the hidden state"""
    n, L, _ = X.shape; h = np.zeros((n, p["Wh"].shape[0]), np.float32)
    hs, zs = [], []
    for t in range(L):
        h = np.tanh(X[:, t, :] @ p["Wx"] + h @ p["Wh"] + p["bh"])
        hs.append(h); zs.append(h @ p["Wo"] + p["bo"])
    return hs, zs

def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def backward(p, X, Y, hs, zs, g):
    n, L, _ = X.shape; dh = p["Wh"].shape[0]
    for k in g: g[k][...] = 0.0
    dnext = np.zeros((n, dh), np.float32)
    # last step also predicts the FINAL carry digit (Y[:, L])
    for t in reversed(range(L)):
        pr = softmax(zs[t])
        target = Y[:, t] if t < L else Y[:, t]
        pr[np.arange(n), target] -= 1.0
        dz = pr / n / L
        g["Wo"] += hs[t].T @ dz; g["bo"] += dz.sum(0)
        dstate = dz @ p["Wo"].T + dnext
        dpre = dstate * (1 - hs[t] ** 2)
        g["Wx"] += X[:, t, :].T @ dpre
        prev_h = hs[t - 1] if t > 0 else np.zeros((n, dh), np.float32)
        g["Wh"] += prev_h.T @ dpre; g["bh"] += dpre.sum(0)
        dnext = dpre @ p["Wh"].T
    return g

def acc(p, A, B, Y):
    X = onehot_digits(A, B); _, zs = forward(p, X); n, L = A.shape
    ok = np.ones(n, bool)
    for t in range(L):
        ok &= (zs[t].argmax(1) == Y[:, t])
    return float(ok.mean())

Atr, Btr, Ytr = make(20000, 0, 50)
Ain, Bin, Yin = make(3000, 0, 50)
Aout, Bout, Yout = make(3000, 50, 100)     # operands NEVER seen in training
A3, B3, Y3 = make(2000, 500, 1000)          # 4-digit results, unseen magnitudes
p = init(); g = {k: np.zeros_like(v) for k, v in p.items()}
lr, log, t0 = 0.02, [], time.time()
for ep in range(40):
    perm = rng.permutation(len(Atr))
    for i in range(0, len(Atr), 256):
        idx = perm[i:i+256]
        X = onehot_digits(Atr[idx], Btr[idx])
        hs, zs = forward(p, X)
        backward(p, X, Ytr[idx], hs, zs, g)
        for k in p: p[k] -= lr * g[k]
    if ep % 4 == 3 or ep == 39:
        row = {"epoch": ep + 1, "seen_range": round(acc(p, Ain, Bin, Yin), 4),
               "UNSEEN_50_99": round(acc(p, Aout, Bout, Yout), 4),
               "unseen_500_999": round(acc(p, A3, B3, Y3), 4)}
        log.append(row); print(json.dumps(row), flush=True)
print("seconds:", round(time.time() - t0, 1), flush=True)
json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "ARITHMETIC_RNN.json"), "w"), indent=1)
