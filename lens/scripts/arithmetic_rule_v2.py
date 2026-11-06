"""LEARN ADDITION AS A RULE - fixed version.

WHAT WAS BROKEN IN THE LAST ATTEMPT (measured: unseen operands 0.0%, seen range 6.6%):
  1. addition has L+1 result digits (the last is the carry OUT) and I only supervised L of them - the most
     important digit was never trained.
  2. the loss was divided by n*L per step, so the effective learning rate was a third of what was set.
  3. no curriculum: straight to 2-digit operands with 3-digit results.
WHAT IT DOES NOW: trains on all L+1 digits, normalises properly, and walks a curriculum 1-digit -> 2-digit.
The ONLY metric that counts is accuracy on operands never seen in training. A memoriser scores 0 there.
"""
import os, json, time
import numpy as np

rng = np.random.default_rng(0)

def digs(n, L):
    return [(n // (10 ** i)) % 10 for i in range(L)]

def make(n, lo, hi, L):
    A = np.zeros((n, L), np.int64); B = np.zeros((n, L), np.int64); Y = np.zeros((n, L + 1), np.int64)
    i = 0
    while i < n:
        a = int(rng.integers(lo, hi)); b = int(rng.integers(lo, hi))
        A[i] = digs(a, L); B[i] = digs(b, L); Y[i] = digs(a + b, L + 1); i += 1
    return A, B, Y

def oh(A, B):
    n, L = A.shape
    X = np.zeros((n, L, 20), np.float32)
    for t in range(L):
        X[np.arange(n), t, A[:, t]] = 1.0
        X[np.arange(n), t, 10 + B[:, t]] = 1.0
    return X

def init(d_in=20, dh=96):
    return {"Wx": (rng.standard_normal((d_in, dh)) * np.sqrt(2.0 / d_in)).astype(np.float32),
            "Wh": (rng.standard_normal((dh, dh)) * np.sqrt(1.0 / dh)).astype(np.float32),
            "bh": np.zeros(dh, np.float32),
            "Wo": (rng.standard_normal((dh, 10)) * np.sqrt(1.0 / dh)).astype(np.float32),
            "bo": np.zeros(10, np.float32)}

def forward(p, X):
    """recurrent over digit positions; the carry lives in h. Returns L+1 heads:
       heads[t] for t < L predict result digit t; heads[L] predicts the carry-out digit."""
    n, L, _ = X.shape
    h = np.zeros((n, p["Wh"].shape[0]), np.float32)
    hs, zs = [], []
    for t in range(L):
        h = np.tanh(X[:, t, :] @ p["Wx"] + h @ p["Wh"] + p["bh"])
        hs.append(h); zs.append(h @ p["Wo"] + p["bo"])
    zs.append(h @ p["Wo"] + p["bo"])          # the carry-out head: same weights, one extra step
    hs.append(h)
    return hs, zs

def backward(p, X, Y, hs, zs, g):
    n, L, _ = X.shape
    for k in g: g[k][...] = 0.0
    dnext = np.zeros((n, p["Wh"].shape[0]), np.float32)
    total = n * (L + 1)                        # normalise over ALL output positions, incl. the carry out
    for t in reversed(range(L + 1)):
        pr = zs[t] - zs[t].max(1, keepdims=True); pr = np.exp(pr); pr /= pr.sum(1, keepdims=True)
        pr[np.arange(n), Y[:, t]] -= 1.0
        dz = pr / total
        g["Wo"] += hs[t].T @ dz; g["bo"] += dz.sum(0)
        if t == L:                             # carry-out head does not feed back into the recurrence
            continue
        dpre = (dz @ p["Wo"].T + dnext) * (1 - hs[t] ** 2)
        g["Wx"] += X[:, t, :].T @ dpre
        prev = hs[t - 1] if t > 0 else np.zeros((n, p["Wh"].shape[0]), np.float32)
        g["Wh"] += prev.T @ dpre; g["bh"] += dpre.sum(0)
        dnext = dpre @ p["Wh"].T
    return g

def acc(p, A, B, Y):
    _, zs = forward(p, oh(A, B)); L = A.shape[1]
    ok = np.ones(len(A), bool)
    for t in range(L + 1):
        ok &= (zs[t].argmax(1) == Y[:, t])
    return float(ok.mean())

def train(p, A, B, Y, epochs, lr, tag):
    g = {k: np.zeros_like(v) for k, v in p.items()}
    for ep in range(epochs):
        perm = rng.permutation(len(A))
        for i in range(0, len(A), 128):
            idx = perm[i:i+128]
            hs, zs = forward(p, oh(A[idx], B[idx]))
            backward(p, oh(A[idx], B[idx]), Y[idx], hs, zs, g)
            for k in p: p[k] -= lr * g[k]
    return p

L = 2
A1, B1, Y1 = make(8000, 0, 10, 1)             # curriculum stage 1: single digits
A2, B2, Y2 = make(20000, 0, 100, 2)           # stage 2: two digits
Ain, Bin, Yin = make(3000, 0, 100, 2)
Aout, Bout, Yout = make(3000, 100, 200, 2)    # operands 100-199: NEVER seen
p = init(); log, t0 = [], time.time()
A1b = np.zeros((len(A1), 2), np.int64); B1b = np.zeros((len(A1), 2), np.int64); Y1b = np.zeros((len(A1), 3), np.int64)
A1b[:, 0] = A1[:, 0]; B1b[:, 0] = B1[:, 0]; Y1b[:, 0] = Y1[:, 0]
print(json.dumps({"stage": "1-digit (carry-out included)"}), flush=True)
p = train(p, A1b, B1b, Y1b, 12, 0.05, "stage1")
print(json.dumps({"stage1_seen": round(acc(p, A1b, B1b, Y1b), 4)}), flush=True)
print(json.dumps({"stage": "2-digit"}), flush=True)
for block in range(6):
    p = train(p, A2, B2, Y2, 5, 0.03, f"block{block}")
    row = {"block": block + 1, "seen": round(acc(p, Ain, Bin, Yin), 4),
           "UNSEEN_100_199": round(acc(p, Aout, Bout, Yout), 4)}
    log.append(row); print(json.dumps(row), flush=True)
print("seconds:", round(time.time() - t0, 1), flush=True)
json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "ARITH2.json"), "w"), indent=1)
