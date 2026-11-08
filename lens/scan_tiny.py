"""lens/scan_tiny.py - a state-carrying (recurrent) core: constant parameters, ANY input length.

WHY THIS EXISTS
The current stack flattens a fixed 32-token window into the input layer, so the weights are shaped
for exactly 32 tokens. That is why 8M windows never gave 8M context: 8M was a count of 32-token
training slices, not a window. Here the SAME weights are applied to one token at a time while a
state is carried forward:

    x_t = emb[token_t]                       # (D,)
    h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)  # (H,)  state
    logits_t = h_t @ Wa + ba                 # (V,)  answer

Parameter count is independent of length. Length is limited only by compute, which is what
"unlimited context" can honestly mean on a small model.

Every gradient below is written by hand and numerically checked in check_grads().
"""
import numpy as np


def init(V, D, H, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    s = lambda a, b, k: (rng.standard_normal((a, b)) * k).astype(dtype)
    return {
        "emb": s(V, D, 0.1), "Wx": s(D, H, 1.0 / np.sqrt(D)), "Wh": s(H, H, 1.0 / np.sqrt(H)),
        "b": np.zeros(H, dtype=dtype), "Wa": s(H, V, 1.0 / np.sqrt(H)), "ba": np.zeros(V, dtype=dtype),
    }


def forward(p, ids, keep=False):
    """ids: 1-D int array of ANY length. Returns (loss, cache)."""
    ids = np.asarray(ids, dtype=np.int64)
    T = len(ids)
    V, H = p["Wa"].shape[1], p["Wh"].shape[0]
    hs = np.zeros((T + 1, H), dtype=np.float64)          # h[0] = 0, h[t] is state AFTER token t
    xs = p["emb"][ids].astype(np.float64)                 # (T, D)
    for t in range(T):
        hs[t + 1] = np.tanh(xs[t] @ p["Wx"] + hs[t] @ p["Wh"] + p["b"])
    lg = hs[1:] @ p["Wa"] + p["ba"]                       # (T, V)
    return lg, {"ids": ids, "xs": xs, "hs": hs, "T": T}, V, H


def loss_and_grads(p, ids, targets):
    """Mean cross-entropy over all positions, gradients by hand (BPTT)."""
    lg, c, V, H = forward(p, ids)
    T = c["T"]
    m = lg.max(axis=1, keepdims=True)
    z = np.exp(lg - m)
    pr = z / z.sum(axis=1, keepdims=True)
    loss = float(-np.log(pr[np.arange(T), targets] + 1e-12).mean())
    g = {k: np.zeros_like(v, dtype=np.float64) for k, v in p.items()}
    dlg = pr.copy()
    dlg[np.arange(T), targets] -= 1.0
    dlg /= T
    g["Wa"] += c["hs"][1:].T @ dlg
    g["ba"] += dlg.sum(axis=0)
    dh = dlg @ p["Wa"].T                                   # dL/dh_t, accumulated from above
    for t in range(T - 1, -1, -1):                         # backprop through time
        da = dh[t] * (1.0 - c["hs"][t + 1] ** 2)           # tanh derivative
        g["b"] += da
        g["Wx"] += c["xs"][t][:, None] @ da[None, :]
        g["Wh"] += c["hs"][t][:, None] @ da[None, :]
        np.add.at(g["emb"], c["ids"][t], da @ p["Wx"].T)
        if t > 0:
            dh[t - 1] += da @ p["Wh"].T
    return loss, g


def check_grads(V=5, D=3, H=4, T=4, seed=1, eps=1e-6):
    """Numerical vs analytic on every parameter. The gate: nothing trains before this passes."""
    p = {k: v.astype(np.float64) for k, v in init(V, D, H, seed=seed, dtype=np.float64).items()}
    rng = np.random.default_rng(seed + 7)
    ids = rng.integers(0, V, size=T)
    tg = rng.integers(0, V, size=T)
    _, g = loss_and_grads(p, ids, tg)
    worst_abs, worst_rel, where = 0.0, 0.0, ""
    for k in p:
        it = np.nditer(p[k], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            o = p[k][i]
            p[k][i] = o + eps; lp, _ = loss_and_grads(p, ids, tg)
            p[k][i] = o - eps; lm, _ = loss_and_grads(p, ids, tg)
            p[k][i] = o
            num = (lp - lm) / (2 * eps)
            ana = g[k][i]
            a = abs(num - ana); r = a / max(1e-12, abs(num) + abs(ana))
            if a > worst_abs:
                worst_abs, worst_rel, where = a, r, f"{k}{list(i)}"
    return worst_abs, worst_rel, where


if __name__ == "__main__":
    ab, rl, w = check_grads()
    print(f"gradient check: worst abs {ab:.3e}  worst rel {rl:.3e}  at {w}")
    print("GATE 1:", "PASS" if ab < 1e-6 else "FAIL")
    # length independence: same weights, 3 tokens then 5000 tokens
    p = init(50, 8, 16, seed=0)
    for T in (3, 32, 1000, 5000):
        lg, c, V, H = forward(p, np.arange(T) % 50)
        print(f"  length {T:>5}: logits {lg.shape}, params {sum(v.size for v in p.values())}")
