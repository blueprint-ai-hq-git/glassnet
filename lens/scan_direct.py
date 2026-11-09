"""State core WITH a direct token-to-logits path - the structural fix the shipped model already uses.

Why: four causes were eliminated by measurement (instability, capacity, bias, frequency prior) and the
readout still cannot turn ~0.01 nats of state signal into different words
(docs/history/FREQ-PRIOR-RULED-OUT.md). The published d512 model does not rely on the recurrent state
alone: it adds a direct path from the token vectors to the logits. This adds the same idea here:

    h_t   = tanh(x_t @ Wx + h_{t-1} @ Wh + b)          recurrent state
    d_t   = relu(x_t @ Wd)                             direct, per token
    logits = h_t @ Wa + ba + d_t @ emb.T               tied output, both paths

Every gradient is written by hand and checked numerically in check_grads().
"""
import numpy as np


def init(V, D, H, seed=0, dtype=np.float32):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lens import scan_tiny as S
    p = S.init(V, D, H, seed=seed, dtype=dtype)
    rng = np.random.default_rng(seed + 11)
    p["Wd"] = (rng.standard_normal((D, D)) * (1.0 / np.sqrt(D))).astype(dtype)
    return p


def loss_and_grads(p, ids, targets, mask=None):
    ids = np.asarray(ids, dtype=np.int64)
    T = len(ids)
    D = p["emb"].shape[1]; H = p["Wh"].shape[0]; V = p["Wa"].shape[1]
    xs = p["emb"][ids].astype(np.float64)                      # (T, D)
    hs = np.zeros((T + 1, H))
    for t in range(T):
        hs[t + 1] = np.tanh(xs[t] @ p["Wx"] + hs[t] @ p["Wh"] + p["b"])
    pre = xs @ p["Wd"]                                         # (T, D)
    d = np.maximum(pre, 0.0)
    lg = hs[1:] @ p["Wa"] + p["ba"] + d @ p["emb"].T           # (T, V)
    if mask is not None:
        lg[:, mask] = -1e9
    mx = lg.max(axis=1, keepdims=True); z = np.exp(lg - mx)
    pr = z / z.sum(axis=1, keepdims=True)
    loss = float(-np.log(pr[np.arange(T), targets] + 1e-12).mean())
    g = {k: np.zeros_like(v, dtype=np.float64) for k, v in p.items()}
    dlg = pr.copy(); dlg[np.arange(T), targets] -= 1.0; dlg /= T
    g["Wa"] += hs[1:].T @ dlg
    g["ba"] += dlg.sum(axis=0)
    g["emb"] += dlg.T @ d                                       # (V,T)@(T,D) = (V,D)
    dd = dlg @ p["emb"]                                        # (T, D)
    dpre = dd * (pre > 0)
    g["Wd"] += xs.T @ dpre
    dxs = dpre @ p["Wd"].T
    dh = dlg @ p["Wa"].T
    for t in range(T - 1, -1, -1):
        da = dh[t] * (1.0 - hs[t + 1] ** 2)
        g["b"] += da
        g["Wx"] += xs[t][:, None] @ da[None, :]
        g["Wh"] += hs[t][:, None] @ da[None, :]
        dxs[t] += da @ p["Wx"].T
        if t > 0:
            dh[t - 1] += da @ p["Wh"].T
    np.add.at(g["emb"], ids, dxs)
    return loss, g


def check_grads(V=5, D=3, H=4, T=4, seed=1, eps=1e-6):
    p = {k: v.astype(np.float64) for k, v in init(V, D, H, seed=seed, dtype=np.float64).items()}
    rng = np.random.default_rng(seed + 7)
    ids = rng.integers(0, V, size=T); tg = rng.integers(0, V, size=T)
    _, g = loss_and_grads(p, ids, tg)
    worst_abs = worst_rel = 0.0; where = ""
    for k in p:
        it = np.nditer(p[k], flags=["multi_index"])
        for _ in it:
            i = it.multi_index; o = p[k][i]
            p[k][i] = o + eps; lp, _ = loss_and_grads(p, ids, tg)
            p[k][i] = o - eps; lm, _ = loss_and_grads(p, ids, tg)
            p[k][i] = o
            num = (lp - lm) / (2 * eps); ana = g[k][i]
            a = abs(num - ana); r = a / max(1e-12, abs(num) + abs(ana))
            if a > worst_abs: worst_abs, worst_rel, where = a, r, f"{k}{list(i)}"
    return worst_abs, worst_rel, where


if __name__ == "__main__":
    ab, rl, w = check_grads()
    print(f"direct-path gradient check: worst abs {ab:.3e} worst rel {rl:.3e} at {w}")
    print("GATE:", "PASS" if ab < 1e-6 else "FAIL")
