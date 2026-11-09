"""Batched state core: B chunks processed at once so the linear algebra is real BLAS work.

The single-chunk trainer does one 128-token chunk per step, which leaves the CPU mostly idle between
small matrix products. Here B chunks run as rows of one matrix, so every product is (B x H) @ (H x V).
Same equations, same gradients - only the batch shape changes, which is the honest way to measure a
speed-up: identical maths, different packing.
"""
import numpy as np


def zero_grads(p):
    return {k: np.zeros_like(v, dtype=np.float64) for k, v in p.items()}


def train_step(p, chunks, tgt_pos, UNK, lr, clip=1.0):
    """chunks: (B, T+1) int ids. Predicts token t+1 from positions t in tgt_pos. Returns loss, acc, gnorm."""
    B, L = chunks.shape
    D = p["emb"].shape[1]; H = p["Wh"].shape[0]; V = p["Wa"].shape[1]
    xs = p["emb"][chunks[:, :-1]].astype(np.float64)          # (B, T, D)
    T = L - 1
    hs = np.zeros((B, T + 1, H))
    for t in range(T):                                        # the only loop; every term is batched
        hs[:, t + 1] = np.tanh(xs[:, t] @ p["Wx"] + hs[:, t] @ p["Wh"] + p["b"])
    tg = chunks[:, 1:][:, tgt_pos]                            # (B, K)
    hsel = hs[:, np.asarray(tgt_pos) + 1]                     # (B, K, H)
    lg = hsel @ p["Wa"] + p["ba"]                             # (B, K, V)
    lg = lg.reshape(-1, V); tg = tg.reshape(-1)
    lg[:, UNK] = -1e9
    mx = lg.max(axis=1, keepdims=True); z = np.exp(lg - mx); pr = z / z.sum(axis=1, keepdims=True)
    loss = float(-np.log(pr[np.arange(len(tg)), tg] + 1e-12).mean())
    acc = float((pr.argmax(axis=1) == tg).mean())
    g = zero_grads(p)
    d = pr.copy(); d[np.arange(len(tg)), tg] -= 1.0; d /= len(tg)
    g["Wa"] += hsel.reshape(-1, H).T @ d
    g["ba"] += d.sum(axis=0)
    dlg = (d @ p["Wa"].T).reshape(B, -1, H)                   # (B, K, H)
    dh = np.zeros((B, T + 1, H))
    dh[:, np.asarray(tgt_pos) + 1] += dlg
    for t in range(T - 1, -1, -1):
        da = dh[:, t + 1] * (1.0 - hs[:, t + 1] ** 2)         # (B, H)
        g["b"] += da.sum(axis=0)
        g["Wx"] += xs[:, t].T @ da
        g["Wh"] += hs[:, t].T @ da
        # NOTE: np.add.at measured FASTER than a per-column np.bincount here (792 vs 641 scored
        # tokens/s at batch 32). Kept deliberately - see docs/history/SPEED-BATCHING.md.
        np.add.at(g["emb"], chunks[:, t], da @ p["Wx"].T)
        dh[:, t] += da @ p["Wh"].T
    norm = float(np.sqrt(sum((v ** 2).sum() for v in g.values())))
    s = min(1.0, clip / (norm + 1e-12))
    for k in p: p[k] = (p[k] - lr * s * g[k]).astype(np.float32)
    return loss, acc, norm


def renorm_recurrent(p, target=1.0):
    """Keep the recurrent loop stable: rescale Wh to a target spectral radius.

    Why: at radius 3.4 the loop saturates tanh (slope 0.005) and every prompt produced the SAME state,
    making the model input-blind (docs/history/INPUT-BLIND-STATECORE.md). Recomputing the radius costs
    one eigen-decomposition of an HxH matrix, which is negligible against the step itself.
    """
    W = p["Wh"].astype(np.float64)
    r = float(np.max(np.abs(np.linalg.eigvals(W))))
    if r > 0:
        p["Wh"] = (W * (target / r)).astype(p["Wh"].dtype)
    return r
