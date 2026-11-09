"""Batched version of the direct-path core. Same maths as lens/scan_direct.py, with a batch dimension.

Correctness is not assumed: check_against_single() compares this batched gradient against the
single-chunk implementation in scan_direct.py on the same data. They must agree to ~1e-12.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lens import scan_direct as SD


def train_step(p, chunks, tgt_pos, UNK, lr, clip=1.0, target_radius=None, dec_lambda=0.0):
    B, L = chunks.shape
    D = p["emb"].shape[1]; H = p["Wh"].shape[0]; V = p["Wa"].shape[1]
    T = L - 1
    xs = p["emb"][chunks[:, :-1]].astype(np.float64)              # (B,T,D)
    hs = np.zeros((B, T + 1, H))
    for t in range(T):
        hs[:, t + 1] = np.tanh(xs[:, t] @ p["Wx"] + hs[:, t] @ p["Wh"] + p["b"])
    pre = xs @ p["Wd"]                                            # (B,T,D)
    dd_ = np.maximum(pre, 0.0)
    pos = np.asarray(tgt_pos)
    hsel = hs[:, pos + 1]                                         # (B,K,H)
    dsel = dd_[:, pos]                                            # (B,K,D)
    lg = hsel @ p["Wa"] + p["ba"] + dsel @ p["emb"].T             # (B,K,V)
    tg = chunks[:, 1:][:, pos].reshape(-1)
    n = len(tg)
    # <unk> is banned from prediction, so a position whose TARGET is <unk> is unlearnable: weight it 0
    # instead of scoring it. Scoring it made the analytic gradient disagree with finite differences on
    # some seeds (the banned logit makes the numeric slope zero while the one-hot slope is not).
    wts = (tg != UNK).astype(np.float64)
    wts = wts / max(wts.sum(), 1.0)
    lg2 = lg.reshape(-1, V)
    lg2[:, UNK] = -1e9
    mx = lg2.max(axis=1, keepdims=True); z = np.exp(lg2 - mx); pr = z / z.sum(axis=1, keepdims=True)
    loss = float(-(wts * np.log(pr[np.arange(n), tg] + 1e-12)).sum())
    acc = float(((pr.argmax(axis=1) == tg) & (wts > 0)).sum() / max((wts > 0).sum(), 1))
    g = {k: np.zeros_like(v, dtype=np.float64) for k, v in p.items()}
    dlg = pr.copy()
    dlg[np.arange(n), tg] -= 1.0
    dlg *= wts[:, None]
    dlgB = dlg.reshape(B, -1, V)
    # ---- state decorrelation: penalise the mean pairwise cosine of the FINAL scored state across the batch.
    # Without it, states collapse back to one direction under longer training (run 6: cosine 0.9716 at step 3750).
    dec_grad = None
    if dec_lambda > 0.0 and B > 1:
        hb = hs[:, int(pos[-1]) + 1]                      # (B,H) final scored state
        nrm = np.linalg.norm(hb, axis=1, keepdims=True) + 1e-12
        u = hb / nrm
        S = u @ u.T
        pairs = B * (B - 1)
        loss_dec = dec_lambda * (2.0 / pairs) * np.triu(S, 1).sum()
        G = (2.0 / pairs) * dec_lambda
        du = G * (u.sum(axis=0, keepdims=True) - u)        # dL/du_i = G * sum_{j != i} u_j
        dh_dec = (du - (u * du).sum(axis=1, keepdims=True) * u) / nrm
        dec_grad = dh_dec
        loss = loss + loss_dec
    g["Wa"] += hsel.reshape(-1, H).T @ dlg
    g["ba"] += dlg.sum(axis=0)
    # explicit gather: dlg rows and dsel rows are both (b,k)-ordered; use einsum so the pairing
    # is written down rather than implied by two independent reshapes
    g["emb"] += np.einsum("bkv,bkd->vd", dlgB, dsel.astype(np.float64))
    dpre = np.zeros((B, T, D))
    dp = np.einsum("bkv,vd->bkd", dlgB, p["emb"].astype(np.float64))   # (B,K,D)
    dpre[:, pos] = dp * (pre[:, pos] > 0)
    g["Wd"] += xs.reshape(-1, D).T @ dpre.reshape(-1, D)
    # pre[t,e] = sum_d xs[t,d] * Wd[d,e], so dL/dxs[t,d] = sum_e dpre[t,e] * Wd[d,e]
    dxs = np.einsum("bte,de->btd", dpre, p["Wd"].astype(np.float64))
    dh = np.zeros((B, T + 1, H))
    dh[:, pos + 1] += np.einsum("bkv,hv->bkh", dlgB, p["Wa"].astype(np.float64))
    if dec_grad is not None:
        dh[:, int(pos[-1]) + 1] += dec_grad   # Wa is (H,V)
    for t in range(T - 1, -1, -1):
        da = dh[:, t + 1] * (1.0 - hs[:, t + 1] ** 2)
        g["b"] += da.sum(axis=0)
        g["Wx"] += xs[:, t].T @ da
        g["Wh"] += hs[:, t].T @ da
        dxs[:, t] += da @ p["Wx"].T
        if t > 0:
            dh[:, t] += da @ p["Wh"].T
    for b in range(B):
        np.add.at(g["emb"], chunks[b, :-1], dxs[b])
    norm = float(np.sqrt(sum((v ** 2).sum() for v in g.values())))
    s = min(1.0, clip / (norm + 1e-12))
    for k in p: p[k] = (p[k] - lr * s * g[k]).astype(np.float32)
    return loss, acc, norm, g


def check_against_single(V=7, D=4, H=5, T=6, B=3, seed=3, tol=1e-11):
    """Batched grads for one chunk must equal the single-chunk implementation exactly."""
    p32 = SD.init(V, D, H, seed=seed)
    rng = np.random.default_rng(seed)
    ch = rng.integers(0, V, size=(B, T + 1))
    pos = np.arange(T)                       # score every position so the two losses are comparable
    p1 = {k: v.astype(np.float64) for k, v in p32.items()}
    _, g1 = SD.loss_and_grads(p1, ch[0, :-1], ch[0, 1:], mask=0)
    p2 = {k: v.astype(np.float64) for k, v in p32.items()}
    _, _, _, gb = train_step(p2, ch[:1], pos, 0, 0.0)          # zero lr: grads only
    worst = 0.0; where = ""
    for k in g1:
        a = np.asarray(gb[k]); b = np.asarray(g1[k])
        if a.shape != b.shape: return 1e9, f"shape {k} {a.shape} vs {b.shape}"
        diff = float(np.abs(a - b).max())
        if diff > worst: worst, where = diff, k
    return worst, where


def check_fd(V=6, D=3, H=4, T=5, B=2, seed=5, eps=1e-6, dec_lambda=0.0):
    """Project-standard gate: finite-difference the batched step itself on a tiny case."""
    p = {k: v.astype(np.float64) for k, v in SD.init(V, D, H, seed=seed).items()}
    rng = np.random.default_rng(seed)
    ch = rng.integers(0, V, size=(B, T + 1))
    pos = np.arange(T - 2, T)
    def loss_of(pp):
        l, a, n, g = train_step({k: v.copy() for k, v in pp.items()}, ch, pos, 0, 0.0, dec_lambda=dec_lambda)
        return l, g
    _, g = loss_of(p)
    worst = 0.0; where = ""
    for k in p:
        it = np.nditer(p[k], flags=["multi_index"])
        for _ in it:
            i = it.multi_index; o = p[k][i]
            p[k][i] = o + eps; lp, _ = loss_of(p)
            p[k][i] = o - eps; lm, _ = loss_of(p)
            p[k][i] = o
            num = (lp - lm) / (2 * eps); ana = g[k][i]
            d = abs(num - ana)
            if d > worst: worst, where = d, f"{k}{list(i)}"
    return worst, where


if __name__ == "__main__":
    w, where = check_fd()
    print(f"batched step finite-difference check: worst abs {w:.3e} at {where}")
    print("GATE:", "PASS" if w < 1e-6 else "FAIL")
