"""train_audio.py — the AUDIO line's entire gate chain in one script.

1. PRE-GATE  concept-formula orthogonality: pairwise |corr| of label columns on TRAIN windows.
             Block > 0.6 (a neuron could not be label-honest), name 0.2-0.6 as risk.
2. TRAIN     manual-lr SGD on our own ops (lens.tensor), no optimizer library.
3. EVAL      per-epoch val accuracy (chance 1/6 = 16.7%) + per-concept AUROC on VAL windows.
4. STRICT    label-honesty audit post-training: |own| vs |best-other| > 0.10 -> 100% target.
5. EXPLAIN   exact-arithmetic reasons from val windows, printed for every class.
Everything logged to runs/audio/, checkpoint saved each epoch. BINARY gates, nothing assumed.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens.audio as A           # noqa: E402
from lens import tensor as T     # noqa: E402
from lens import backend as B    # noqa: E402


class SGD:
    """Plain gradient descent with a manual learning rate — our own optimizer, no library."""

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is None:
                raise RuntimeError(f"no gradient for {p.name}")
            p.data = p.data - self.lr * p.grad

CACHE = os.path.join(PROJ, "data", "audio", "audio_v1.npz")
META = os.path.join(PROJ, "data", "audio", "audio_v1_meta.json")
RUNS = os.path.join(PROJ, "lens", "runs", "audio")
SEED = 0
D, W, F = 64, 50, 65
BATCH = 128
EPOCHS = 40
LR = 0.4
LR_DECAY = 0.5
LAM_CONCEPT = 0.3


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def auroc(y, s):
    """Own rank-based AUROC (no library). y: 0/1, s: scores."""
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    tpr = np.cumsum(y) / npos
    fpr = np.cumsum(1 - y) / nneg
    return float(np.trapezoid(tpr, fpr))


def first_true_class(Y, order):
    """Class = first TRUE concept in the documented priority order. ACOUSTIC FIRST: has_voiced
    beats is_silence beats is_lecture, so a voiced lecture window is class has_voiced (the sound
    is the answer; the provenance is a reason, not a class)."""
    idx = np.argmax(Y[:, order], axis=1)          # first true column (they are exclusive-ish)
    return idx, [order[i] for i in idx]


def main():
    os.makedirs(RUNS, exist_ok=True)
    z = np.load(CACHE)
    X, Y, split = z["X"], z["Y"], z["split"]
    meta = json.load(open(META))
    concepts = meta["concepts"]
    K = len(concepts)
    names = list(concepts)
    tr = np.where(split == 0)[0]
    va = np.where(split == 1)[0]
    print(f"  audio data: X {X.shape}  Y {Y.shape}  train {len(tr)}  val {len(va)}  classes {K}")
    for i, nm in enumerate(names):
        print(f"    [{i}] {nm:<16} support train {Y[tr, i].mean():.3f}")

    # ---- PRE-GATE: formula orthogonality (never train a collinear set) ----
    Yt = Y[tr].astype(np.float64)
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            M[i, j] = abs(corr(Yt[:, i], Yt[:, j]))
    np.fill_diagonal(M, 0.0)
    bad = [(names[i], names[j], float(M[i, j])) for i in range(K) for j in range(i + 1, K)
           if M[i, j] > 0.6]
    risk = [(names[i], names[j], float(M[i, j])) for i in range(K) for j in range(i + 1, K)
            if 0.2 < M[i, j] <= 0.6]
    print("\n  PRE-GATE (train formulas):")
    for a, b, c in bad:
        print(f"    BLOCK {a} <-> {b} {c:.2f}")
    for a, b, c in risk[:10]:
        print(f"    risk  {a} <-> {b} {c:.2f}")
    if bad:
        print("  PRE-GATE: FAIL — collinear formulas; do not train this set.")
        sys.exit(2)

    # ---- model + classes ----
    rng = np.random.default_rng(SEED)
    C_order = [1, 0]  # has_voiced beats is_silence
    Ytr = np.ascontiguousarray(Y[tr].astype(np.float32))
    ytr, _ = first_true_class(Ytr, C_order)
    counts = np.bincount(ytr, minlength=K).astype(np.float64)
    class_weights = (len(ytr) / (K * counts)).astype(np.float32)
    print(f"  class balance train: {counts.astype(int)}  ce-class-weights: {class_weights}")
    concepts_obj = [type("C", (), {"name": n, "formula": meta["formulas"][n], "labels": None})()
                    for n in names]
    net = A.AudioNet(concepts_obj, d=D, window=W, f_bins=F, seed=SEED,
                     lam_concept=LAM_CONCEPT, class_order=C_order)
    print(f"  model: {net.n_params():,} params (d={D}, K={K}, W={W}, F={F})")

    opt = SGD(net.parameters(), lr=LR)
    nb = int(np.ceil(len(tr) / BATCH))
    log = {"args": {"d": D, "window": W, "f_bins": F, "batch": BATCH, "epochs": EPOCHS,
                    "lr": LR, "lr_decay": LR_DECAY, "lam_concept": LAM_CONCEPT, "seed": SEED},
           "history": [], "gates": {}}
    best_val, best_ep = 0.0, 0

    for ep in range(1, EPOCHS + 1):
        lr = LR * (1.0 - LR_DECAY * (ep - 1) / EPOCHS)
        opt.lr = lr
        perm = rng.permutation(len(tr))
        tot_ce = tot_ac = tot_n = 0.0
        correct = 0
        for b in range(nb):
            i0, i1 = b * BATCH, min((b + 1) * BATCH, len(tr))
            ids = perm[i0:i1]
            xb = X[tr][ids]
            yb = ytr[ids]
            opt.zero_grad()
            loss, parts = net.loss(xb, yb, Ytr[ids], class_weights=class_weights)
            loss.backward()
            opt.step()
            tot_ce += parts["ce"] * len(ids)
            tot_n += len(ids)
            logits, h = net.forward(T.Tensor(B.to_dev(xb)))
            pred = np.argmax(B.to_cpu(logits.data), axis=1)
            correct += int((pred == yb).sum())
        tr_acc = correct / tot_n
        # val
        nv = len(va)
        vlogits = np.zeros((nv, K), dtype=np.float32)
        vh = np.zeros((nv, K), dtype=np.float32)
        for b in range(int(np.ceil(nv / BATCH))):
            i0, i1 = b * BATCH, min((b + 1) * BATCH, nv)
            ids = va[i0:i1]
            xb = X[ids]
            lo, h = net.forward(T.Tensor(B.to_dev(xb)))
            vlogits[i0:i1] = B.to_cpu(lo.data)
            vh[i0:i1] = B.to_cpu(h.data)
        yva, _ = first_true_class(np.ascontiguousarray(Y[va].astype(np.float32)), C_order)
        va_acc = float(np.mean(np.argmax(vlogits, axis=1) == yva))
        aucs = {names[c]: round(auroc(Y[va, c].astype(np.float32), vh[:, c]), 4) for c in range(K)}
        rec = {"epoch": ep, "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4),
               "ce": round(tot_ce / tot_n, 4), "auc": aucs}
        log["history"].append(rec)
        print(f"  ep {ep:2d}/{EPOCHS}  lr {lr:.3f}  train_acc {tr_acc:.3f}  "
              f"val_acc {va_acc:.3f}  auc {aucs}")
        if va_acc > best_val:
            best_val, best_ep = va_acc, ep
            A.save(net, os.path.join(RUNS, f"audio_v1_ep{ep}.npz"))
            json.dump(log, open(os.path.join(RUNS, "audio_v1_log.json"), "w"), indent=1)

    # ---- gates ----
    print("\n" + "=" * 104)
    print("  GATES (pre-registered, binary)")
    g_acc = best_val > 0.5
    print(f"  g1 val_acc > 50% (chance 16.7%)  -> {'PASS' if g_acc else 'FAIL'}  ({best_val:.3f} @ ep{best_ep})")
    g_auc = all(v >= 0.8 for v in aucs.values())
    print(f"  g2 every concept AUROC >= 0.8     -> {'PASS' if g_auc else 'FAIL'}  {aucs}")

    # ---- STRICT label-honesty audit on val (h vs ALL formulas from the cache = the same Y) ----
    print("\n  STRICT LABEL-HONESTY (val windows, |own| vs |best-other| margin > 0.10):")
    rows = []
    for i in range(K):
        own = corr(vh[:, i], Y[va, i])
        other_abs, jb = max(((abs(corr(vh[:, i], Y[va, j])), j) for j in range(K) if j != i),
                            default=(0.0, -1))
        honest = own - other_abs > 0.10
        mis = other_abs - own > 0.10
        rows.append((names[i], round(own, 3), round(other_abs, 3),
                     names[jb] if jb >= 0 else "", "honest" if honest else ("MISLABELLED" if mis else "weak")))
        print(f"    {names[i]:<16} own {own:6.3f}  best-other {other_abs:6.3f}  ({names[jb] if jb >= 0 else '':<16})  {rows[-1][4]}")
    n_hon = sum(1 for r in rows if r[4] == "honest")
    g_strict = n_hon == K
    print(f"  g3 strict label-honesty = 100%    -> {'PASS' if g_strict else 'FAIL'}  ({n_hon}/{K})")

    # ---- explain examples ----
    print("\n  explain() examples (exact arithmetic, val windows; class names via the priority order):")
    per_class = {}
    for i in np.random.default_rng(1).choice(len(va), size=min(12, len(va)), replace=False):
        idx = va[i]
        e = A.explain(net, X[idx:idx + 1], top_k=3)
        per_class.setdefault(e["class"], 0)
        print(f"    truth-class {net.answer_classes[yva[i]]:<12} -> {e['class_name']:<12} "
              f"reasons {[f'{n}:{s:.2f}' for n, s in e['reasons']]}")

    verdict = g_acc and g_auc and g_strict
    log["gates"] = {"g1_val_acc": bool(g_acc), "g2_auroc": bool(g_auc),
                    "g3_strict_100": bool(g_strict), "overall": bool(verdict)}
    json.dump(log, open(os.path.join(RUNS, "audio_v1_log.json"), "w"), indent=1)
    print(f"\n  AUDIO LINE VERDICT: {'PASS — 100% readable, all gates green' if verdict else 'FAIL — gates above name the gaps'}")
    sys.exit(0 if verdict else 2)


if __name__ == "__main__":
    main()