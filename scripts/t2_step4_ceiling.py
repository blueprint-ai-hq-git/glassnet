"""t2_step4_ceiling.py — STEP 4: baseline + ceiling probe, then capacity.

Claude item 4: "'~18% is the task ceiling' is unproven. Add a majority-word
baseline, then test whether a bigger normal model or a longer window moves
val. Small runs only."

The majority-word baseline is already measured in Step 1 (16.32% on val).
This script answers the ceiling question with three probes, all VALIDATION
only, all against that baseline and the 18.3% readable level:

  P1 capacity : normal models at 1x / 3x / 7x the matched parameter count
                (hidden 24 / 72 / 168) -> does more capacity move val?
  P2 window   : longer context (WIN 16 and 24) with the matched-size normal
                -> does more context move val?
  P3 readable : the readable at 3x capacity (K=72) -> is the readable
                capacity-limited or structure-limited?

Small protocol: max 32 epochs, early stopping patience 8, lr 0.1, batch
128, seed 0 (probe scope, labelled as such). Test split never read.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
import numpy as np

from glassnet import tiny_data
from glassnet.tensor import relu, Tensor
from glassnet.layers import Dense
from glassnet.losses import cross_entropy
from glassnet.optim import SGD
from run_t2_iter003 import (K, D, WIN, LR, BATCH, LAM_A, TinyLM3, embedding,
                            _flatten)
from t2_step2_sweep import eval_chunked

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(HERE, "runs", "t2", "step4")
os.makedirs(RUNDIR, exist_ok=True)
OUT = os.path.join(RUNDIR, "results.json")
LOG = os.path.join(RUNDIR, "step4.log")

MAX_EPOCHS = int(os.environ.get("T2_MAX_EPOCHS", "32"))
PATIENCE = int(os.environ.get("T2_PATIENCE", "8"))


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


class NormalVar:
    """Normal LM with configurable hidden width and window length."""

    def __init__(self, V, hidden, D, W, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        s = D ** -0.5
        self.emb = Tensor(rng.standard_normal((V, D)).astype(np.float32) * s,
                          requires_grad=True, name="emb")
        self.h1 = Dense(D * W, hidden, rng=rng, name="h1")
        self.h2 = Dense(hidden, V, rng=rng, name="h2")

    def forward(self, ids):
        return self.h2(relu(self.h1(_flatten(embedding(self.emb, ids)))))

    def loss(self, ids, y):
        return cross_entropy(self.forward(ids), y)

    def parameters(self):
        return [self.emb] + self.h1.parameters() + self.h2.parameters()


def n_params_normal(V, hidden, D, W):
    return V * D + (D * W) * hidden + hidden + hidden * V + V


def train_eval(net, d, kind, seed, tag):
    rngd = np.random.default_rng(seed * 131 + 7)
    opt = SGD(net.parameters(), lr=LR)
    X, y = d["Xtr"], d["ytr"]
    fwd = (lambda A: net.forward(A)["logits"]) if kind == "readable" \
        else (lambda A: net.forward(A))
    best = {"val": -1.0, "epoch": 0, "w": None}
    bad = 0
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            if kind == "readable":
                loss, _ = net.loss(X[idx], y[idx])
            else:
                loss = net.loss(X[idx], y[idx])
            loss.backward()
            opt.step()
        v, _ = eval_chunked(fwd, d["Xva"], d["yva"])
        if v > best["val"]:
            best = {"val": v, "epoch": ep + 1,
                    "w": {p.name: p.data.copy() for p in net.parameters()}}
            bad = 0
        else:
            bad += 1
        print(f"    {tag} ep{ep+1:3d} val {v:5.2f}% (best {best['val']:.2f} "
              f"@{best['epoch']})", flush=True)
        if bad >= PATIENCE:
            break
    for p in net.parameters():
        p.data[:] = best["w"][p.name]
    secs = round(time.time() - t0)
    log(f"    {tag}: val {best['val']:.2f}% @ ep{best['epoch']} ({secs}s)")
    return best["val"], best["epoch"], secs


def main():
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    keys = {r["tag"] for r in done}

    def record(tag, **kw):
        rec = {"tag": tag, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
               **kw}
        done.append(rec)
        json.dump(done, open(OUT, "w"), indent=1)

    d8 = tiny_data.build(win=8)
    V = d8["V"]
    vals, cnts = np.unique(d8["ytr"], return_counts=True)
    maj = int(vals[cnts.argmax()])
    maj_val = float(np.mean(d8["yva"] == maj)) * 100
    log(f"STEP4: baseline (always predict '<unk>') = {maj_val:.2f}% on val; "
        f"readable level = 18.31%")

    # ---- P1: capacity sweep on the normal (matched / 3x / 7x) ----
    for hidden in (24, 72, 168):
        tag = f"P1_normal_h{hidden}"
        if tag in keys:
            continue
        net = NormalVar(V, hidden, D, 8, rng=np.random.default_rng(0))
        p = n_params_normal(V, hidden, D, 8)
        log(f"  {tag}: {p} params ({p/724401:.2f}x matched)")
        v, ep, secs = train_eval(net, d8, "normal", 0, tag)
        record(tag, probe="capacity", model="normal", hidden=hidden,
               params=p, ratio=p / 724401, val=v, best_epoch=ep, secs=secs)

    # ---- P3: readable at 3x capacity (K=72) ----
    tag = "P3_readable_K72"
    if tag not in keys:
        net = TinyLM3(V, 72, D, 8, rng=np.random.default_rng(0))
        p = V * D + (D * 8) * 72 + 72 + 72 * V + V
        log(f"  {tag}: {p} params ({p/724401:.2f}x matched)")
        v, ep, secs = train_eval(net, d8, "readable", 0, tag)
        record(tag, probe="capacity", model="readable", K=72, params=p,
               ratio=p / 724401, val=v, best_epoch=ep, secs=secs)

    # ---- P2: longer window with the matched normal ----
    for win in (16, 24):
        tag = f"P2_normal_win{win}"
        if tag in keys:
            continue
        dd = tiny_data.build(win=win)
        net = NormalVar(dd["V"], 24, D, win, rng=np.random.default_rng(0))
        p = n_params_normal(dd["V"], 24, D, win)
        log(f"  {tag}: {p} params, windows {len(dd['Xtr'])}")
        v, ep, secs = train_eval(net, dd, "normal", 0, tag)
        record(tag, probe="window", model="normal", win=win, params=p,
               val=v, best_epoch=ep, secs=secs)

    log("STEP4 SUMMARY")
    for r in done:
        log(f"  {r['tag']:22s} val {r['val']:6.2f}%  (baseline {maj_val:.2f}, "
            f"readable-ref 18.31)")
    json.dump({"runs": done, "majority_val": maj_val},
              open(OUT, "w"), indent=1)
    log("saved runs/t2/step4/results.json")


if __name__ == "__main__":
    main()