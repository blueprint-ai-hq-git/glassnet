"""train_t2_normal_concat.py — the MISSING CONTROL for T2.

The readable concat model (724,401 params) has been pinned at ~18.3% VAL.
Without a matched normal network we cannot say whether that is a
readable-architecture failure or simply the task ceiling for a 724k-param
model. This trains the exact matched control:

  readable : emb(V,64) -> concat window (512) -> sigmoid concepts (24)
             -> 8001-way logits      = V*D + 512*24+24 + 24*8001+8001
  normal   : emb(V,64) -> concat window (512) -> relu hidden (24)
             -> 8001-way logits      = V*D + 512*24+24 + 24*8001+8001
  BOTH     = 724,401 parameters exactly. Same epochs/lr/batch/seed/data.

Difference is ONLY the readable structure (named concept layer + sparsity
penalty + sigmoid) vs a plain relu bottleneck. VALIDATION only; the test
split is never read. Kill-resilient: checkpoint + resume every 16 epochs.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
import numpy as np

from glassnet import tiny_data
from glassnet.tensor import Tensor, relu
from glassnet.layers import Dense
from glassnet.losses import cross_entropy
from glassnet.optim import SGD
from run_t2_iter003 import (K, D, WIN, EPOCHS, LR, BATCH, embedding, _flatten,
                            acc_of)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(HERE, "runs", "t2", "normal_concat")
os.makedirs(RUNDIR, exist_ok=True)
SEED = int(os.environ.get("T2_NORMAL_SEED", "0"))
SAVE_EVERY = int(os.environ.get("T2_SAVE_EVERY", "16"))


class TinyNormal:
    """Matched control: embedding + concat window + relu bottleneck."""

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


def n_params(V, hidden, D, W):
    return V * D + (D * W) * hidden + hidden + hidden * V + V


def main():
    d = tiny_data.build(win=WIN)
    n_p = n_params(d["V"], K, D, WIN)
    print(f"normal control: emb+concat+relu({K}); params {n_p} "
          f"(readable = 724401)", flush=True)
    rng = np.random.default_rng(SEED)
    net = TinyNormal(d["V"], K, D, WIN, rng=rng)
    assert n_params(d["V"], K, D, WIN) == 724401, "param mismatch"

    ck = os.path.join(RUNDIR, f"normal_seed{SEED}.npz")
    ep0 = 0
    parts = []
    for f in os.listdir(RUNDIR):
        if f.startswith(f"normal_seed{SEED}_ep") and f.endswith(".npz"):
            try:
                parts.append((int(f.split("_ep")[1][:-4]), f))
            except ValueError:
                continue
    if not parts and os.path.exists(ck):
        z = np.load(ck)
        for p in net.parameters():
            if p.name in z:
                p.data[:] = z[p.name]
        print(f"  loaded finished checkpoint {os.path.basename(ck)}",
              flush=True)
        v = acc_of(net.forward(d["Xva"]), d["yva"]) * 100
        print(f"VAL {v:.2f}% (already trained)", flush=True)
        return
    if parts:
        ep0, newest = max(parts)
        z = np.load(os.path.join(RUNDIR, newest))
        for p in net.parameters():
            if p.name in z:
                p.data[:] = z[p.name]
        print(f"  resumed from epoch {ep0}", flush=True)

    opt = SGD(net.parameters(), lr=LR)
    rngd = np.random.default_rng(SEED * 131 + 7)
    X, y = d["Xtr"], d["ytr"]
    best_v = -1.0
    for ep in range(ep0, EPOCHS):
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            net.loss(X[idx], y[idx]).backward()
            opt.step()
        if (ep + 1) % SAVE_EVERY == 0:
            np.savez(os.path.join(RUNDIR, f"normal_seed{SEED}_ep{ep + 1}.npz"),
                     **{p.name: p.data for p in net.parameters()})
        v = acc_of(net.forward(d["Xva"]), d["yva"])
        best_v = max(best_v, v)
        print(f"  ep{ep + 1:4d}/{EPOCHS} val {v*100:.2f}%", flush=True)
    np.savez(ck, **{p.name: p.data for p in net.parameters()})
    ta = acc_of(net.forward(d["Xtr"][:16384]), d["ytr"][:16384]) * 100
    print(f"[normal-concat] VAL {best_v*100:.2f}% | train(16k) {ta:.2f}% | "
          f"params {n_p}", flush=True)
    json.dump({"task": "tiny_shakespeare", "mode": "matched-normal-concat",
               "params": n_p, "readable_params": 724401, "epochs": EPOCHS,
               "lr": LR, "batch": BATCH, "seed": SEED,
               "val": best_v * 100, "train_acc": ta,
               "note": "matched control for the concat readable model; "
                       "VALIDATION only, test never read"},
              open(os.path.join(RUNDIR, "results.json"), "w"), indent=1)
    print("saved runs/t2/normal_concat/results.json", flush=True)


if __name__ == "__main__":
    main()