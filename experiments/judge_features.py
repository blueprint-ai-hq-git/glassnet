"""PHASE 4 experiment — pixel features vs concept features for the judge.

Question: the pixel judge memorised per-corruption signatures and failed
the unseen-type test (54.7%). Do GlassNet's own 14 named concepts
generalise better, because every corruption shoves images OUT of the
normal concept band?
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.tensor import Tensor, sigmoid, relu  # noqa: E402
from glassnet.layers import Dense  # noqa: E402
from glassnet.losses import bce_logits  # noqa: E402
from glassnet.optim import SGD  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402
from glassnet.concepts import compute  # noqa: E402
from glassnet.judge import TYPE_NAMES, build_pile, Cleaner  # noqa: E402


class J2:
    """Tiny MLP with configurable input width (same engine as the judge)."""

    def __init__(self, in_w, h=32, seed=0):
        rng = np.random.default_rng(seed)
        self.d1 = Dense(in_w, h, rng=rng, name="j1")
        self.d2 = Dense(h, 1, rng=rng, name="j2")

    def forward(self, x):
        return self.d2(relu(self.d1(x)))

    def loss(self, x, y):
        return bce_logits(self.forward(x), y)

    def parameters(self):
        return self.d1.parameters() + self.d2.parameters()

    def prob(self, X, b=512):
        return np.concatenate([sigmoid(self.forward(Tensor(X[i:i + b]))).data
                               for i in range(0, len(X), b)]).ravel()

    def fit(self, X, y, epochs=8, batch=64, lr=0.1):
        opt = SGD(self.parameters(), lr=lr)
        rngd = np.random.default_rng(1234)
        n = len(X)
        y2 = y.astype(np.float32).reshape(-1, 1)
        for ep in range(epochs):
            order = rngd.permutation(n)
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                opt.zero_grad()
                loss = self.loss(Tensor(X[idx]), y2[idx])
                loss.backward()
                opt.step()


def main():
    Xtr, ytr, Xte, yte = load_pixels("data")
    rng = np.random.default_rng(42)
    pile = build_pile(Xtr, ytr, rng)
    _, Xc, yc, tc, _ = Cleaner().clean(pile["X"], pile["y"], pile["types"])
    z = np.load("data/concepts.npz")
    Cx, _, _ = compute(Xc, z["lo"], z["hi"])
    pix_stats = np.stack([Xc.mean(1), Xc.std(1)], axis=1)
    feats = {"PIXELS raw": Xc,
             "CONCEPTS only": Cx,
             "CONCEPTS + stats": np.concatenate([Cx, pix_stats], axis=1),
             "CONCEPTS + mean/std + ink-bbox": np.concatenate(
                 [Cx, pix_stats, Xc.mean(1).reshape(-1, 1)], axis=1)}

    good_i = np.where(yc == 0)[0]
    bad_i = np.where(yc == 1)[0]
    r2 = np.random.default_rng(7)
    gtr = r2.choice(good_i, int(0.6 * len(good_i)), replace=False)
    btr = r2.choice(bad_i, int(0.6 * len(bad_i)), replace=False)
    tri = np.concatenate([gtr, btr])
    tei = np.setdiff1d(np.arange(len(Xc)), tri)
    seen_t = TYPE_NAMES[:4]
    tri2 = np.where(np.isin(tc, ["good"] + seen_t))[0]
    mu = np.isin(tc, TYPE_NAMES[4:])
    goodm = tc == "good"

    print(f"{'features':22s} {'held-out':>9s} {'unseen-type':>12s}")
    for label, F in feats.items():
        # random split (the letter of the gate)
        m = J2(F.shape[1])
        m.fit(F[tri], yc[tri])
        a1 = float(((m.prob(F[tei]) > 0.5) == yc[tei]).mean())
        # unseen-type split: model sees ONLY 4 of the 8 corruption types
        m2 = J2(F.shape[1], seed=1)
        m2.fit(F[tri2], yc[tri2])
        Xu = np.concatenate([F[mu], F[goodm]])
        yu = np.concatenate([yc[mu], yc[goodm]])
        a2 = float(((m2.prob(Xu) > 0.5) == yu).mean())
        print(f"{label:22s} {a1*100:8.2f}% {a2*100:11.2f}%")


if __name__ == "__main__":
    main()