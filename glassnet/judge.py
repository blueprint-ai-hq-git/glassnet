"""glassnet/judge.py — Phase 4: the data judge.

1. A small hand-labeled pile: real Fashion-MNIST images (GOOD) + systematic
   corruptions (BAD) + injected cheap junk (duplicates, empty, flat, NaN).
2. A cheap automatic cleaner (stdlib only) that removes duplicates, empty
   items and obvious junk BEFORE the judge sees anything — with
   injected-vs-found counts (nothing hidden).
3. A tiny from-scratch judge model (our engine: 784 -> 32 -> 1, sigmoid,
   BCE) trained on the cleaned pile.

All numbers living inside the defaults are documented in HIDDEN_DEFAULTS.md.
"""
import hashlib

import numpy as np

from .tensor import Tensor, sigmoid, relu
from .layers import Dense
from .losses import bce_logits
from .optim import SGD

JUDGE_DEFAULTS = {
    # ---- the hand-labeled pile ----
    "good_count": 1200,          # good rows (balanced over classes)
    "bad_per_type": 150,         # bad rows per corruption type
    "corruption_count": 8,       # corruption types
    "injected_dupes": 40,        # exact copies of good rows
    "injected_empty": 20,        # all-blank rows
    "injected_gray": 20,         # constant mid-gray rows
    "injected_nan": 20,          # rows containing NaN pixels
    # ---- the cheap cleaner ----
    "empty_mean_max": 0.01,      # mean pixel below this = empty/blank
    "flat_std_max": 0.005,       # pixel std below this = constant/obvious junk
    "dup_l1_max": 0.02,          # max mean abs diff = near-duplicate
    "dup_bucket_size": 14,       # downsample side for near-dup hashing
    "dup_levels": 4,             # quantization levels for near-dup hashing
    # ---- the judge model + training ----
    "hidden": 32,
    "epochs": 10,                # was 6 (v1 gate): 8 epochs -> 72% unseen-type vs 54.7% at 6
    "batch_size": 64,
    "learning_rate": 0.1,
    "train_fraction": 0.6,       # of the cleaned pile (rest held out)
    "ensemble_seeds": [0, 1, 2],  # inference = mean probability of 3 seeds
}

TYPE_NAMES = ["noise_full", "noise_partial", "scrambled", "rotated90",
              "upside_down", "half_bled", "box_overlay", "salt_pepper"]


# ------------------------------------------------------------------ pile
def _corrupt(img, kind, rng):
    img = img.reshape(28, 28).copy()
    if kind == "noise_full":
        return np.clip(rng.normal(0.35, 0.28, img.shape), 0, 1)
    if kind == "noise_partial":
        h0 = int(rng.integers(0, 14)); w0 = int(rng.integers(0, 14))
        h = int(rng.integers(7, 13)); w = int(rng.integers(7, 13))
        img[h0:min(h0 + h, 28), w0:min(w0 + w, 28)] = \
            np.clip(rng.normal(0.35, 0.28, (min(h0 + h, 28) - h0,
                                            min(w0 + w, 28) - w0)), 0, 1)
        return img
    if kind == "scrambled":
        blocks = img.reshape(4, 7, 4, 7).transpose(0, 2, 1, 3).reshape(16, 49)
        blocks = blocks[rng.permutation(16)]
        return blocks.reshape(28, 28)
    if kind == "rotated90":
        return np.rot90(img)
    if kind == "upside_down":
        return img[::-1]
    if kind == "half_bled":
        img[:14, :] = 0.0
        return img
    if kind == "box_overlay":
        h0 = int(rng.integers(2, 17)); w0 = int(rng.integers(2, 17))
        h = int(rng.integers(4, 10));  w = int(rng.integers(4, 10))
        img[h0:min(h0 + h, 28), w0:min(w0 + w, 28)] = 1.0
        return img
    if kind == "salt_pepper":
        mask = rng.random(img.shape) < 0.20
        vals = np.where(rng.random(mask.sum()) < 0.5, 1.0, 0.0)
        img[mask] = vals
        return img
    raise ValueError(kind)


def _class_indexes(ytr):
    return [np.where(ytr == c)[0] for c in range(10)]


def build_pile(Xtr, ytr, rng, d=None):
    """The real pile builder (see sketch notes in this file's docstring)."""
    d = dict(JUDGE_DEFAULTS if d is None else d)
    cls_idx = _class_indexes(ytr)
    good = []
    per_class = d["good_count"] // 10
    for c in range(10):
        picks = rng.choice(cls_idx[c], per_class, replace=False)
        good.append(Xtr[picks])
    X_good = np.concatenate(good)                       # (good_count, 784)
    kinds = TYPE_NAMES[:d["corruption_count"]]
    X_bad, bad_types = [], []
    for kind in kinds:
        picks = rng.choice(len(X_good), d["bad_per_type"], replace=False)
        X_bad.append(np.stack([_corrupt(X_good[p].reshape(28, 28), kind, rng)
                               .reshape(-1) for p in picks]))
        bad_types += [kind] * d["bad_per_type"]

    junk = [
        ("dupe", X_good[rng.choice(len(X_good), d["injected_dupes"], replace=False)].copy()),
        ("empty", np.zeros((d["injected_empty"], 784), np.float32)),
        ("gray", np.full((d["injected_gray"], 784), 0.5, np.float32)),
        ("nan", np.full((d["injected_nan"], 784), np.nan, np.float32)),
    ]
    X_junk, junk_types = [], []
    for tag, arr in junk:
        X_junk.append(arr); junk_types += [tag] * arr.shape[0]

    X = np.concatenate([X_good] + X_bad + X_junk).astype(np.float32)
    y = np.concatenate([np.zeros(len(X_good), np.int64)] +
                       [np.ones(sum(len(a) for a in X_bad), np.int64)] +
                       [np.ones(sum(a.shape[0] for _, a in junk), np.int64)])
    types = np.array(["good"] * len(X_good) + bad_types + junk_types)
    return {"X": X, "y": y, "types": types,
            "injected": {"dupe": d["injected_dupes"], "empty": d["injected_empty"],
                         "gray": d["injected_gray"], "nan": d["injected_nan"]}}


# ------------------------------------------------------------------ cleaner
class Cleaner:
    """Cheap automatic cleaning: NaN -> empty/blank -> flat/gray -> duplicates.
    Reports what it removed, category by category (vs what was injected)."""

    def __init__(self, d=None):
        self.d = dict(JUDGE_DEFAULTS if d is None else d)

    @staticmethod
    def _hash(a):
        return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()

    def clean(self, X, y, types):
        keep = np.arange(len(X))
        report = {}
        # NaN / inf
        fin = np.isfinite(X[keep]).all(axis=1)
        report["non-finite"] = int((~fin).sum())
        keep = keep[fin]
        # empty / blank (near-zero mean)
        means = X[keep].mean(axis=1)
        emp = means < self.d["empty_mean_max"]
        report["empty/blank"] = int(emp.sum())
        keep = keep[~emp]
        # flat / obvious single-value junk
        stds = X[keep].std(axis=1)
        fl = stds < self.d["flat_std_max"]
        report["flat(dupe-like)"] = int(fl.sum())
        keep = keep[~fl]
        # duplicates: exact first (md5), then near (bucket hash + L1)
        seen_exact = {}
        mask_exact = np.zeros(len(keep), bool)
        for i, gi in enumerate(keep):
            h = self._hash(X[gi].view(np.uint8))
            if h in seen_exact:
                mask_exact[i] = True
            else:
                seen_exact[h] = i
        report["duplicates(exact)"] = int(mask_exact.sum())
        keep_ex = keep[~mask_exact]
        # near-duplicates: bucket = downsampled quantized fingerprint
        bs, lv = self.d["dup_bucket_size"], self.d["dup_levels"]
        fp = []
        for i in range(0, len(keep_ex), 512):  # chunk for memory
            mini = X[keep_ex[i:i + 512]].reshape(-1, 28, 28)
            small = mini[:, ::28 // bs, ::28 // bs]
            q = (small * (lv - 1)).astype(np.uint8).reshape(len(mini), -1)
            fp.append(q)
        fp = np.concatenate(fp)
        buckets = {}
        mask_near = np.zeros(len(keep_ex), bool)
        dmax = self.d["dup_l1_max"]
        for i in range(len(keep_ex)):
            key = self._hash(fp[i])
            prev = buckets.get(key)
            if prev is not None:
                if np.abs(X[keep_ex[i]] - X[keep_ex[prev]]).mean() < dmax:
                    mask_near[i] = True
                    continue
            buckets[key] = i
        report["duplicates(near)"] = int(mask_near.sum())
        final = np.concatenate([keep_ex[~mask_near]])
        report["total_removed"] = int(len(X) - len(final))
        return final, X[final], y[final], types[final], report


# ------------------------------------------------------------------ judge
class JudgeModel:
    """784 -> hidden -> 1 (sigmoid) — P(bad). Trained with our own SGD + BCE."""

    def __init__(self, rng, hidden=None):
        self.h = hidden or JUDGE_DEFAULTS["hidden"]
        self.d1 = Dense(784, self.h, rng=rng, name="judge1")
        self.d2 = Dense(self.h, 1, rng=rng, name="judge2")

    def forward(self, x):
        return self.d2(relu(self.d1(x)))

    def loss(self, x, y):
        return bce_logits(self.forward(x), y)

    def parameters(self):
        return self.d1.parameters() + self.d2.parameters()

    def prob_bad(self, X, batch=512):
        out = []
        for i in range(0, len(X), batch):
            s = sigmoid(self.forward(Tensor(X[i:i + batch]))).data
            out.append(s)
        return np.concatenate(out).ravel()

    def predict(self, X, batch=512):
        return self.prob_bad(X, batch) > 0.5

    def train(self, Xtr, ytr, epochs=None, batch_size=None, lr=None):
        d = JUDGE_DEFAULTS
        epochs = epochs or d["epochs"]
        batch_size = batch_size or d["batch_size"]
        lr = lr or d["learning_rate"]
        opt = SGD(self.parameters(), lr=lr)
        rng = np.random.default_rng(1234)
        n = len(Xtr)
        y2 = ytr.astype(np.float32).reshape(-1, 1)
        for epoch in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = order[i:i + batch_size]
                opt.zero_grad()
                loss = self.loss(Tensor(Xtr[idx]), y2[idx])
                loss.backward()
                opt.step()

    def save(self, path, **extra):
        out = {p.name: p.data for layer in (self.d1, self.d2)
               for p in layer.parameters()}
        out.update(extra)
        np.savez(path, **out)

    @staticmethod
    def load(path):
        z = np.load(path)
        h = z["judge1.W"].shape[1]
        m = JudgeModel(np.random.default_rng(0), hidden=h)
        for layer in (m.d1, m.d2):
            for p in layer.parameters():
                p.data[:] = z[p.name].astype(np.float32)
        return m