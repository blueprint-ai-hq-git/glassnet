"""loop_lib.py — the autonomous-loop toolbox.

Discipline (mission rules):
- ALL tuning decisions happen on the VALIDATION split.
- The TEST set is quarantined; every use is logged in TEST-USE.md.
- Normal comparisons use a same-size normal net; both param counts reported.
- Key results: 3 seeds; a win smaller than the seed noise = "no difference".
- Every new concept set must pass the fidelity audit + lie-neuron control
  (loop_audit.py) before it can be promoted.
- Old results are never deleted: each iteration writes to its own folder.
"""
import hashlib
import json
import os

import numpy as np

from .tensor import Tensor
from .layers import Dense, Sequential
from .losses import cross_entropy
from .optim import SGD
from .eval import accuracy as acc_fn
from .autopilot import train_autopilot, AUTO_PILOT_DEFAULTS
from .phase2_model import GlassNetPhase2
from . import concepts as C

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs", "loop")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(RUNS, exist_ok=True)
SPLIT_SEED = 1234


def log_line(text):
    with open(os.path.join(RUNS, "console.log"), "a") as f:
        f.write(text + "\n")
    print(text, flush=True)


# ---------------------------------------------------------------- data split
def load_pathmnist(force=False):
    """PathMNIST (MedMNIST v2, CC BY 4.0 — verified 2026-10-01 via the
    official Zenodo record API). Official splits kept: train 89,996 /
    val 10,000 / test 7,180. RGB -> greyscale (channel mean), /255, 784.
    Quarantine: the test split is only used via milestone_check."""
    dpath = os.path.join(DATA_DIR, "medmnist", "pathmnist.npz")
    cache = os.path.join(DATA_DIR, "loop_pathmnist_gray.npz")
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        return (z["Xtr"], z["ytr"], z["Xval"], z["yval"], z["Xte"], z["yte"])
    EXPECTED_SIZE = 205615438
    EXPECTED_MD5 = "a8b06965200029087d5bd730944a56c1"
    if not os.path.exists(dpath) or os.path.getsize(dpath) != EXPECTED_SIZE:
        if os.path.exists(dpath):
            log_line("[medmnist] corrupt/partial download removed")
            os.remove(dpath)
        os.makedirs(os.path.dirname(dpath), exist_ok=True)
        # canonical file URL resolved from the record API (no guessing)
        import urllib.request, json as _json
        with urllib.request.urlopen(
                "https://zenodo.org/api/records/6496656", timeout=60) as r:
            rec = _json.load(r)
        url = next(f["links"]["self"] for f in rec["files"]
                   if f["key"] == "pathmnist.npz")
        log_line("[medmnist] downloading pathmnist.npz "
                 f"({EXPECTED_SIZE/1e6:.0f} MB, CC BY 4.0) ...")
        for attempt in range(1, 4):
            if os.path.exists(dpath):
                os.remove(dpath)
            try:
                urllib.request.urlretrieve(url, dpath)
            except Exception as e:
                log_line(f"[medmnist] attempt {attempt} failed: {e}")
                continue
            got = os.path.getsize(dpath)
            if got == EXPECTED_SIZE:
                break
            log_line(f"[medmnist] attempt {attempt} truncated "
                     f"({got} bytes) — retrying")
        got = os.path.getsize(dpath)
        if got != EXPECTED_SIZE:
            os.remove(dpath)
            raise RuntimeError(f"pathmnist download truncated: "
                               f"{got} != {EXPECTED_SIZE} bytes")
        import hashlib
        h = hashlib.md5()
        with open(dpath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != EXPECTED_MD5:
            os.remove(dpath)
            raise RuntimeError(f"pathmnist md5 mismatch: "
                               f"{h.hexdigest()} != {EXPECTED_MD5}")
        log_line("[medmnist] download verified (size + md5)")
    z = np.load(dpath)
    Xtr, ytr = z["train_images"], z["train_labels"].ravel()
    Xval, yval = z["val_images"], z["val_labels"].ravel()
    Xte, yte = z["test_images"], z["test_labels"].ravel()
    if Xtr.shape != (89996, 28, 28, 3):
        raise ValueError(f"pathmnist train shape {Xtr.shape} unexpected")
    if set(np.unique(ytr)) != set(range(9)):
        raise ValueError("pathmnist labels outside 0..8 — corrupt download")

    def gray(a):
        return a.mean(axis=3).reshape(len(a), -1).astype(np.float32) / 255.0

    G = (gray(Xtr), ytr.astype(np.int64), gray(Xval), yval.astype(np.int64),
         gray(Xte), yte.astype(np.int64))
    np.savez(cache, Xtr=G[0], ytr=G[1], Xval=G[2], yval=G[3],
             Xte=G[4], yte=G[5])
    log_line(f"[medmnist] cached -> {cache} "
             f"(train {G[0].shape}, val {G[2].shape}, test {G[4].shape})")
    return G


def load_split(task="fashion-mnist", force=False):
    """train/val/test under loop discipline. fashion-mnist: 50k/10k/10k
    (stratified from the 60k train; official 10k quarantined).
    pathmnist: the dataset's own official splits (test quarantined).
    banking77: stratified 80/10/10 (vocab + scaling from TRAIN only)."""
    if task == "pathmnist":
        return load_pathmnist(force)
    if task == "banking77":
        from . import text_data
        (Xtr, ytr, Xval, yval, Xte, yte,
         t_tr, t_va, t_te, uniq) = text_data.load(force)
        return Xtr, ytr, Xval, yval, Xte, yte
    path = os.path.join(DATA_DIR, "loop_split.npz")
    if os.path.exists(path) and not force:
        z = np.load(path)
        return (z["Xtr"], z["ytr"], z["Xval"], z["yval"], z["Xte"], z["yte"])
    from .data import load as load_pixels
    X, y, Xte, yte = load_pixels(DATA_DIR)
    rng = np.random.default_rng(SPLIT_SEED)
    tr, va = [], []
    for c in range(10):
        ci = np.where(y == c)[0]
        p = rng.permutation(ci)
        va.append(p[:1000])
        tr.append(p[1000:])
    tr = np.concatenate(tr)
    va = np.concatenate(va)
    np.savez(path, Xtr=X[tr], ytr=y[tr], Xval=X[va], yval=y[va],
             Xte=Xte, yte=yte)
    log_line(f"[split] wrote data/loop_split.npz (train {len(tr)} / "
             f"val {len(va)} / test {len(Xte)} quarantined)")
    return X[tr], y[tr], X[va], y[va], Xte, yte


# ------------------------------------------------------------- concept sets
def set_id(names):
    return hashlib.sha1("|".join(names).encode()).hexdigest()[:10]


def concept_cache_path(names, task="fashion-mnist"):
    return os.path.join(DATA_DIR, f"loop_concepts_{task}_{set_id(names)}.npz")


def load_concept_set(Xtr, ytr, Xval, yval, Xte, yte, names, task="fashion-mnist",
                     force=False):
    """Concept targets for a named set, normalised on TRAIN only.
    Returns Ctr, Cval, Cte, lo, hi. Caches per set+task (never overwrites
    the Phase-2 archive data/concepts.npz). For text tasks (banking77) the
    concepts come from the word/lexicon module over the raw texts."""
    names = list(names)
    path = concept_cache_path(names, task)
    if os.path.exists(path) and not force:
        z = np.load(path)
        return z["Ctr"], z["Cval"], z["Cte"], z["lo"], z["hi"]
    if task == "banking77":
        from . import text_concepts, text_data
        (_, _, _, _, _, _,
         t_tr, t_va, t_te, _) = text_data.load(force)
        Ctr, lo, hi = text_concepts.compute_selected(t_tr, names)
        Cval, _, _ = text_concepts.compute_selected(t_va, names, lo, hi)
        Cte, _, _ = text_concepts.compute_selected(t_te, names, lo, hi)
    else:
        Ctr, lo, hi = C.compute_selected(Xtr, names)
        Cval, _, _ = C.compute_selected(Xval, names, lo, hi)
        Cte, _, _ = C.compute_selected(Xte, names, lo, hi)
    np.savez(path, Ctr=Ctr, Cval=Cval, Cte=Cte, lo=lo, hi=hi)
    log_line(f"[concepts] cached set {len(names)} names, task {task} "
             f"({set_id(names)}): {names[0]}...{names[-1]}")
    return Ctr, Cval, Cte, np.asarray(lo, np.float32), np.asarray(hi, np.float32)


def verify_phase2_archive_parity(X):
    """The refactored feature functions must reproduce the archived Phase-2
    concept cache exactly (same math, same order)."""
    z = np.load(os.path.join(DATA_DIR, "concepts.npz"))
    Ctr_old = z["Ctr"]
    from .data import load as load_pixels
    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR)
    Ctr_new, _, _ = C.compute(Xtr)
    diff = float(np.abs(Ctr_new - Ctr_old).max())
    if diff > 1e-7:
        raise RuntimeError(f"concept archive parity broken: max diff {diff}")
    log_line(f"[parity] concept archive reproduced exactly (max diff {diff:.2e})")
    return diff


# ---------------------------------------------------------------- normal net
def normal_params(hidden, classes=10, d_in=784):
    return d_in * hidden + hidden * classes + hidden + classes


def match_hidden(target_params, classes=10, d_in=784):
    best, best_h = abs(normal_params(4, classes, d_in) - target_params), 4
    for h in range(4, 301):
        d = abs(normal_params(h, classes, d_in) - target_params)
        if d < best:
            best, best_h = d, h
    return best_h


def train_normal(Xtr, ytr, Xval, yval, seed, hidden=15, epochs=12,
                 batch=128, lr=0.1, classes=10, d_in=784):
    """Standard MLP d_in->hidden->classes (our engine), same schedule family
    as the auto-pilot: 12 epochs, batch 128, lr 0.1, best-validated snapshot.
    Returns (net, best_val_acc)."""
    rng = np.random.default_rng(seed)
    net = Sequential(Dense(d_in, hidden, rng=rng, name="n1"),
                     _relu(), Dense(hidden, classes, rng=rng, name="n2"))
    opt = SGD(net.parameters(), lr=lr)
    rngd = np.random.default_rng(seed * 777 + 1)
    n = len(Xtr)
    best_val, best_w = 0.0, None
    for epoch in range(epochs):
        order = rngd.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            out = net(Tensor(Xtr[idx]))
            loss = cross_entropy(out, ytr[idx])
            loss.backward()
            opt.step()
        v = acc_fn(net, Xval, yval)
        if v > best_val:
            best_val = v
            best_w = {p.name: p.data.copy() for layer in net.layers
                      if hasattr(layer, "parameters")
                      for p in layer.parameters()}
    if best_w is not None:
        for layer in net.layers:
            if hasattr(layer, "parameters"):
                for p in layer.parameters():
                    p.data[:] = best_w[p.name]
    return net, best_val


def _relu():
    from .tensor import relu
    return relu


def normal_benchmark(Xtr, ytr, Xval, yval, hidden, seeds=(0, 1, 2),
                     classes=10, d_in=784, epochs=12):
    """3-seed same-size normal net. Returns (vals, nets[seed])."""
    vals, nets = {}, {}
    for s in seeds:
        net, v = train_normal(Xtr, ytr, Xval, yval, seed=s, hidden=hidden,
                              classes=classes, d_in=d_in, epochs=epochs)
        vals[s] = v
        nets[s] = net
    return vals, nets


# ------------------------------------------------------------- readable run
def readable_benchmark(Xtr, ytr, Ctr, Xval, yval, Cval, names, R=30,
                       seeds=(0, 1, 2), concept_hidden=0, rule_linear=False,
                       alpha=1.0, no_rules=False, classes=10, d_in=784,
                       epochs=12):
    """3-seed auto-pilot run of the readable arch on the chosen concept set.
    The autopilot's eval set IS the validation split (decisions stay on val).
    Returns (vals, infos, nets) — vals[s] = post-prune VAL accuracy."""
    d = dict(AUTO_PILOT_DEFAULTS)
    d["num_concepts"] = len(names)
    d["num_rules"] = int(R)
    d["concept_hidden"] = int(concept_hidden)
    d["rule_linear"] = bool(rule_linear)
    d["concept_loss_alpha"] = float(alpha)
    d["no_rules"] = bool(no_rules)
    d["num_classes"] = int(classes)
    d["input_dim"] = int(d_in)
    d["max_epochs"] = int(epochs)
    vals, infos, nets = {}, {}, {}
    for s in seeds:
        net, info = train_autopilot(Xtr, ytr, Ctr, Xval, yval, Cval,
                                    seed=s, log_actions=False, defaults=d)
        vals[s] = info["post_prune_acc"]
        infos[s] = info
        nets[s] = net
    return vals, infos, nets


def readable_params(K, R, concept_hidden=0, no_rules=False, classes=10,
                    d_in=784):
    if no_rules:
        base = d_in * K + K + classes * K + classes
        if concept_hidden:
            base += (d_in * concept_hidden + concept_hidden
                     + concept_hidden * K - d_in * K)
        return base
    base = d_in * K + K + K * R + R + classes * R + classes
    if concept_hidden:
        base += (d_in * concept_hidden + concept_hidden
                 + concept_hidden * K - d_in * K)
    return base


def promote_check(new_vals, best_vals):
    """Verdict per mission: keep only if the mean win beats the seed noise.
    noise = (std_new + std_best) / 2."""
    mu_n, sd_n = float(np.mean(list(new_vals.values()))), float(np.std(list(new_vals.values())))
    mu_b, sd_b = float(np.mean(list(best_vals.values()))), float(np.std(list(best_vals.values())))
    noise = (sd_n + sd_b) / 2.0
    delta = mu_n - mu_b
    if delta > noise:
        return "KEEP", mu_n, sd_n, delta, noise
    if delta < -noise:
        return "REVERT", mu_n, sd_n, delta, noise
    return "NO-DIFF", mu_n, sd_n, delta, noise


def load_current_best(task="fashion-mnist"):
    path = os.path.join(RUNS, f"CURRENT-BEST_{task}.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None


def save_current_best(entry, task=None):
    task = task or entry.get("task", "fashion-mnist")
    json.dump(entry, open(os.path.join(RUNS, f"CURRENT-BEST_{task}.json"),
                          "w"), indent=1)


def log_test_use(milestone, numbers):
    path = os.path.join(RUNS, "TEST-USE.md")
    with open(path, "a") as f:
        f.write(f"- {milestone}: {numbers}\n")
    log_line(f"[test-use] {milestone}: {numbers}")