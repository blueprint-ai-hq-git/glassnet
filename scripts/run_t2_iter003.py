"""RUNG T2 iter 003 — input representation: bag-of-embeddings -> CONCAT window.

Pre-registered (written BEFORE the run):
WHY: t2-001/002 (bag = SUM of window embeddings) froze at 16.38 VAL at both
48 and 96 epochs (identical numbers, best-epoch ~40 of 96): doubling the
budget changed nothing. A SUM cannot represent word order ("the dog" vs
"dog the" -> same bag) and the model sits just above the 9.7% <unk> prior.
A eposition+order representation — the concatenated window (D*WIN input,
W=8, D=64 -> 512-d) — is the minimal input change that keeps the entire
rung-3 readable machinery (concepts, sparsity, classes-read-concepts,
audit, lie probe). Readability is kept causal: a concept's row splits into
8 position-blocks; the logit contribution of word w at position p is
row_block_p @ emb_w, so naming and audit stay the same causal meters.
SUCCESS TEST (unchanged formulas): A) val acc >= 19.44 (2x unk prior);
B) >= 50% of 24 names pass the standing audit (resid AUROC >= 0.55 AND
logit contrib >= 0.25); C) lie probe (conditional on A+B, as in 002).
Scope: readable-only (normal comparison deferred to the gate stage —
its written venue; 6 nets x 96ep would exceed the 2h cap). 96 epochs,
6 BLAS threads, ~100 min for 3 seeds.
Prediction (before run): concat unlocks the mapping the bag could not
express — TRAIN acc jumps (the key diagnostic: if train acc stays ~20%,
the wall is expressivity, not generalisation), VAL acc rises toward
20-30%; audit improves only partially (naming still old-style in this
run — the cosine/support naming fix is deferred to 004 if B blocks).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from glassnet.tensor import (Tensor, sigmoid, abs_, sum_all, sum_axis,
                             embedding, relu, take_cols)
from glassnet.layers import Dense
from glassnet.losses import cross_entropy, bce_logits
from glassnet.optim import SGD
from glassnet import tiny_data

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runs", "t2")
ITERDIR = os.path.join(RUNS, os.environ.get("T2_ITER", "iter003"))

K, D, WIN = 24, 64, 8
EPOCHS = int(os.environ.get("T2_EPOCHS", "96"))
LR, BATCH = 0.1, 128
LAM_A = float(os.environ.get("T2_LAM_A", "0.2"))
LAM_W = float(os.environ.get("T2_LAM_W", "1.0"))
# --- auxiliary supervised concept loss (owner-approved switch) ---------------
# DEFAULT 0.0 -> the term is never built and the model is bit-for-bit the old
# one. Targets are frozen in CONCEPT-TARGETS.md and derived from TRAIN only.
LAM_CONCEPT = float(os.environ.get("T2_LAM_CONCEPT", "0.0"))
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_concept_targets():
    """Concept targets for the auxiliary loss.

    T2_CONCEPT_SET=words    (default) single-word targets, v1 (CONCEPT-TARGETS.md)
    T2_CONCEPT_SET=clusters           word-cluster targets, v2 (CONCEPT-TARGETS-02.md)

    Both are returned in the same shape: one id-LIST per supervised column, so a
    single word is just a list of length 1 and the loss code is identical.
    """
    if LAM_CONCEPT <= 0:
        return None
    kind = os.environ.get("T2_CONCEPT_SET", "words")
    if kind == "meaningful_ext":
        # TARGET-E.md: the extended set (9 concepts at the UNCHANGED 2.5% floor). The file's
        # qualifying list is a superset of Target C's six, so this is a strict addition.
        path = os.path.join(_PROJ, "runs", "t3", "meaningful_targets_ext.json")
        if not os.path.exists(path):
            raise RuntimeError(f"T2_CONCEPT_SET=meaningful_ext but no file at {path}")
        j = json.load(open(path))
        allt = j["qualifying"]
        n_taught = int(os.environ.get("T2_N_TAUGHT", str(len(allt))))
        tg = allt[:n_taught]
        pw = []
        for t in tg:
            s = max(float(t["support"]), 1e-6)
            pw.append((1.0 - s) / s)
        return {"kind": kind, "n_taught": n_taught, "n_targets": len(allt),
                "id_lists": [np.array(t["ids"], dtype=np.int64) for t in tg],
                "pos_weight": np.array(pw, dtype=np.float32),
                "words": [t["name"] for t in tg],
                "cols": np.arange(len(tg), dtype=np.int64)}
    if kind == "meaningful":
        # TARGET-C.md: real ideas (pronouns, dialogue lines, negation, archaic address,
        # wh-words) instead of single-word clusters. Derived from TRAIN only, in
        # runs/t3/meaningful_targets.json. First T2_N_TAUGHT columns are taught.
        path = os.path.join(_PROJ, "runs", "t3", "meaningful_targets.json")
        if not os.path.exists(path):
            raise RuntimeError(f"T2_CONCEPT_SET=meaningful but no file at {path}")
        j = json.load(open(path))
        allt = j["targets"]
        n_taught = int(os.environ.get("T2_N_TAUGHT", str(len(allt))))
        tg = allt[:n_taught]
        pw = []
        for t in tg:
            s = max(float(t["support"]), 1e-6)
            pw.append((1.0 - s) / s)          # balanced BCE weight, from TRAIN support
        return {"kind": kind, "n_taught": n_taught, "n_targets": len(allt),
                "id_lists": [np.array(t["ids"], dtype=np.int64) for t in tg],
                "pos_weight": np.array(pw, dtype=np.float32),
                "words": [t["name"] for t in tg],
                "cols": np.arange(len(tg), dtype=np.int64)}
    if kind in ("clusters", "clusters16", "clusters32"):
        # clusters32 = the TARGET-A2.md k=32 set. clusters16 = the TARGET-A3.md k=16 set.
        # Only the first T2_N_TAUGHT columns are taught; the rest are pre-registered but
        # receive NO supervision gradient, which is what makes the hold-out meaningful.
        fname = {"clusters32": "concept_clusters32.json",
                 "clusters16": "concept_clusters16.json"}.get(kind,
                                                              "concept_clusters.json")
        default_taught = {"clusters32": "24", "clusters16": "10"}.get(kind, "16")
        n_taught = int(os.environ.get("T2_N_TAUGHT", default_taught))
        path = os.path.join(_PROJ, "runs", "t3", fname)
        if not os.path.exists(path):
            raise RuntimeError(f"T2_CONCEPT_SET={kind} but no file at {path}")
        j = json.load(open(path))
        cl = j["clusters"][:n_taught]
        return {"kind": kind, "n_taught": n_taught, "n_targets": len(j["clusters"]),
                "id_lists": [np.array(c["ids"], dtype=np.int64) for c in cl],
                "pos_weight": np.array([c["pos_weight"] for c in cl],
                                       dtype=np.float32),
                "words": ["_".join(c["words"][:6]) for c in cl],
                "cols": np.arange(len(cl), dtype=np.int64)}
    path = os.path.join(_PROJ, "runs", "t3", "concept_targets.json")
    if not os.path.exists(path):
        raise RuntimeError(f"T2_LAM_CONCEPT={LAM_CONCEPT} but no targets at {path}")
    j = json.load(open(path))
    return {"kind": "words",
            "id_lists": [np.array([t["id"]], dtype=np.int64) for t in j["targets"]],
            "pos_weight": np.array([t["pos_weight"] for t in j["targets"]],
                                   dtype=np.float32),
            "words": [t["word"] for t in j["targets"]],
            "cols": np.arange(len(j["targets"]), dtype=np.int64)}


CONCEPT_TARGETS = _load_concept_targets()
SEEDS = tuple(int(s) for s in
              os.environ.get("T2_SEEDS", "0,1,2").split(","))

STOP = {"i", "my", "me", "we", "you", "your", "the", "a", "an", "to", "up",
        "can", "is", "are", "am", "was", "were", "for", "why", "how", "what",
        "where", "when", "which", "who", "do", "does", "did", "it", "on",
        "in", "at", "and", "or", "of", "with", "about", "want", "need",
        "get", "got", "please", "would", "could", "should", "this", "that",
        "have", "has", "had", "be", "been", "so", "but", "not", "no", "yes",
        "ok", "thanks", "as", "if", "by", "from", "will", "shall", "may",
        "must", "now", "then", "yet", "all", "thy", "thou", "thee", "your",
        "his", "her", "our", "their", "sir", "lord", "duke", "duchess",
        "king", "queen", "s", "t", "d", "ll", "re", "ve", "o", "a"}


def _mean(x):
    return sum_all(x) * (1.0 / x.data.size)


class TinyLM3:
    """Readable next-word: CONCAT window (W*D) -> K concepts -> logits."""

    def __init__(self, V, K, D, W, rng=None, lam_act=LAM_A, lam_w=LAM_W,
                 lam_concept=LAM_CONCEPT):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.V, self.K, self.D, self.W = V, K, D, W
        self.lam_act, self.lam_w = lam_act, lam_w
        self.lam_concept = lam_concept
        s = D ** -0.5
        self.emb = Tensor(rng.standard_normal((V, D)).astype(np.float32) * s,
                          requires_grad=True, name="emb")
        self.concept = Dense(D * W, K, rng=rng, name="concept")
        self.answer = Dense(K, V, rng=rng, name="answer")
        self.frozen = []

    def freeze(self, cols):
        self.frozen = list(cols)

    def forward(self, ids):
        E = embedding(self.emb, ids)               # (N,W,D)
        x = _flatten(E)                            # (N, W*D)
        z = self.concept(x)
        h = sigmoid(z)
        logits = self.answer(h)
        return {"concepts": h, "concept_pre": z, "logits": logits}

    def loss(self, ids, y):
        f = self.forward(ids)
        ce = cross_entropy(f["logits"], y)
        sa = self.lam_act * _mean(f["concepts"])
        sw = self.lam_w * _mean(abs_(self.concept.W))
        total = ce + sa + sw
        parts = {"ce": ce, "sp_act": sa, "sp_w": sw}
        # Auxiliary supervised concept loss. Only the columns in
        # CONCEPT_TARGETS["cols"] receive gradient from it; the remaining
        # columns are the unsupervised circularity control.
        if CONCEPT_TARGETS is not None and self.lam_concept > 0:
            t = CONCEPT_TARGETS
            Xb = ids if isinstance(ids, np.ndarray) else np.asarray(ids)
            # ysup[n, c] = 1 iff window n contains ANY member of target c.
            # NOTE: np.isin(Xb, ids) returns an array shaped like Xb (N,W) --
            # "which TOKENS are targets" -- which is NOT the (N,C) matrix the
            # loss needs. Broadcast the comparison instead. A single-word target
            # is just a member list of length 1, so one loop covers both sets.
            n_cols = len(t["id_lists"])
            ysup = np.zeros((Xb.shape[0], n_cols), dtype=np.float32)
            for c, idc in enumerate(t["id_lists"]):
                if idc.size:
                    ysup[:, c] = (Xb[:, :, None] == idc[None, None, :]).any(
                        axis=2).any(axis=1)
            if ysup.shape[1] != t["cols"].size:
                raise RuntimeError(f"target matrix {ysup.shape} vs "
                                   f"{t['cols'].size} supervised columns")
            aux = bce_logits(take_cols(f["concept_pre"], t["cols"]), ysup,
                             w=t["pos_weight"])
            total = total + self.lam_concept * aux
            parts["concept_aux"] = aux
        return total, parts

    def parameters(self):
        return [self.emb] + self.concept.parameters() + self.answer.parameters()

    def save(self, path):
        np.savez(path, emb=self.emb.data, cW=self.concept.W.data,
                 cb=self.concept.b.data, aW=self.answer.W.data,
                 ab=self.answer.b.data)


def _flatten(E):
    """(N,W,D) -> (N,W*D), gradient = reshape of E's grad."""
    out = Tensor(E.data.reshape(E.data.shape[0], -1),
                 requires_grad=E.requires_grad, _prev=(E,), name="flatten")

    def bw():
        if out.grad is None:
            return
        _acc_flat(E, out.grad.reshape(E.data.shape))

    out._backward = bw
    return out


def _acc_flat(E, g):
    if not E.requires_grad:
        return
    E.grad = g if E.grad is None else E.grad + g


def net_placeholder(d, s):
    """Construct the net for training (kept separate for clarity)."""
    return TinyLM3(d["V"], K, D, WIN, rng=np.random.default_rng(s))


readable_params = lambda V, D, K, W: V * D + (D * W) * K + K + K * V + V


def match_h(V, D, K, W):
    rp = readable_params(V, D, K, W)
    return (rp - V * D - V) // (D * W + 1 + V)


def _save_with(net, path, extra=None):
    """net.save + extra arrays (best-so-far weights for honest resume)."""
    np.savez(path, emb=net.emb.data, cW=net.concept.W.data,
             cb=net.concept.b.data, aW=net.answer.W.data,
             ab=net.answer.b.data, **(extra or {}))


def acc_of(logits, y):
    return float(np.mean(logits.data.argmax(1) == np.asarray(y)))


def train(net, d, seed, iterdir, tag="final", n_epochs=None,
          save_every=None):
    """Train with intra-seed partial checkpoints: saves every save_every
    epochs as {tag}_seed{seed}_ep{N}.npz and resumes from the newest partial
    if one exists (external-kill protection — a killed seed continues, it
    does not restart)."""
    n_epochs = n_epochs or EPOCHS
    save_every = n_epochs if os.environ.get("T2_SAVE_EVERY") == "end" \
        else int(os.environ.get("T2_SAVE_EVERY", max(1, n_epochs // 4)))
    rngd = np.random.default_rng(seed * 131 + 7)
    opt = SGD(net.parameters(), lr=LR)
    best_v, best_ws = -1.0, None
    X, y = d["Xtr"], d["ytr"]

    def start_epoch():
        nonlocal best_v, best_ws
        parts = []
        for f in os.listdir(iterdir):
            if f.startswith(f"{tag}_seed{seed}_ep") and f.endswith(".npz"):
                try:
                    parts.append((int(f.split("_ep")[1][:-4]), f))
                except ValueError:
                    continue
        if not parts:
            return 0
        ep0, newest = max(parts)          # numeric max, not lexicographic
        z = np.load(os.path.join(iterdir, newest))
        for p in net.parameters():
            if p.name in z:
                p.data[:] = z[p.name]
        if "best_v" in z:
            best_v = float(z["best_v"])
            best_ws = {p.name: z[f"best_{p.name}"].copy()
                       for p in net.parameters()
                       if f"best_{p.name}" in z}
        print(f"  resumed {tag} seed{seed} from epoch {ep0} "
              f"(best_v so far {best_v*100:.2f}%)", flush=True)
        return ep0

    ep0 = start_epoch()
    for ep in range(ep0, n_epochs):
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss, _ = net.loss(X[idx], y[idx])
            loss.backward()
            if net.frozen:
                for k in net.frozen:
                    net.concept.W.grad[:, k] = 0.0
                    net.concept.b.grad[k] = 0.0
            opt.step()
        if (ep + 1) % save_every == 0:
            extra = {"best_v": np.float32(best_v)}
            for p in net.parameters():
                if best_ws and p.name in best_ws:
                    extra[f"best_{p.name}"] = best_ws[p.name]
            _save_with(net, os.path.join(iterdir,
                          f"{tag}_seed{seed}_ep{ep + 1}.npz"), extra)
        v = acc_of(net.forward(d["Xva"])["logits"], d["yva"])
        print(f"  ep{ep + 1:4d}/{n_epochs} seed{seed} val {v*100:.2f}%",
              flush=True)
        if v > best_v:
            best_v, best_ws = v, {p.name: p.data.copy()
                                  for p in net.parameters()}
    for p in net.parameters():
        p.data[:] = best_ws[p.name]
    net.save(os.path.join(iterdir, f"{tag}_seed{seed}.npz"))
    tr_acc = acc_of(net.forward(X[:16384])["logits"], y[:16384])
    return net, best_v, tr_acc


def _fired_matrix(net, X, chunk=20000):
    """Boolean "does concept k fire (>0.5)" over X, computed in chunks.

    NOT a single net.forward(X), for two measured reasons:
      1. a full-set forward on 182,377 windows allocates ~2.7 GB in one shot;
      2. the autograd graph contains REFERENCE CYCLES — every op's _backward
         closure closes over its own output — so refcounting cannot free a
         forward's intermediates and the cyclic collector reclaims them only
         lazily. Measured: RSS grew ~277 MB per 20k chunk and the Meter-02
         re-audit was OOM-killed at 4.18 GB inside the worker cgroup.
    An explicit collection per chunk keeps RSS flat (measured peak 2601 MB -> 366 MB).
    The activation VALUES are identical to the single-shot forward, so no result
    changes; this is a memory-safety fix only.
    """
    import gc
    X = np.asarray(X)
    fired = None
    for i in range(0, len(X), chunk):
        h = np.asarray(net.forward(X[i:i + chunk])["concepts"].data)
        if fired is None:
            fired = np.empty((len(X), h.shape[1]), dtype=bool)
        fired[i:i + chunk] = h > 0.5
        del h
        gc.collect()
    return fired


def auto_name_t2c(net, d, Xtr, topn=8, min_support=0.01):
    """Position-aware causal naming: for concept k, word w's expected logit
    contribution = mean over fired windows of sum over positions p where
    w sits of row_block_p @ emb_w. Words must appear in >= min_support of
    fired windows (kills the rare-token name hijack)."""
    W, D = net.W, net.D
    row = net.concept.W.data                # (D*W, K)
    emb = net.emb.data                      # (V, D)
    X = np.asarray(Xtr)
    # chunked + explicit gc: a single full-set forward costs ~2.7 GB and leaks
    # ~277 MB per 20k chunk through the autograd graph's reference cycles.
    fired = _fired_matrix(net, X)
    names = []
    for k in range(net.K):
        # word support in fired windows
        E = X[fired[:, k]]
        if len(E) == 0:
            names.append(f"neuron{k}")
            continue
        sup = np.zeros(net.V)
        np.add.at(sup, E.ravel(), 1.0)
        sup /= len(E)
        # score: per position block
        blocks = row[:, k].reshape(W, D)        # (W,D)
        score = np.zeros(net.V)
        for p in range(W):
            score += blocks[p] @ emb.T          # (V,) per position
        cand = [(j, score[j]) for j in range(len(d["vocab"]))
                if d["vocab"][j] not in STOP and sup[j] >= min_support
                and score[j] > 0]
        cand.sort(key=lambda t: -t[1])
        ws = [d["vocab"][j] for j, _ in cand[:topn]]
        names.append("_".join(ws[:6]) if ws else f"neuron{k}")
    return names


def auto_name_t2c_pa(net, d, Xtr, topn=8, min_support=0.01):
    """Position-AWARE naming — the quantity the docstring above always described.

    auto_name_t2c scores word j as sum over ALL positions p of block_p . emb_j,
    which credits a word for weight blocks at positions where it never appears.
    The audit measures the contribution at the positions where the word actually
    sits, so the two disagreed and the contribution clause failed for naming
    reasons rather than model reasons.

    Here: score_j = mean over fired windows of (sum over positions p where j
    sits of block_p . emb_j) — exactly the audited quantity, computed on TRAIN
    (the audit then verifies it independently on VAL).
    """
    W, D = net.W, net.D
    row = net.concept.W.data                # (D*W, K)
    emb = net.emb.data                      # (V, D)
    X = np.asarray(Xtr)
    # chunked + explicit gc: a single full-set forward costs ~2.7 GB and leaks
    # ~277 MB per 20k chunk through the autograd graph's reference cycles.
    fired = _fired_matrix(net, X)
    names = []
    for k in range(net.K):
        Xf = X[fired[:, k]]
        if len(Xf) == 0:
            names.append(f"neuron{k}")
            continue
        sup = np.zeros(net.V)
        np.add.at(sup, Xf.ravel(), 1.0)
        sup /= len(Xf)
        blocks = row[:, k].reshape(W, D)              # (W, D)
        counts = np.zeros((W, net.V))
        for p in range(W):
            np.add.at(counts[p], Xf[:, p], 1.0)
        score = (counts * (blocks @ emb.T)).sum(axis=0) / len(Xf)
        cand = [(j, score[j]) for j in range(len(d["vocab"]))
                if d["vocab"][j] not in STOP and sup[j] >= min_support
                and score[j] > 0]
        cand.sort(key=lambda t: -t[1])
        ws = [d["vocab"][j] for j, _ in cand[:topn]]
        names.append("_".join(ws[:6]) if ws else f"neuron{k}")
    return names


def _auroc(score, target01):
    pos = target01 > 0
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def audit_t2c(net, names, d, Xva, min_firing=0.0):
    """Standing meter, position-aware: AUROC (window contains a named word,
    residualized on window length) AND logit contribution (mean over fired
    windows of the total named-word logit gain: for each named word w, sum
    over its positions p in the window of block_p @ emb_w).

    Meter-02 (METER-RULE-02.md) adds a FIRING-RATE FLOOR: the column must fire on
    at least `min_firing` of the validation windows (activation > 0.5). Without it
    a nearly dead column can be named to match its own few firings and score AUROC
    ~1.0 by construction. min_firing=0.0 reproduces the old meter exactly, so every
    earlier result stays reproducible. The firing rate is recorded on every row so
    other floors can be applied later without recomputation.
    """
    X = np.asarray(Xva)
    h = net.forward(X)["concepts"].data
    L = np.log1p(np.asarray([(x != d["V"] - 1).sum() for x in X]))
    A = np.vstack([np.ones_like(L), L]).T
    W, D = net.W, net.D
    emb = net.emb.data
    vmap = {w: i for i, w in enumerate(d["vocab"])}
    rows = []
    for k, nm in enumerate(names):
        firing = float((h[:, k] > 0.5).mean())
        words = set(nm.split("_")) - {"neuron"}
        if not words or nm.startswith("neuron"):
            rows.append({"name": nm, "auroc": float("nan"),
                         "contrib": float("nan"), "firing": firing,
                         "flag": "NO-NAME"})
            continue
        ids = [vmap[w] for w in words if w in vmap]
        contains = np.stack([np.isin(X, i).any(axis=1) for i in ids]).any(0)
        coef, *_ = np.linalg.lstsq(A, h[:, k], rcond=None)
        resid = h[:, k] - A @ coef
        a = _auroc(resid, contains)
        fired = h[:, k] > 0.5
        contrib = 0.0
        if fired.any() and ids:
            bk = net.concept.W.data[:, k].reshape(W, D)   # (W,D)
            tot = 0.0
            for i in ids:
                wvec = bk @ emb[i]                       # (W,) per position
                take = fired & np.isin(X, i).any(axis=1)
                if take.any():
                    # for each fired window with w: sum over its positions
                    Xt = X[take]
                    pos_mask = (Xt == i)                  # (n,W)
                    tot += float((wvec[None, :] * pos_mask).sum())
            contrib = tot / float(fired.sum())
        below_floor = firing < min_firing
        broken = not (a >= 0.55 and contrib >= 0.25) or below_floor
        rows.append({"name": nm, "auroc": a, "contrib": contrib,
                     "pos_frac": float(contains.mean()),
                     "firing": firing,
                     "below_floor": bool(below_floor),
                     "flag": "BROKEN" if broken else "-"})
    n_pass = sum(1 for r in rows if r["flag"] == "-")
    return rows, n_pass


def main():
    os.makedirs(ITERDIR, exist_ok=True)
    d = tiny_data.build(win=WIN)
    t0 = time.time()
    prior = float(np.bincount(d["ytr"], minlength=d["V"]).max()
                  / len(d["ytr"]))
    print(f"[t2-003] prior {prior*100:.2f}% | V={d['V']} | input {D*WIN}-d")

    rvals, trs, nets = {}, {}, {}
    for s in SEEDS:
        ckpt = os.path.join(ITERDIR, f"final_seed{s}.npz")
        if os.path.exists(ckpt):
            # resume: checkpoint exists (e.g. external kill after seed 0)
            net = TinyLM3(d["V"], K, D, WIN, rng=np.random.default_rng(s))
            z = np.load(ckpt)
            net.emb.data[:] = z["emb"]
            net.concept.W.data[:] = z["cW"]
            net.concept.b.data[:] = z["cb"]
            net.answer.W.data[:] = z["aW"]
            net.answer.b.data[:] = z["ab"]
            v = acc_of(net.forward(d["Xva"])["logits"], d["yva"])
            ta = acc_of(net.forward(d["Xtr"][:16384])["logits"],
                        d["ytr"][:16384])
            print(f"  seed{s}: RESUMED from checkpoint (val {v*100:.2f}%)")
        else:
            net, v, ta = train(net_placeholder(d, s), d, s, ITERDIR)
            v = acc_of(net.forward(d["Xva"])["logits"], d["yva"])
            ta = acc_of(net.forward(d["Xtr"][:16384])["logits"],
                        d["ytr"][:16384])
            print(f"  seed{s}: val {v*100:.2f}% | train(16k) {ta*100:.2f}%")
        rvals[s] = v * 100
        trs[s] = ta * 100
        nets[s] = net
    rm, rs = np.mean(list(rvals.values())), np.std(list(rvals.values()))
    tm = np.mean(list(trs.values()))
    print(f"[t2-003] readable VAL {rm:.2f} ± {rs:.2f} | train {tm:.2f}% | "
          f"params {readable_params(d['V'], D, K, WIN)}")

    net0 = nets[0]
    names = auto_name_t2c(net0, d, d["Xtr"], )
    rows, n_pass = audit_t2c(net0, names, d, d["Xva"])
    for r in rows:
        print(f"    {r['name'][:42]:42s} auroc={r['auroc']:.2f} "
              f"contrib={r['contrib']:.3f} {r['flag']}")
    print(f"[t2-003] audit pass {n_pass}/{K}")

    okA = bool(rm >= 2 * prior * 100)
    okB = bool(n_pass / K >= 0.5)
    if okA and okB:
        probe = TinyLM3(d["V"], K, D, WIN, rng=np.random.default_rng(1))
        probe.freeze([2])
        _, _, _ = train(probe, d, 1, ITERDIR, tag="probe")
        pn = auto_name_t2c(probe, d, d["Xtr"])
        pr, _ = audit_t2c(probe, pn, d, d["Xva"])
        lie = pr[2]
        lie_caught = lie["flag"] == "BROKEN" or lie["auroc"] < 0.55
        print(f"[t2-003] lie probe: auroc={lie['auroc']:.2f} "
              f"contrib={lie['contrib']:.3f} -> CAUGHT={lie_caught}")
    else:
        lie_caught = False
        print("[t2-003] lie probe DEFERRED (A and B did not both pass)")
    verdict = "KEEP" if (okA and okB and lie_caught) else "REVERT"
    print(f"[t2-003] A={okA} B={okB} C={lie_caught} -> {verdict} "
          f"({time.time()-t0:.0f}s)")
    json.dump({"iteration": "t2-003", "task": "tiny_shakespeare",
               "mode": "concat-window-readable", "K": K, "D": D, "WIN": WIN,
               "epochs": EPOCHS, "lr": LR, "lam_act": LAM_A, "lam_w": LAM_W,
               "change": "input: sum-bag -> CONCAT window (order+position)",
               "prior": prior, "rval_vals": rvals, "rval_mean": rm,
               "rval_std": rs, "train_acc_mean": tm,
               "readable_params": readable_params(d["V"], D, K, WIN),
               "audit_pass": n_pass, "audit_total": K,
               "lie_caught": lie_caught, "okA": okA, "okB": okB,
               "okC": bool(lie_caught), "verdict": verdict,
               "names": names},
              open(os.path.join(ITERDIR, "results.json"), "w"), indent=1)
    print("saved", os.path.join(ITERDIR, "results.json"))


if __name__ == "__main__":
    main()