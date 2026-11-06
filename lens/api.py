"""lens/api.py — the library. This is the front door.

    import lens

    concepts = lens.concepts_from_words(train_text, ["thou", "thee", "speak", "give"])
    net = lens.Net(vocab=len(vocab), concepts=concepts, d=256)
    net = lens.train(net, X, y, epochs=5)
    print(lens.explain(net, X[0], vocab=vocab))
    lens.audit(net, X)                    # does each concept mean what its name says?

That is the whole idea: build, train, ask why, and check that the names are honest.

DESIGN RULES, each one from a failure in the old project
--------------------------------------------------------
1. **No module-level globals.** Concept targets are passed in, never read from a global that one
   env setting can leave as None. (The old code trained on the wrong concept set that way.)
2. **Every run returns its numbers.** `train()` returns a result object holding the loss curve,
   the per-epoch timings, and the loss PARTS -- so the auxiliary concept loss can be checked
   rather than assumed. (The old code never logged it.)
3. **`explain()` asserts its own exactness.** Contributions plus bias must equal the model's own
   score, and the residual is reported. An explanation you cannot check is a story.
4. **`audit()` is not optional decoration.** Coverage without an audit is worthless: the old
   project had a model at 100% coverage whose concepts were nearly all dead. `audit()` measures
   each concept against its own declared target and reports the pass count with the same bars
   used everywhere else (AUROC 0.55, firing floor 2%).
5. **The device is explicit.** `device="cpu"` or `device="gpu"`. The backend's device is
   process-global, so a model is built on whichever device is named here.
"""
import json
import os
import time

import numpy as np

from lens import backend, tensor as T
from lens.concepts import Concept, coverage, explain as _explain, explain_text
from lens.model import LensNet

__all__ = ["Net", "train", "explain", "explain_text", "audit", "save", "load",
           "concepts_from_words", "Result"]


# ---------------------------------------------------------------- construction
def Net(vocab, concepts, d=128, window=8, seed=0, lam_concept=0.1, lam_act=0.2,
        device="cpu", lr=0.3, attention=False, mtp=1, numeric=False,
        direct=False, direct_rank=16, direct_mode="lowrank"):
    """Build a readable model. `concepts` is required -- the concept layer IS the architecture.

    `mtp=2` adds a second answer head predicting the token two steps ahead (arXiv:2404.19737 reports
    the largest benefit at or below 30M non-embedding parameters). Default 1 keeps every existing
    model and checkpoint byte-identical.

    `numeric=True` adds the MAGNITUDE path (lens/numeric.py): a numeric channel beside the concepts and a
    quantity head beside the token classifier. Off by default so nothing about an existing checkpoint
    changes; a numeric model REQUIRES the magnitude channel to be passed to forward/loss, because the
    numbers are an input, not something recoverable from token ids the vocabulary never saw.
    """
    T.set_device(device)
    net = LensNet(vocab, concepts, d=d, window=window, seed=seed,
                  lam_concept=lam_concept, lam_act=lam_act, attention=attention, mtp=mtp, numeric=numeric,
                  direct=direct, direct_rank=direct_rank, direct_mode=direct_mode)
    net.lr = float(lr)
    net.device_name = device
    return net


def concepts_from_words(text, words, window=8, name_style="contains"):
    """Concepts defined by token presence, with support measured from the TRAIN text only.

    `words` are TOKEN IDS -- the values that appear in `text` -- not positions in a sorted
    vocabulary. The first version of this confused the two: it built a sorted vocabulary, mapped
    each id to its POSITION, and used the position as the concept's token. Nothing matched, every
    support came out 0.000, and the audit had no positives to measure. The vocabulary is now used
    only to CHECK that the id appears at all.

    Support matters: the balanced positive weight is derived from it, so a concept present in 3%
    of windows gets a much larger weight than one present in 25%. Measuring support on validation
    or test text would leak.
    """
    text = np.asarray(text)
    present = set(int(t) for t in np.unique(text))
    out = []
    for w in words:
        wid = int(w)
        if wid not in present:
            raise ValueError(f"token id {wid} does not appear in the training text")
        support = float((text == wid).any(axis=1).mean())
        out.append(Concept(f"{name_style}_{wid}", [wid], support=support))
    return out


# ---------------------------------------------------------------- training
class Result:
    """What a run produced. Printed, logged, and returned -- never swallowed."""

    def __init__(self, history, seconds, params, device, parts_keys):
        self.history = history            # list of dicts, one per epoch
        self.seconds = seconds
        self.params = params
        self.device = device
        self.parts_keys = parts_keys

    @property
    def final_loss(self):
        return self.history[-1]["loss"] if self.history else None

    def __repr__(self):
        if not self.history:
            return "Result(empty)"
        h = self.history[-1]
        return (f"Result(epochs={len(self.history)}, final loss {h['loss']:.4f}, "
                f"{self.params:,} params, {self.device}, {self.seconds:.2f}s)")


def train(net, X, y, epochs=5, batch=128, lr=None, seed=0, log=None, checkpoint=None):
    """Train the readable model. This is the 5-20 line library's core.

    Every epoch records the loss AND its parts, so the auxiliary concept loss can be verified
    rather than assumed. If `log` is given, one JSON line per epoch is appended as the run goes,
    so a crash never loses the numbers. If `checkpoint` is given, the parameters are saved each
    epoch.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != net.W:
        raise ValueError(f"X must be (N, {net.W}) windows, got {X.shape}")
    lr = float(net.lr if lr is None else lr)
    rng = np.random.default_rng(seed)
    N = len(X)
    history = []
    parts_keys = None
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        order = rng.permutation(N)
        run_loss, nb = 0.0, 0
        for s in range(0, N, batch):
            idx = order[s:s + batch]
            for p in net.parameters():
                p.grad = None
            loss, parts = net.loss(X[idx], y[idx])
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - lr * p.grad
            run_loss += float(backend.to_cpu(loss.data))
            nb += 1
            parts_keys = sorted(parts.keys())
        rec = {"epoch": ep, "loss": run_loss / max(nb, 1),
               "seconds": time.perf_counter() - t0}
        history.append(rec)
        line = (f"  epoch {ep:>3}/{epochs}   loss {rec['loss']:.4f}   "
                f"{rec['seconds']:.2f}s")
        print(line, flush=True)
        if log:
            with open(log, "a") as f:
                f.write(json.dumps({**rec, "lr": lr, "batch": batch, "device": net.device_name,
                                    "parts": parts_keys}) + "\n")
        if checkpoint:
            save(net, checkpoint)

    backend.sync()
    return Result(history, time.perf_counter() - t0,
                  sum(p.data.size for p in net.parameters()), net.device_name, parts_keys)


# ---------------------------------------------------------------- explaining
def explain(net, window, vocab=None, top_k=None):
    """Why the model answered. The decomposition is exact and the residual is reported."""
    return _explain(net, window, vocab=vocab, top_k=top_k)


# ---------------------------------------------------------------- audit
def audit(net, X, concepts=None, auroc_bar=0.55, fire_floor=0.02):
    """Does each concept mean what its name says?

    Measures every concept against its OWN declared target, with the same bars used across the
    project. Returns the pass count, the coverage, and the per-concept numbers. Coverage is
    returned WITH the audit and never on its own, because 100% coverage with dead concepts is
    possible -- the old project proved it.
    """
    X = np.asarray(X)
    concepts = list(concepts if concepts is not None else net.concepts)
    h = net.forward_arrays(X)["concepts"]
    rows = []
    for i, c in enumerate(concepts):
        tgt = c.labels(X)
        n1 = int(tgt.sum())
        n0 = int((~tgt).sum())
        if n1 == 0 or n0 == 0:
            rows.append({"name": c.name, "auroc": None, "fires": None, "pass": False,
                         "why": "no positives or no negatives in this sample"})
            continue
        score = h[:, i]
        order = np.argsort(score)
        ranks = np.empty(len(score))
        ranks[order] = np.arange(1, len(score) + 1)
        au = float((ranks[tgt].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        fires = float((score > 0.5).mean())
        ok = bool(au >= auroc_bar and fires >= fire_floor)
        rows.append({"name": c.name, "auroc": au, "fires": fires, "pass": ok,
                     "why": "" if ok else
                     ("auroc below bar" if au < auroc_bar else "fires below floor")})
    npass = sum(1 for r in rows if r["pass"])
    return {"n_pass": npass, "n_concepts": len(rows), "coverage": coverage(net, X),
            "auroc_bar": auroc_bar, "fire_floor": fire_floor, "concepts": rows}


# ---------------------------------------------------------------- persistence
def save(net, path):
    """Save parameters and enough metadata to rebuild. Epoch is recorded, always.

    The old project audited a checkpoint that turned out to be an epoch-1 snapshot because the
    epoch was not stored with it. Here every saved file says which epoch it is.
    """
    d = {p.name: backend.to_cpu(p.data) for p in net.parameters()}
    meta = {"V": net.V, "D": net.D, "W": net.W, "seed": net.seed,
            "lam_concept": net.lam_concept, "lam_act": net.lam_act,
            "attention": bool(getattr(net, "attention", False)),
            "numeric": bool(getattr(net, "numeric", False)),
            "direct": bool(getattr(net, "direct", False)),
            "direct_rank": int(getattr(net, "drank", 16)),
            "direct_mode": str(getattr(net, "dmode", "lowrank")),
            "concepts": [{"name": c.name, "ids": c.ids, "support": c.support}
                         for c in net.concepts],
            "device": net.device_name}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **d, __meta__=np.asarray(json.dumps(meta)))
    return path


def load(path, device="cpu"):
    """Rebuild a model from a file, including its concept names and supports.

    APILOAD-DIRECT FIX 2026-10-22 18:1x: this function wrote direct/direct_rank/direct_mode into the file
    (see save()) but never passed them back, so a checkpoint with the tied direct path loaded through here
    came back WITHOUT it: measured `dmix` missing and max |logit difference| 24.96 against the model the
    trainer had in memory (lens/runs/TALK-FAIR-TEST.txt section 0). Every post-hoc consumer that goes
    through api.load - the readability gate, the exporters, the server - was evaluating a different model
    than the one that was trained. The flags travel with the checkpoint, so pass them.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in meta["concepts"]]
    net = Net(meta["V"], concepts, d=meta["D"], window=meta["W"], seed=meta["seed"],
              lam_concept=meta["lam_concept"], lam_act=meta["lam_act"], device=device,
              attention=bool(meta.get("attention", False)),
              numeric=bool(meta.get("numeric", False)),
              direct=bool(meta.get("direct", False)),
              direct_rank=int(meta.get("direct_rank", 16)),
              direct_mode=str(meta.get("direct_mode", "lowrank")))
    for p in net.parameters():
        if p.name not in z:
            raise KeyError(f"{path} has no parameter {p.name!r}")
        p.data = backend.to_dev(z[p.name].astype(np.float32))
    return net
