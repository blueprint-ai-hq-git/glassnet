"""train_image.py — the IMAGE model, on our own engine, with named concepts.

Compares, on the same data, same epochs, same seed:
  READABLE  -- image -> patches -> 15-18 NAMED concept neurons -> answer
  PLAIN     -- a normal MLP with the SAME parameter count

The rule from the project: always compare against a normal network with a similar parameter count,
and report both parameter counts. A win smaller than the seed noise is called "no difference".
"""
import argparse
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import image_concepts                                     # noqa: E402
import lens                                               # noqa: E402
import lens.tensor as T                                   # noqa: E402
from lens import backend as B                             # noqa: E402
from lens.model import Dense                              # noqa: E402
from lens.vision import VisionNet                         # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "image")
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(m, flush=True)


def load_fashion():
    cache = os.path.join(PROJ, "data", "fashion_mnist_cache.npz")
    d = np.load(cache)
    Xtr, ytr = d["X_train"].astype(np.float32), d["y_train"].astype(np.int64)
    Xte, yte = d["X_test"].astype(np.float32), d["y_test"].astype(np.int64)
    if Xtr.max() > 1.5:
        Xtr, Xte = Xtr / 255.0, Xte / 255.0
    return Xtr, ytr, Xte, yte


class PlainMLP:
    """The control: same job, no named concepts. A plain hidden layer."""

    def __init__(self, n_in, hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.l1 = Dense(n_in, hidden, rng, name="plain1")
        self.l2 = Dense(hidden, n_out, rng, name="plain2")

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def forward(self, X):
        x = T.Tensor(B.to_dev(np.asarray(X, dtype=np.float32)))
        return self.l2(T.relu(self.l1(x)))

    def loss(self, X, y):
        lg = self.forward(X)
        return T.softmax_ce(lg, y), {"ce": 0.0}


def audit(net, X, y, concepts):
    """Does each neuron mean what its label says? AUROC vs its own formula + firing rate."""
    out = net.forward_arrays(X)
    h = out["concepts"]
    Aw = B.to_cpu(net.answer.W.data)                     # (K, C)
    res = []
    for k, c in enumerate(concepts):
        lab = c.labels(X)
        a = h[:, k]
        # AUROC by rank comparison, no library
        pos, neg = a[lab > 0.5], a[lab < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            au = None
        else:
            order = np.argsort(np.concatenate([pos, neg]))
            ranks = np.empty(len(order), dtype=np.float64)
            ranks[order] = np.arange(1, len(order) + 1)
            rp = ranks[:len(pos)].sum()
            au = float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        fires = float((a > 0.5).mean())
        contrib = float(np.abs(Aw[k]).mean() * a.mean())
        passed = (au is not None and au >= 0.55) and (fires >= 0.02) and (contrib >= 0.05)
        res.append({"name": c.name, "auroc": au, "fires": fires, "contrib": contrib,
                    "support": c.support, "pass": bool(passed)})
    npass = sum(r["pass"] for r in res)
    return {"concepts": res, "n_pass": npass, "n_concepts": len(res),
            "coverage": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--patch", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--lr-decay", type=float, default=0.5)
    ap.add_argument("--lam-concept", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default="img1")
    ap.add_argument("--train-n", type=int, default=60000)
    ap.add_argument("--extra", default="none", choices=["none", "real", "dummy"],
                    help="K-transfer test (2026-10-08): append the FROZEN extended concepts "
                         "(real) or random pixel projections (dummy) to the base 12, using the "
                         "SAME support-floor filter the seetalk sweep used. Nothing else about "
                         "this proven loop changes.")
    ap.add_argument("--extra-k", type=int, default=10)
    ap.add_argument("--no-plain", action="store_true")
    a = ap.parse_args()

    log("=" * 100)
    log("THE IMAGE MODEL — patches in, NAMED concepts in the middle, classes out")
    log("=" * 100)
    T.set_device(a.device)

    Xtr, ytr, Xte, yte = load_fashion()
    if a.train_n < len(Xtr):
        sel = np.random.default_rng(0).permutation(len(Xtr))[:a.train_n]
        Xtr, ytr = Xtr[sel], ytr[sel]
    log(f"  Fashion-MNIST (MIT)  train {Xtr.shape}  test {Xte.shape}  "
        f"classes {len(np.unique(ytr))}  pixels in [{Xtr.min():.2f},{Xtr.max():.2f}]")

    log("")
    log("  --- the NAMED concepts, each with its formula ---")
    concepts, cspec = image_concepts.build(Xtr, min_support=0.02, verbose=True)
    log(f"  {cspec['kept']} kept, {cspec['dropped']} dropped of {cspec['n_defined']} defined")
    if a.extra != "none":
        # K-TRANSFER TEST (2026-10-08): does the seetalk finding "K=12 is starved, +10 dims buys
        # +4.4, real ~= dummy" hold for the SEE-ONLY model too? Same proven loop; only the concept
        # set changes. Extended definitions come from the FROZEN registry (concepts_more/
        # concepts_sweep), support floor 0.02 exactly as the sweep used. NOTE: such a checkpoint
        # loads only through train_seetalk's extended registry, not lens.vision.load_vision.
        if a.extra == "real":
            from concepts_more import FormulaConcept, extended_specs
            pool = extended_specs()
        else:
            from concepts_sweep import FormulaConcept, dummy_specs_n
            pool = dummy_specs_n(Xtr, seed=123, count=a.extra_k)
        n0 = len(concepts)
        for name, fn, f in pool:
            if len(concepts) >= n0 + a.extra_k:
                break
            lab = np.asarray(fn(Xtr), dtype=np.float32)
            s = float(lab.mean())
            if s < 0.02 or s > 0.98:
                continue
            concepts.append(FormulaConcept(name, fn, f, support=s))
        cspec["extra_kind"] = a.extra
        cspec["extra_kept"] = len(concepts) - n0
        log(f"  EXTRA {a.extra}: +{len(concepts) - n0} concepts -> K={len(concepts)}")

    net = VisionNet(concepts, n_classes=10, d=a.d, patch=a.patch, seed=a.seed,
                    lam_concept=a.lam_concept)
    npar = int(sum(p.data.size for p in net.parameters()))
    log("")
    log(f"  READABLE: {npar:,} params   patches {net.P} x {a.d}  ->  {net.K} concepts  ->  10")
    log(f"            the middle layer is {net.K} named neurons; that is what a human reads.")

    # matched plain control: 784 -> h -> 10, same total parameter count
    n_in, n_out = 784, 10
    h_hidden = max(2, round((npar - n_out - 10) / (n_in + n_out)))
    plain = PlainMLP(n_in, h_hidden, n_out, seed=a.seed)
    ppar = int(sum(p.data.size for p in plain.parameters()))
    log(f"  PLAIN   : {ppar:,} params  784 -> {h_hidden} -> 10   ({abs(ppar-npar)/npar*100:.2f}% "
        f"difference in size)")

    results = {}
    arms = [("readable", net, lambda m, X: m.forward(X))]
    if not a.no_plain:
        arms.append(("plain", plain, lambda m, X: m.forward(X)))
    for tag, model, forward_fn in arms:
        log("")
        log(f"  --- training {tag} ---")
        rng = np.random.default_rng(a.seed)
        hist = []
        for ep in range(1, a.epochs + 1):
            t0 = time.perf_counter()
            order = rng.permutation(len(Xtr))
            lr_now = a.lr * (1.0 - a.lr_decay * (ep - 1) / max(a.epochs - 1, 1))
            tot, nb = 0.0, 0
            for s in range(0, len(Xtr), a.batch):
                idx = order[s:s + a.batch]
                for p in model.parameters():
                    p.grad = None
                loss, parts = model.loss(Xtr[idx], ytr[idx])
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        p.data = p.data - lr_now * p.grad
                tot += float(B.to_cpu(loss.data))
                nb += 1
            # eval
            def predict(model, X):
                """Logits as host numpy. VisionNet.forward returns a dict; PlainMLP returns a
                Tensor -- so ask for arrays where that exists instead of guessing the type."""
                if hasattr(model, "forward_arrays"):
                    return model.forward_arrays(X)["logits"]
                out = model.forward(X)
                return B.to_cpu(out.data if hasattr(out, "data") else out)

            def acc(X, y):
                right = 0
                for s in range(0, len(X), 512):
                    idx = np.arange(s, min(s + 512, len(X)))
                    lg = predict(model, X[idx])
                    right += int((lg.argmax(axis=1) == y[idx]).sum())
                return right / len(X)
            tra = acc(Xtr[:10000], ytr[:10000])
            tea = acc(Xte, yte)
            rec = {"epoch": ep, "loss": tot / max(nb, 1), "train_acc": tra, "test_acc": tea,
                   "lr": lr_now, "seconds": time.perf_counter() - t0}
            hist.append(rec)
            log(f"    ep {ep:>2}/{a.epochs}  loss {rec['loss']:.4f}  train {tra*100:6.2f}%  "
                f"TEST {tea*100:6.2f}%  lr {lr_now:.3f}  ({rec['seconds']:.0f}s)")
            results[tag] = {"history": hist, "params": npar if tag == "readable" else ppar}
            if tag == "readable":
                # checkpoint every epoch, and the last one is what the virtual-camera and any
                # downstream test loads. A crash never costs more than one epoch.
                from lens.vision import save_vision
                save_vision(net, os.path.join(OUT, f"{a.out}_ep{ep}.npz"), kind="vision")

    log("")
    log("=" * 100)
    log("THE FIDELITY AUDIT — does each image neuron mean what its name says?")
    log("=" * 100)
    aud = audit(net, Xte, yte, concepts)
    for c in aud["concepts"]:
        au = "None " if c["auroc"] is None else f"{c['auroc']:.4f}"
        log(f"    {c['name']:<24} AUROC {au}  fires {c['fires']:>6.3f}  "
            f"contrib {c['contrib']:.3f}  {'PASS' if c['pass'] else 'FAIL'}")
    log(f"    -> {aud['n_pass']}/{aud['n_concepts']} pass")

    log("")
    log("=" * 100)
    log("READABLE vs PLAIN")
    log("=" * 100)
    r = results["readable"]
    log(f"    readable  {r['params']:>8,} params   TEST {r['history'][-1]['test_acc']*100:6.2f}%")
    if "plain" in results:
        p = results["plain"]
        log(f"    plain     {p['params']:>8,} params   TEST {p['history'][-1]['test_acc']*100:6.2f}%")
        delta = (r['history'][-1]['test_acc'] - p['history'][-1]['test_acc']) * 100
        log(f"    delta {delta:+.2f} points")
    log(f"    concepts passing {aud['n_pass']}/{aud['n_concepts']}")

    json.dump({"args": vars(a), "concepts": cspec, "audit": aud, "results": results,
               "readable_params": npar, "plain_params": ppar},
              open(os.path.join(OUT, f"{a.out}_log.json"), "w"), indent=1)
    log("")
    log(f"  saved + {a.out}_log.json")
    log("=" * 100)


if __name__ == "__main__":
    main()
