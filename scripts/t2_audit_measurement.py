"""t2_audit_measurement.py — STEP 1: audit the train/val measurement.

Question (Claude item 2): why is the readable's TRAIN accuracy (14.43) BELOW
its VAL accuracy (18.31)?

Suspected harness causes (not model):
 (A) train acc is measured on a FIXED chunk Xtr[:16384] = the HEAD of the
     corpus (windows come from contiguous positional slices), so it is a
     biased sample;
 (B) VAL is a MAX-OVER-EPOCHS statistic (best_v) while train is a single
     point estimate -> val is optimistically inflated.

Also delivers Claude item 4's majority-word baseline and item 3's loss
numbers. No training. Test split never touched.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
import numpy as np

from glassnet import tiny_data
from run_t2_iter003 import K, D, WIN, TinyLM3, acc_of
from train_t2_normal_concat import TinyNormal

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runs", "t2")
RNG = np.random.default_rng(12345)


def plain_ce(logits, y):
    """Mean cross-entropy (no sparsity terms) — comparable across models."""
    z = logits.data
    z = z - z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    p = ex / ex.sum(axis=1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def load_readable(path, V):
    net = TinyLM3(V, K, D, WIN)
    z = np.load(path)
    net.emb.data[:] = z["emb"]
    net.concept.W.data[:] = z["cW"]
    net.concept.b.data[:] = z["cb"]
    net.answer.W.data[:] = z["aW"]
    net.answer.b.data[:] = z["ab"]
    return net


def load_normal(path, V):
    net = TinyNormal(V, K, D, WIN)
    z = np.load(path)
    net.emb.data[:] = z["emb"]
    net.h1.W.data[:] = z["h1.W"]
    net.h1.b.data[:] = z["h1.b"]
    net.h2.W.data[:] = z["h2.W"]
    net.h2.b.data[:] = z["h2.b"]
    return net


def eval_chunked(fwd, X, y, chunk=4096):
    """Accuracy + plain CE in chunks — a full-set forward would build a
    len(X) x 8001 logits array (182k x 8001 x 4B = 5.8 GB) and thrash."""
    n = len(X)
    correct, ce_sum = 0, 0.0
    for i in range(0, n, chunk):
        xs, ys = X[i:i + chunk], y[i:i + chunk]
        z = fwd(xs).data
        z = z - z.max(axis=1, keepdims=True)
        ex = np.exp(z)
        p = ex / ex.sum(axis=1, keepdims=True)
        correct += int((z.argmax(axis=1) == ys).sum())
        ce_sum += float(-np.log(p[np.arange(len(ys)), ys] + 1e-12).sum())
    return correct / n * 100, ce_sum / n


def measure(fwd, tag, d, V, vocab, res):
    Xtr, ytr = d["Xtr"], d["ytr"]
    head = min(16384, len(Xtr))
    ridx = RNG.choice(len(Xtr), size=head, replace=False)
    print(f"  [{tag}]", flush=True)
    rows = {}
    for name, X, y in (("val (full)", d["Xva"], d["yva"]),
                       ("train head[:16384]", Xtr[:head], ytr[:head]),
                       ("train random 16384", Xtr[ridx], ytr[ridx]),
                       ("train FULL", Xtr, ytr)):
        a, ce = eval_chunked(fwd, X, y)
        rows[name] = {"n": int(len(y)), "acc": a, "ce": ce}
        print(f"    {name:22s} n={len(y):6d} acc={a:6.2f}%  ce={ce:.3f}",
              flush=True)
    unk = V - 1
    for nm, y in (("val targets", d["yva"]), ("train-head targets", ytr[:head]),
                  ("train-FULL targets", ytr)):
        vals, cnts = np.unique(y, return_counts=True)
        t = cnts.argmax()
        print(f"    {nm:22s} n={len(y):6d} unk={np.mean(y == unk)*100:5.2f}% "
              f"top={wname(V, vocab, int(vals[t]))!r} "
              f"share={cnts[t]/len(y)*100:5.2f}%", flush=True)
    res[tag] = rows
    return rows


def wname(V, vocab, cls):
    """Class -> display name ('<unk>' for the final unknown class)."""
    return "<unk>" if cls == V - 1 else vocab[cls]


def main():
    d = tiny_data.build(win=WIN)
    V, vocab = d["V"], d["vocab"]
    ytr, yva = d["ytr"], d["yva"]

    vals, cnts = np.unique(ytr, return_counts=True)
    maj = int(vals[cnts.argmax()])
    maj_val = float(np.mean(yva == maj)) * 100
    maj_train = float(np.mean(ytr == maj)) * 100
    print(f"MAJORITY-WORD BASELINE: most frequent train target = "
          f"{wname(V, vocab, maj)!r} (class {maj}); {maj_val:.2f}% of val, "
          f"{maj_train:.2f}% of train", flush=True)
    print(f"  -> always predicting it scores {maj_val:.2f}% on val\n",
          flush=True)

    out = {"majority": {"class": maj, "word": wname(V, vocab, maj),
                        "val_acc": maj_val, "train_acc": maj_train}}

    rnet = load_readable(os.path.join(RUNS, "iter003", "final_seed0.npz"), V)
    measure(lambda X: rnet.forward(X)["logits"], "readable(iter003 best-val)",
            d, V, vocab, out)

    # the normal's trainer saved FINAL weights; scan its partials for the
    # true best-val checkpoint (harness bug found by this audit)
    ndir = os.path.join(RUNS, "normal_concat")
    cands = sorted(f for f in os.listdir(ndir)
                   if f.startswith("normal_seed0") and f.endswith(".npz"))
    best = (None, -1.0, None)
    for f in cands:
        n = load_normal(os.path.join(ndir, f), V)
        v, _ = eval_chunked(lambda X: n.forward(X), d["Xva"], d["yva"])
        print(f"    ckpt {f:28s} val {v:6.2f}%", flush=True)
        if v > best[1]:
            best = (f, v, n)
    print(f"  [normal] BEST checkpoint = {best[0]} (val {best[1]:.2f}%)\n",
          flush=True)
    out["normal_best_ckpt"] = {"file": best[0], "val": best[1]}
    measure(lambda X: best[2].forward(X), "normal(best-val ckpt)", d, V, vocab,
            out)

    json.dump(out, open(os.path.join(RUNS, "measure_audit.json"), "w"), indent=1)
    print("\nsaved runs/t2/measure_audit.json", flush=True)


if __name__ == "__main__":
    main()