"""t2_step2_sweep.py — STEP 2 (+ STEP 3 early stopping folded in).

5 seeds (0-4) per model, identical protocol for both:
  max 48 epochs, early stopping patience 10 on val accuracy, best-val
  weights kept. lr 0.1, batch 128, concat window, same data order per seed.

Reporting (fixed by the Step-1 audit):
  - val accuracy on the full val split
  - val accuracy EXCLUDING <unk> targets (matched basis — val has 16.32%
    unk vs train 9.72%, so raw accuracy is inflated)
  - train accuracy on a fixed random 32768 sample (representative; the old
    head-chunk metric was biased)
  - best epoch, stop epoch
  - paired per-seed difference + McNemar on the same val examples
  - the majority-word baseline as the reference line

Resume-safe: appends one record per (model, seed) immediately.
VALIDATION only — the test split is never read.
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
from glassnet.tensor import relu
from glassnet.losses import cross_entropy
from glassnet.optim import SGD
from run_t2_iter003 import (K, D, WIN, LR, BATCH, LAM_A, TinyLM3, embedding,
                            _flatten)
from train_t2_normal_concat import TinyNormal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(HERE, "runs", "t2", "step2")
os.makedirs(RUNDIR, exist_ok=True)
OUT = os.path.join(RUNDIR, "sweep.json")
LOG = os.path.join(RUNDIR, "sweep.log")

MAX_EPOCHS = int(os.environ.get("T2_MAX_EPOCHS", "48"))
PATIENCE = int(os.environ.get("T2_PATIENCE", "10"))
SEEDS = (0, 1, 2, 3, 4)
MODELS = ("readable", "normal")
TRAIN_EVAL_N = 32768


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def eval_chunked(fwd, X, y, chunk=4096):
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


def per_example_correct(fwd, X, y, chunk=4096):
    out = np.empty(len(X), dtype=bool)
    for i in range(0, len(X), chunk):
        z = fwd(X[i:i + chunk]).data
        out[i:i + chunk] = z.argmax(axis=1) == y[i:i + chunk]
    return out


def make_model(kind, V, seed):
    if kind == "readable":
        return TinyLM3(V, K, D, WIN, rng=np.random.default_rng(seed))
    return TinyNormal(V, K, D, WIN, rng=np.random.default_rng(seed))


def fwd_of(kind, net):
    return (lambda X: net.forward(X)["logits"]) if kind == "readable" \
        else (lambda X: net.forward(X))


def train(kind, net, d, seed, Xte_eval, yte_eval):
    rngd = np.random.default_rng(seed * 131 + 7)
    opt = SGD(net.parameters(), lr=LR)
    X, y = d["Xtr"], d["ytr"]
    fwd = fwd_of(kind, net)
    best = {"val": -1.0, "epoch": 0, "w": None}
    bad = 0
    for ep in range(MAX_EPOCHS):
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            if kind == "readable":
                loss, _ = net.loss(X[idx], y[idx])
            else:
                loss = cross_entropy(net.forward(X[idx]), y[idx])
            loss.backward()
            opt.step()
        v, _ = eval_chunked(fwd, d["Xva"], d["yva"])
        if v > best["val"]:
            best = {"val": v, "epoch": ep + 1,
                    "w": {p.name: p.data.copy() for p in net.parameters()}}
            bad = 0
        else:
            bad += 1
        print(f"    ep{ep+1:3d} val {v:5.2f}% (best {best['val']:.2f} @ "
              f"{best['epoch']}, patience {bad}/{PATIENCE})", flush=True)
        if bad >= PATIENCE:
            print(f"    early stop at epoch {ep+1}", flush=True)
            break
    for p in net.parameters():
        p.data[:] = best["w"][p.name]
    return best, fwd


def main():
    d = tiny_data.build(win=WIN)
    V, vocab = d["V"], d["vocab"]
    unk = V - 1
    yva = d["yva"]
    keep = yva != unk                      # non-unk val examples
    rng = np.random.default_rng(999)
    tr_idx = rng.choice(len(d["Xtr"]), size=TRAIN_EVAL_N, replace=False)

    vals, cnts = np.unique(d["ytr"], return_counts=True)
    maj = int(vals[cnts.argmax()])
    maj_val = float(np.mean(yva == maj)) * 100
    log(f"STEP2: max_epochs={MAX_EPOCHS} patience={PATIENCE} seeds={SEEDS}")
    log(f"  majority-word baseline on val = {maj_val:.2f}% "
        f"(class {maj} = '<unk>'); val unk share "
        f"{np.mean(yva == unk)*100:.2f}%")

    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    keys = {(r["model"], r["seed"]) for r in done}

    for kind in MODELS:
        for seed in SEEDS:
            if (kind, seed) in keys:
                continue
            log(f"  [{kind} seed {seed}] training")
            t0 = time.time()
            net = make_model(kind, V, seed)
            best, fwd = train(kind, net, d, seed, None, None)
            v_all, ce_all = eval_chunked(fwd, d["Xva"], yva)
            v_keep, ce_keep = eval_chunked(fwd, d["Xva"][keep], yva[keep])
            t_acc, t_ce = eval_chunked(fwd, d["Xtr"][tr_idx], d["ytr"][tr_idx])
            t_keep_mask = d["ytr"][tr_idx] != unk
            t_keep, _ = eval_chunked(fwd, d["Xtr"][tr_idx][t_keep_mask],
                                     d["ytr"][tr_idx][t_keep_mask])
            correct = per_example_correct(fwd, d["Xva"], yva)
            rec = {"model": kind, "seed": seed, "max_epochs": MAX_EPOCHS,
                   "patience": PATIENCE, "lr": LR, "batch": BATCH,
                   "lam_act": LAM_A if kind == "readable" else None,
                   "val_acc": v_all, "val_ce": ce_all,
                   "val_acc_excl_unk": v_keep, "val_ce_excl_unk": ce_keep,
                   "train_acc_32k": t_acc, "train_ce_32k": t_ce,
                   "train_acc_32k_excl_unk": t_keep,
                   "best_epoch": best["epoch"], "best_val": best["val"],
                   "secs": round(time.time() - t0),
                   "correct_val": correct.astype(np.uint8).tolist()}
            done.append(rec)
            json.dump(done, open(OUT, "w"))
            log(f"    -> val {v_all:.2f}% (excl-unk {v_keep:.2f}%) | "
                f"train32k {t_acc:.2f}% (excl-unk {t_keep:.2f}%) | "
                f"best ep {best['epoch']} | {rec['secs']}s")

    # ---- summary ----
    log("STEP2 SUMMARY")
    stats = {}
    for kind in MODELS:
        rs = [r for r in done if r["model"] == kind]
        if len(rs) != len(SEEDS):
            log(f"  [{kind}] incomplete: {len(rs)}/{len(SEEDS)}")
            continue
        for key in ("val_acc", "val_acc_excl_unk", "train_acc_32k",
                    "train_acc_32k_excl_unk", "best_epoch"):
            arr = np.array([r[key] for r in rs], dtype=float)
            stats[f"{kind}.{key}"] = {"mean": float(arr.mean()),
                                      "std": float(arr.std()),
                                      "vals": arr.tolist()}
        log(f"  {kind:9s} val {stats[kind+'.val_acc']['mean']:.2f} ± "
            f"{stats[kind+'.val_acc']['std']:.2f} | "
            f"val-excl-unk {stats[kind+'.val_acc_excl_unk']['mean']:.2f} ± "
            f"{stats[kind+'.val_acc_excl_unk']['std']:.2f} | "
            f"train32k {stats[kind+'.train_acc_32k']['mean']:.2f} ± "
            f"{stats[kind+'.train_acc_32k']['std']:.2f} | "
            f"best-ep {stats[kind+'.best_epoch']['mean']:.1f}")
    rr = {r["seed"]: r for r in done if r["model"] == "readable"}
    nn = {r["seed"]: r for r in done if r["model"] == "normal"}
    if len(rr) == len(SEEDS) and len(nn) == len(SEEDS):
        diffs = [rr[s]["val_acc"] - nn[s]["val_acc"] for s in SEEDS]
        dm, ds = float(np.mean(diffs)), float(np.std(diffs))
        se = ds / np.sqrt(len(diffs)) if ds > 0 else 0.0
        log(f"  PAIRED diff (readable - normal) per seed: "
            f"{[round(x,2) for x in diffs]}")
        log(f"  paired mean {dm:+.2f} pts, std {ds:.2f}, SE {se:.2f}, "
            f"t = {dm/se if se else float('nan'):.2f}")
        # McNemar on identical val examples
        b = c = 0
        for s in SEEDS:
            rc = np.array(rr[s]["correct_val"], dtype=bool)
            nc = np.array(nn[s]["correct_val"], dtype=bool)
            b += int((rc & ~nc).sum())
            c += int((~rc & nc).sum())
        stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
        log(f"  McNemar (all seeds pooled): readable-only-correct {b}, "
            f"normal-only-correct {c}, chi2 {stat:.2f} "
            f"({'significant' if stat > 3.84 else 'not significant'} at 0.05)")
        stats["paired"] = {"diffs": diffs, "mean": dm, "std": ds, "se": se,
                           "mcnemar_b": b, "mcnemar_c": c, "chi2": stat}
    json.dump({"runs": done, "stats": stats, "majority_val": maj_val},
              open(OUT, "w"), indent=1)
    log("saved runs/t2/step2/sweep.json")


if __name__ == "__main__":
    main()