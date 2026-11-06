"""t2_bias_init.py — reviewer item (2): output-bias init to log unigram freqs.

ONE CHANGE vs item (1): the FINAL layer's bias starts at the log unigram
frequencies of the TRAINING set, for BOTH models.
    b_j = log( count_train(j) / N_train )   ->  softmax(b) == the unigram prior
  readable: net.answer.b   (Dense K -> V)
  normal:   net.h2.b       (Dense hidden -> V)

Everything else is imported from the sweep module so it is identical to the runs
that produced the Step 2 and item-(1) numbers: same model init, same shuffle,
same lr/batch, same stopping rule (patience 15, max 96, no early stop before
epoch 30).

Same seeds as item (1) so the comparison is PAIRED — the item-(1) runs are the
no-bias control.
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

from glassnet import tiny_data
from glassnet.losses import cross_entropy
from glassnet.optim import SGD

from t2_step2_sweep import eval_chunked, make_model, fwd_of, MAX_EPOCHS, PATIENCE
from run_t2_iter003 import LR, BATCH

MIN_EPOCH_BEFORE_STOP = 30          # same rule as item (1)
RUNS = [("readable", 1), ("readable", 4),
        ("normal", 0), ("normal", 1), ("normal", 2)]
RUNDIR = os.path.join(PROJ, "runs", "t2", "biasinit")
os.makedirs(RUNDIR, exist_ok=True)
OUT = os.path.join(RUNDIR, "biasinit.json")
LOG = os.path.join(RUNDIR, "biasinit.log")
UNK = 8000

# item (1) control values: same seeds, same protocol, NO bias init
CONTROL = {("readable", 1): (18.35, 6), ("readable", 4): (18.15, 4),
           ("normal", 0): (17.94, 1), ("normal", 1): (17.77, 1),
           ("normal", 2): (17.79, 1)}


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def apply_unigram_bias(net, kind, ytr, V):
    """THE ONE CHANGE: set the output bias to log unigram frequencies."""
    counts = np.bincount(np.asarray(ytr), minlength=V).astype(np.float64)
    p = counts / counts.sum()
    b = np.log(np.maximum(p, 1e-12)).astype(np.float32)
    layer = net.answer if kind == "readable" else net.h2
    layer.b.data = b
    return p, b


def train(kind, net, d, seed):
    rngd = np.random.default_rng(seed * 131 + 7)
    opt = SGD(net.parameters(), lr=LR)
    X, y = d["Xtr"], d["ytr"]
    fwd = fwd_of(kind, net)
    best = {"val": -1.0, "epoch": 0}
    bad = 0
    hist = []
    stopped_at = MAX_EPOCHS
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
        hist.append(v)
        if v > best["val"]:
            best = {"val": v, "epoch": ep + 1}
            bad = 0
        else:
            bad += 1
        log(f"    {kind} s{seed} ep {ep+1:3d} val {v:6.2f}% "
            f"(best {best['val']:.2f} @ {best['epoch']}, bad {bad})")
        if bad >= PATIENCE and (ep + 1) >= MIN_EPOCH_BEFORE_STOP:
            stopped_at = ep + 1
            log(f"    -> early stop at epoch {stopped_at}")
            break
    return best, hist, stopped_at


def plateau_len(hist, baseline):
    n = 0
    for v in hist:
        if abs(v - baseline) < 1e-9:
            n += 1
        else:
            break
    return n


def main():
    d = tiny_data.build(win=8)
    V = d["V"]
    yva = d["yva"]
    baseline = float((yva == UNK).mean() * 100)

    log("=" * 74)
    log("REVIEWER ITEM (2): output-bias init to log unigram frequencies")
    log(f"ONE CHANGE: final-layer bias = log(train unigram). Stopping rule = "
        f"item (1) (patience {PATIENCE}, max {MAX_EPOCHS}, no stop before "
        f"{MIN_EPOCH_BEFORE_STOP})")
    log(f"majority baseline on val = {baseline:.4f}%")
    log("=" * 74)

    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    keys = {(r["model"], r["seed"]) for r in done}

    for kind, seed in RUNS:
        if (kind, seed) in keys:
            log(f"  [{kind} seed {seed}] already done — skipped")
            continue
        log(f"  [{kind} seed {seed}] training (with unigram bias init)")
        t0 = time.time()
        net = make_model(kind, V, seed)
        p, b = apply_unigram_bias(net, kind, d["ytr"], V)
        # sanity: softmax(b) must equal the unigram prior
        eb = np.exp(b - b.max())
        sm = eb / eb.sum()
        log(f"    bias init: top classes by softmax(b): "
            f"{[(int(i), round(float(sm[i]), 4)) for i in np.argsort(-sm)[:3]]}"
            f"  p[<unk>]={p[UNK]:.4f}  log p[<unk>]={b[UNK]:.3f}")
        v0, _ = eval_chunked(fwd_of(kind, net), d["Xva"], yva)
        log(f"    val at init (before any training): {v0:.2f}%")
        best, hist, stopped = train(kind, net, d, seed)
        rec = {"model": kind, "seed": seed, "val": best["val"],
               "best_epoch": best["epoch"], "stopped_at": stopped,
               "plateau_len": plateau_len(hist, baseline),
               "val_at_init": round(v0, 4),
               "hist": [round(h, 4) for h in hist],
               "secs": round(time.time() - t0)}
        done.append(rec)
        json.dump(done, open(OUT, "w"))
        cv, cp = CONTROL.get((kind, seed), (None, None))
        log(f"    -> {kind} seed {seed}: val {best['val']:.2f}% @ ep "
            f"{best['epoch']}, stopped {stopped}, plateau {rec['plateau_len']} "
            f"ep  | control (no bias): val {cv}%, plateau {cp} ep")

    log("")
    log("PAIRED COMPARISON vs item (1) controls (same seeds, only the bias differs)")
    log(f"  {'run':22s} {'no-bias val':>11s} {'bias val':>9s} {'no-bias plat':>12s} {'bias plat':>9s}")
    for r in done:
        cv, cp = CONTROL.get((r["model"], r["seed"]), (None, None))
        log(f"  {r['model']+' s'+str(r['seed']):22s} {cv:>11} {r['val']:>9.2f} "
            f"{cp:>12} {r['plateau_len']:>9}")
    rp = [r["plateau_len"] for r in done if r["model"] == "readable"]
    np_ = [r["plateau_len"] for r in done if r["model"] == "normal"]
    log("")
    log(f"  mean plateau  readable {np.mean(rp):.1f} ep  (no-bias control "
        f"{np.mean([CONTROL[(k,s)][1] for k,s in RUNS if k=='readable']):.1f})")
    log(f"  mean plateau  normal   {np.mean(np_):.1f} ep  (no-bias control "
        f"{np.mean([CONTROL[(k,s)][1] for k,s in RUNS if k=='normal']):.1f})")
    log("")
    log("PREDICTIONS: P1 plateau shortens both / P2 readable shortens more / "
        "P3 val unchanged or better / P4 (falsifiable) no change -> the plateau "
        "is about the representation, not the prior")
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
