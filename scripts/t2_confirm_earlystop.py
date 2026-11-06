"""t2_confirm_earlystop.py — reviewer item (1): confirm the early-stopping
explanation for the "dead" readable run.

ONE CHANGE vs the Step 2 sweep: early stopping may not fire before epoch 30.
Every other component is IMPORTED from the sweep module, so the model init,
shuffle, lr, batch, evaluation and patience are byte-identical to the runs that
produced the Step 2 numbers.

Stopping semantics (stated precisely so there is no ambiguity later):
  - `bad` counts consecutive epochs without a strict val-accuracy improvement.
  - the stop fires only when `bad >= PATIENCE` AND the epoch number is >= 30.
  - so a run that never improves after epoch 1 stops at EXACTLY epoch 30, and a
    run that improves keeps going until 15 epochs without improvement, never
    before epoch 30.

Runs: readable seeds 1 (the dead one) and 4 (live control);
      normal seeds 0, 1, 2 (live controls).

Pre-registration and the binary success criterion are in LOG.md, written
before this file was run.
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

# identical components, imported — not re-written
from t2_step2_sweep import eval_chunked, make_model, fwd_of, MAX_EPOCHS, PATIENCE
from run_t2_iter003 import LR, BATCH

MIN_EPOCH_BEFORE_STOP = 30          # <-- THE ONE CHANGE
RUNS = [("readable", 1), ("readable", 4),
        ("normal", 0), ("normal", 1), ("normal", 2)]
RUNDIR = os.path.join(PROJ, "runs", "t2", "confirm")
os.makedirs(RUNDIR, exist_ok=True)
OUT = os.path.join(RUNDIR, "confirm.json")
LOG = os.path.join(RUNDIR, "confirm.log")
UNK = 8000


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def train(kind, net, d, seed):
    rngd = np.random.default_rng(seed * 131 + 7)      # same shuffle as the sweep
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
            log(f"    -> early stop at epoch {stopped_at} (min-epoch rule)")
            break
    return best, hist, stopped_at


def plateau_len(hist, baseline):
    """Leading epochs whose val accuracy equals the majority baseline."""
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
    log("REVIEWER ITEM (1): confirm the early-stopping explanation")
    log(f"ONE CHANGE: no early stop before epoch {MIN_EPOCH_BEFORE_STOP} "
        f"(patience {PATIENCE}, max {MAX_EPOCHS})")
    log(f"majority baseline on val = {baseline:.4f}% (class {UNK} = '<unk>')")
    log("=" * 74)

    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    keys = {(r["model"], r["seed"]) for r in done}

    for kind, seed in RUNS:
        if (kind, seed) in keys:
            log(f"  [{kind} seed {seed}] already done — skipped")
            continue
        log(f"  [{kind} seed {seed}] training")
        t0 = time.time()
        net = make_model(kind, V, seed)               # identical init to the sweep
        best, hist, stopped = train(kind, net, d, seed)
        rec = {"model": kind, "seed": seed, "val": best["val"],
               "best_epoch": best["epoch"], "stopped_at": stopped,
               "plateau_len": plateau_len(hist, baseline),
               "hist": [round(h, 4) for h in hist],
               "secs": round(time.time() - t0)}
        done.append(rec)
        json.dump(done, open(OUT, "w"))
        log(f"    -> {kind} seed {seed}: val {best['val']:.2f}% @ ep "
            f"{best['epoch']}, stopped {stopped}, plateau {rec['plateau_len']} "
            f"epochs, {rec['secs']}s")

    # ---- verdict against the pre-registered criterion ----
    log("")
    log("VERDICT vs PRE-REGISTERED CRITERION")
    s1 = next((r for r in done if r["model"] == "readable" and r["seed"] == 1), None)
    if s1:
        rec = s1["val"] > baseline + 1e-9
        log(f"  readable seed 1: val {s1['val']:.2f}% vs baseline "
            f"{baseline:.2f}% -> {'RECOVERED' if rec else 'DID NOT RECOVER'}")
        if rec:
            log("  BINARY CRITERION: CONFIRMED — the Step 2 mean is contaminated "
                "by early stopping")
        else:
            log("  BINARY CRITERION: NOT CONFIRMED — the dead run is a genuine "
                "failure, not an early-stopping artifact")
    log("")
    for r in done:
        log(f"  {r['model']:9s} seed {r['seed']}: val {r['val']:6.2f}% "
            f"@ ep {r['best_epoch']:3d}  stopped {r['stopped_at']:3d}  "
            f"plateau {r['plateau_len']} epochs")
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
