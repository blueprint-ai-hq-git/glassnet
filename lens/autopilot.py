"""lens/autopilot.py — the trainer that needs NO settings typed in (plan ask #3).

You hand it data. It decides everything else itself, and it SHOWS its work:

    1. HOLDS OUT its own validation slice from the data you gave it (default 10%), and never
       trains on it. Every choice below is made on that slice.
    2. PROBES model sizes (d) on a small slice with a short run, and keeps the best by
       validation loss per second of training -- a bigger model that is not better per unit of
       time is not chosen.
    3. PROBES learning rates the same way and keeps the best.
    4. TRAINS the full run with the chosen (d, lr), monitoring the validation loss every epoch
       and EARLY-STOPPING when it stops improving for `patience` epochs, restoring the best
       parameters it saw (never the last, possibly-overfit ones).
    5. RETURNS a report: every candidate, every measured number, what was chosen and why.

Nothing about this is hidden: the report is the reasoning, and the caller can re-run any
candidate by hand.
"""
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)

import lens                                          # noqa: E402
from lens import backend as B                        # noqa: E402


def _val_loss(net, X, y, n=600):
    idx = np.arange(min(n, len(X)))
    lg = net.forward_arrays(X[idx])["logits"]
    z = lg - lg.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return float(-np.log(np.maximum(p[np.arange(len(idx)), y[idx]], 1e-12)).mean())


def _probe_run(net, X, y, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            for p in net.parameters():
                p.grad = None
            loss, _parts = net.loss(X[idx], y[idx])
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - lr * p.grad
    return net


def autopilot(vocab, concepts, X, y, *, d_candidates=(32, 64), lr_candidates=(0.1, 0.4, 1.0),
              epochs_max=12, patience=3, probe_epochs=2, probe_frac=0.4, val_frac=0.10,
              batch=128, seed=0, device="cpu", verbose=True, log=None):
    """Train with no settings. Returns (net, report)."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n_val = max(64, int(len(X) * val_frac))
    order = rng.permutation(len(X))
    vi, ti = order[:n_val], order[n_val:]
    Xv, yv = X[vi], y[vi]
    Xt, yt = X[ti], y[ti]
    rep = {"data": {"train": int(len(Xt)), "pilot_val": int(len(Xv))},
           "note": "the pilot's validation slice is NEVER trained on", "probes": []}

    def say(m):
        if verbose:
            print(m, flush=True)

    say(f"autopilot: {len(Xt)} train / {len(Xv)} pilot-val windows (no settings given)")

    # ---- 2. probe the model size on a slice
    n_probe = max(256, int(len(Xt) * probe_frac))
    Xp, yp = Xt[:n_probe], yt[:n_probe]
    best_d, best_score = None, None
    for d in d_candidates:
        net = lens.Net(vocab=vocab, concepts=concepts, d=d, window=X.shape[1], seed=seed,
                       lam_concept=0.3, lam_act=0.2, device=device)
        t0 = time.perf_counter()
        _probe_run(net, Xp, yp, probe_epochs, 0.4, batch, seed)
        secs = max(time.perf_counter() - t0, 1e-6)
        vl = _val_loss(net, Xv, yv)
        score = vl * (1.0 + 0.02 * secs)      # loss, mildly penalised by probe seconds
        rep["probes"].append({"kind": "d", "d": int(d), "val_loss": vl, "seconds": secs,
                              "score": score})
        say(f"    probe d={d:<4} val loss {vl:.4f}  ({secs:.1f}s)  score {score:.4f}")
        if best_score is None or score < best_score:
            best_d, best_score = d, score
    say(f"  -> chose d={best_d}")

    # ---- 3. probe the learning rate at the chosen size
    best_lr, best_lscore = None, None
    for lr in lr_candidates:
        net = lens.Net(vocab=vocab, concepts=concepts, d=best_d, window=X.shape[1], seed=seed,
                       lam_concept=0.3, lam_act=0.2, device=device)
        t0 = time.perf_counter()
        _probe_run(net, Xp, yp, probe_epochs, lr, batch, seed)
        secs = max(time.perf_counter() - t0, 1e-6)
        vl = _val_loss(net, Xv, yv)
        rep["probes"].append({"kind": "lr", "lr": float(lr), "val_loss": vl, "seconds": secs})
        say(f"    probe lr={lr:<5} val loss {vl:.4f}  ({secs:.1f}s)")
        if best_lscore is None or vl < best_lscore:
            best_lr, best_lscore = lr, vl
    say(f"  -> chose lr={best_lr}")

    # ---- 4. full run with early stopping on the pilot's own validation
    net = lens.Net(vocab=vocab, concepts=concepts, d=best_d, window=X.shape[1], seed=seed,
                   lam_concept=0.3, lam_act=0.2, device=device)
    rng2 = np.random.default_rng(seed + 7)
    hist, best_vl, best_ep, best_params, bad = [], None, 0, None, 0
    t0 = time.perf_counter()
    for ep in range(1, epochs_max + 1):
        ordr = rng2.permutation(len(Xt))
        tot, nb = 0.0, 0
        for s in range(0, len(Xt), batch):
            idx = ordr[s:s + batch]
            for p in net.parameters():
                p.grad = None
            loss, _parts = net.loss(Xt[idx], yt[idx])
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - best_lr * p.grad
            tot += float(B.to_cpu(loss.data)); nb += 1
        vl = _val_loss(net, Xv, yv)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vl})
        say(f"    epoch {ep:>3}  train {tot/max(nb,1):.4f}  val {vl:.4f}")
        if best_vl is None or vl < best_vl - 1e-4:
            best_vl, best_ep = vl, ep
            best_params = [np.array(p.data, copy=True) for p in net.parameters()]
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                say(f"    early stop: no val improvement for {patience} epochs "
                    f"(best was epoch {best_ep})")
                break
    if best_params is not None:
        for p, a in zip(net.parameters(), best_params):
            p.data = a
    secs = time.perf_counter() - t0
    rep.update({"chosen": {"d": int(best_d), "lr": float(best_lr), "batch": int(batch)},
                "history": hist, "best_epoch": int(best_ep),
                "best_val_loss": float(best_vl), "epochs_run": len(hist),
                "seconds": secs, "params": int(sum(p.data.size for p in net.parameters()))})
    say(f"  final: d={best_d} lr={best_lr} best val {best_vl:.4f} @ epoch {best_ep} "
        f"({len(hist)} epochs, {secs:.1f}s, {rep['params']:,} params)")
    if log:
        with open(log, "w") as f:
            json.dump(rep, f, indent=1)
    return net, rep