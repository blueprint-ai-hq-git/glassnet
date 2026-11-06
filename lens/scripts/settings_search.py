#!/usr/bin/env python3
"""settings_search.py - find the BEST SETTING for every training method at every size, by measurement.

The user's order, 2026-10-19: "don't try to make a model first, first find the best setting for all the
training methods, all sizes of the model". And the score must be RULE LEARNING, not memorisation: "everything
is learned not memorised".

WHAT IT SWEEPS
  methods : sft, rl  (distill is scored on the real corpus separately because it needs a trained teacher)
  sizes   : d = 32, 64, 128, 256, 512 (about 2k to 7M parameters)
  settings: learning rate, epochs, and for rl the number of sampled answers

HOW IT SCORES (this is the part that matters)
  PRIMARY    extrapolation accuracy on arithmetic pairs outside the training range (10..19). A model that
             memorised a table cannot answer these, so this is the honest measure of "learned the rule".
  SECONDARY  held-out accuracy on unseen pairs from the SAME range.
  PENALTY    the memorisation gap (train - held out). A large gap with high train accuracy is memorisation.

Every trial writes a row to lens/runs/SETTINGS-SEARCH.csv, and the best setting per (method, size) is written
to lens/recipes_search.json, which is what easy mode will read. Nothing is chosen by taste: if a trial is
worse, its number says so and it loses.

Usage: python3 lens/scripts/settings_search.py [--sizes 32 64 128] [--methods sft rl] [--quick]
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

import lens  # noqa: E402
import lens.methods as M  # noqa: E402
import lens.opt as opt_mod  # noqa: E402
import rules_test as RT  # noqa: E402

CSV = "lens/runs/SETTINGS-SEARCH.csv"
OUT = "lens/recipes_search.json"


def bench_data(seed=0, window=12):
    pairs = [(x, y) for x in range(10) for y in range(10)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    held = [pairs[i] for i in perm[:20]]
    train = [pairs[i] for i in perm[20:]]
    extra = [(x, y) for x in range(10, 20) for y in range(10, 20)]
    vocab = RT.build_vocab([RT.sentence(x, y) for x, y in pairs + extra])
    vmap = {w: i for i, w in enumerate(vocab)}
    Xtr, Ytr = RT.make_set(train, vocab, window)
    Xhe, Yhe = RT.make_set(held, vocab, window)
    Xex, Yex = RT.make_set(extra, vocab, window)
    return vocab, vmap, (Xtr, Ytr), (Xhe, Yhe), (Xex, Yex), train


def trial(method, d, lr, epochs, samples, seed=0, window=12):
    """One measured trial. Returns a row dict; every number comes from this run."""
    vocab, vmap, (Xtr, Ytr), (Xhe, Yhe), (Xex, Yex), train = bench_data(seed, window)
    net = lens.Net(len(vocab), RT.concepts_for(vocab), d=d, window=window, seed=seed, device="cpu")
    params = sum(p.data.size for p in net.parameters())
    rng = np.random.default_rng(seed)
    opt = opt_mod.RelativeGrad(net.parameters(), lr=lr, mode="relgrad")
    t0 = time.time()
    bs = min(32, len(Xtr))
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = order[i:i + bs]
            for p in net.parameters():
                p.grad = None
            out = net.loss(Xtr[idx], Ytr[idx], use_concept_loss=True)
            (out[0] if isinstance(out, tuple) else out).backward()
            opt.step()
    acc_tr = RT.accuracy(net, Xtr, Ytr)
    acc_he = RT.accuracy(net, Xhe, Yhe)
    acc_ex = RT.accuracy(net, Xex, Yex)
    rl_stats = None
    if method == "rl":
        prompts = [(f"human : what is {x} + {y} ? assistant :", str(x + y)) for x, y in train]
        rl_stats = M.rl_verifiable(net, vocab, vmap, prompts, M.verify_sum, samples=samples, rounds=3, lr=lr)
        acc_tr = RT.accuracy(net, Xtr, Ytr)
        acc_he = RT.accuracy(net, Xhe, Yhe)
        acc_ex = RT.accuracy(net, Xex, Yex)
    gap = acc_tr - acc_he
    return {"method": method, "d": d, "lr": lr, "epochs": epochs, "samples": samples,
            "params": params, "train_acc": round(acc_tr, 4), "heldout_acc": round(acc_he, 4),
            "extrapolation_acc": round(acc_ex, 4), "gap": round(gap, 4),
            "reward_rate": (round(float(np.mean(rl_stats["reward_rate"])), 3) if rl_stats else None),
            "seconds": round(time.time() - t0, 1),
            "score": round(acc_ex * 1.0 + acc_he * 0.5 - max(gap, 0) * 0.5, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--methods", nargs="+", default=["sft", "rl"])
    ap.add_argument("--quick", action="store_true", help="smaller grid, for a first pass")
    a = ap.parse_args()

    lrs = [0.02, 0.1] if a.quick else [0.01, 0.02, 0.05, 0.1, 0.4]
    epochs_list = [200] if a.quick else [100, 300, 800]
    samples_list = [8] if a.quick else [2, 8, 32]

    os.makedirs("lens/runs", exist_ok=True)
    new = not os.path.exists(CSV)
    f = open(CSV, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["method", "d", "lr", "epochs", "samples", "params", "train_acc",
                                      "heldout_acc", "extrapolation_acc", "gap", "reward_rate", "seconds",
                                      "score"])
    if new:
        w.writeheader()
    print(f"SETTINGS SEARCH - scoring by RULE LEARNING (extrapolation) with a memorisation penalty")
    print(f"  methods {a.methods}  sizes {a.sizes}  lrs {lrs}  epochs {epochs_list}  "
          f"rl samples {samples_list}")
    print(f"  {'method':6s} {'d':>4s} {'lr':>5s} {'ep':>4s} {'smp':>4s} {'train':>6s} {'held':>6s} "
          f"{'extra':>6s} {'gap':>6s} {'score':>7s}")
    best = {}
    for method in a.methods:
        for d, lr, ep in itertools.product(a.sizes, lrs, epochs_list):
            smp = samples_list if method == "rl" else [0]
            for s in smp:
                row = trial(method, d, lr, ep, s)
                w.writerow(row); f.flush()
                print(f"  {row['method']:6s} {row['d']:>4d} {row['lr']:>5.2f} {row['epochs']:>4d} "
                      f"{row['samples']:>4d} {row['train_acc']:>6.2f} {row['heldout_acc']:>6.2f} "
                      f"{row['extrapolation_acc']:>6.2f} {row['gap']:>6.2f} {row['score']:>7.3f}")
                key = (method, d)
                if key not in best or row["score"] > best[key]["score"]:
                    best[key] = row
    f.close()
    out = {f"{m}|{d}": r for (m, d), r in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1]))}
    json.dump({"note": "best setting per (method, size) measured on the arithmetic rule bench; score = "
                       "extrapolation + 0.5*held-out - 0.5*memorisation gap",
               "best": out}, open(OUT, "w"), indent=1)
    print(f"\n  best per (method, size) written to {OUT}:")
    for k, r in out.items():
        print(f"    {k:10s} lr {r['lr']:<5} epochs {r['epochs']:<4} samples {r['samples']:<3} "
              f"extra {r['extrapolation_acc']*100:.0f}%  held {r['heldout_acc']*100:.0f}%  "
              f"gap {r['gap']*100:.0f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())