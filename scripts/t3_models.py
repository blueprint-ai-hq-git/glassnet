"""t3_models.py — reviewer item (3), part 2: neural models on the redesign.

Dataset: tiny_shakespeare, vocabulary 1501 (1500 + <unk>), 8-token window.
Models: the readable (concept bottleneck) and the matched normal, at EQUAL
parameter counts (normal hidden == readable K).

Reported per run: validation ACCURACY and validation PERPLEXITY
(ppl = exp(mean CE)), plus best epoch, plateau length and wall time.
Perplexity is the honest metric here because <unk> is 29.64% of val targets.

Protocol: the item-(1) stopping rule (patience 15, max 96, no early stop before
epoch 30) — the rule that was shown to be necessary to avoid the early-stopping
artifact. Validation only; the test split is never read.

Environment:
  T3_K        comma list of K values (default "128")
  T3_SEEDS    comma list of seeds   (default "0,1,2")
  T3_MODELS   comma list            (default "readable,normal")
  T3_LR       learning rate         (default 0.1)
  T3_EPOCHS   max epochs            (default 96)
  T3_MIN_EPOCH min epochs before any early stop (default 30)
"""
import os
import sys
import json
import math
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

from glassnet import tiny_data
from glassnet.losses import cross_entropy
from glassnet.optim import SGD
from glassnet.tensor import relu, set_backend, backend_name
from run_t2_iter003 import D, WIN, BATCH, TinyLM3, embedding, _flatten
from train_t2_normal_concat import TinyNormal

# item (3) is a NEW comparison group, so it can run entirely on one backend.
# Default is CPU; T3_BACKEND=cupy switches every op to the GPU (verified: the
# gradient gate passes on GPU and CPU<->GPU parity is 100% argmax agreement).
BACKEND = os.environ.get("T3_BACKEND", "numpy")
set_backend(BACKEND)

VOCAB_CAP = 1500
RUNDIR = os.path.join(PROJ, "runs", "t3")
os.makedirs(RUNDIR, exist_ok=True)

K_LIST = [int(x) for x in os.environ.get("T3_K", "128").split(",")]
SEEDS = [int(x) for x in os.environ.get("T3_SEEDS", "0,1,2").split(",")]
MODELS = os.environ.get("T3_MODELS", "readable,normal").split(",")
LR = float(os.environ.get("T3_LR", "0.1"))
MAX_EPOCHS = int(os.environ.get("T3_EPOCHS", "96"))
MIN_EPOCH = int(os.environ.get("T3_MIN_EPOCH", "30"))
PATIENCE = int(os.environ.get("T3_PATIENCE", "15"))

TAG = os.environ.get("T3_TAG", f"K{K_LIST[0]}_lr{LR}")
LAM_A = float(os.environ.get("T2_LAM_A", "0.2"))
LAM_W = float(os.environ.get("T2_LAM_W", "1.0"))
LAM_CONCEPT = float(os.environ.get("T2_LAM_CONCEPT", "0.0"))
OUT = os.path.join(RUNDIR, f"models_{TAG}.json")
LOG = os.path.join(RUNDIR, "models.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def eval_full(fwd, X, y, chunk=4096):
    """Returns (accuracy %, mean CE) — chunked to avoid huge logits arrays.
    Backend-aware: cupy forbids implicit host conversion, so use .get()."""
    n = len(X)
    correct, ce_sum = 0, 0.0
    for i in range(0, n, chunk):
        xs, ys = X[i:i + chunk], y[i:i + chunk]
        z = fwd(xs).data
        if hasattr(z, "get"):          # cupy array -> host
            z = z.get()
        z = np.asarray(z)
        z = z - z.max(axis=1, keepdims=True)
        ex = np.exp(z)
        p = ex / ex.sum(axis=1, keepdims=True)
        correct += int((z.argmax(axis=1) == ys).sum())
        ce_sum += float(-np.log(p[np.arange(len(ys)), ys] + 1e-12).sum())
    return correct / n * 100, ce_sum / n


def make(kind, V, K, seed):
    if kind == "readable":
        return TinyLM3(V, K, D, WIN, rng=np.random.default_rng(seed))
    return TinyNormal(V, K, D, WIN, rng=np.random.default_rng(seed))


def fwd_of(kind, net):
    return (lambda X: net.forward(X)["logits"]) if kind == "readable" \
        else (lambda X: net.forward(X))


def n_params(V, K):
    return V * D + (D * WIN) * K + K + K * V + V


def train(kind, net, d, seed, baseline, save_path=None):
    rngd = np.random.default_rng(seed * 131 + 7)
    opt = SGD(net.parameters(), lr=LR)
    X, y = d["Xtr"], d["ytr"]
    fwd = fwd_of(kind, net)
    best = {"val": -1.0, "epoch": 0, "ce": None}
    best_w = None
    bad, hist, stopped_at = 0, [], MAX_EPOCHS
    for ep in range(MAX_EPOCHS):
        t0 = time.time()
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
        v, ce = eval_full(fwd, d["Xva"], d["yva"])
        hist.append(v)
        if v > best["val"]:
            best = {"val": v, "epoch": ep + 1, "ce": ce}
            # keep the BEST weights on the host, so the audit can run on the
            # exact model whose numbers are reported (no retraining, no drift).
            best_w = {p.name: (p.data.get() if hasattr(p.data, "get")
                               else np.array(p.data))
                      for p in net.parameters()}
            bad = 0
        else:
            bad += 1
        log(f"    {kind} K{net.K if kind=='readable' else net.h1.W.data.shape[1]} "
            f"s{seed} ep {ep+1:3d} val {v:6.2f}% ppl {math.exp(ce):8.2f} "
            f"(best {best['val']:.2f} @ {best['epoch']}, bad {bad}, "
            f"{time.time()-t0:.0f}s)")
        if bad >= PATIENCE and (ep + 1) >= MIN_EPOCH:
            stopped_at = ep + 1
            log(f"    -> early stop at epoch {stopped_at}")
            break
    plateau = 0
    for v in hist:
        if abs(v - baseline) < 1e-9:
            plateau += 1
        else:
            break
    if save_path is not None and best_w is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **best_w)
    return best, plateau, stopped_at


def main():
    d = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
    V = d["V"]
    UNK = V - 1
    baseline = float((d["yva"] == UNK).mean() * 100)

    log("=" * 78)
    log(f"ITEM (3) NEURAL MODELS — vocab_cap={VOCAB_CAP} V={V} win={WIN} "
        f"lr={LR} batch={BATCH}")
    log(f"backend = {backend_name()}")
    log(f"stopping: patience {PATIENCE}, max {MAX_EPOCHS}, no stop before {MIN_EPOCH}")
    log(f"majority (<unk>) baseline on val = {baseline:.2f}%")
    log(f"reference: unigram acc 29.64 ppl 119.60 | bigram acc 30.35 ppl 99.50")
    log(f"tag={TAG}")
    log("=" * 78)

    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    keys = {(r["model"], r["K"], r["seed"]) for r in done}

    for K in K_LIST:
        for kind in MODELS:
            for seed in SEEDS:
                if (kind, K, seed) in keys:
                    log(f"  [{kind} K={K} seed {seed}] already done — skipped")
                    continue
                log(f"  [{kind} K={K} seed {seed}] training "
                    f"({n_params(V, K):,} params)")
                t0 = time.time()
                net = make(kind, V, K, seed)
                ckpt = os.path.join(RUNDIR, "ckpt",
                                    f"{kind}_K{K}_lr{LR}"
                                    f"{'' if LAM_A == 0.2 else '_lam' + str(LAM_A).replace('.','')}"
                                    f"{'' if LAM_W == 1.0 else '_w' + str(LAM_W).replace('.','')}"
                                    f"{'' if LAM_CONCEPT == 0 else '_lc' + str(LAM_CONCEPT).replace('.','')}"
                                    # concept SET must be in the name, or v2 cluster
                                    # checkpoints silently overwrite v1 single-word ones
                                    f"{'' if os.environ.get('T2_CONCEPT_SET', 'words') == 'words' else '_' + os.environ.get('T2_CONCEPT_SET', 'words')}"
                                    # T3_CKPT_TAG keeps a new milestone's checkpoints from
                                    # overwriting an earlier milestone's at the SAME lr/lc/K
                                    # (Target B vs A3 would otherwise collide, and A3's
                                    # audit reads these files, so a collision would break
                                    # A3's reproducibility). Default empty = unchanged.
                                    f"{'' if not os.environ.get('T3_CKPT_TAG') else '_' + os.environ['T3_CKPT_TAG']}"
                                    f"_seed{seed}.npz")
                best, plateau, stopped = train(kind, net, d, seed, baseline,
                                               save_path=ckpt)
                rec = {"model": kind, "K": K, "seed": seed,
                       "concept_set": os.environ.get("T2_CONCEPT_SET", "words"),
                       "lam_concept": LAM_CONCEPT, "lam_act": LAM_A, "lam_w": LAM_W,
                       "params": n_params(V, K), "lr": LR,
                       "val": best["val"], "val_ppl": math.exp(best["ce"]),
                       "best_epoch": best["epoch"], "stopped_at": stopped,
                       "plateau_len": plateau,
                       "secs": round(time.time() - t0)}
                done.append(rec)
                json.dump(done, open(OUT, "w"))
                log(f"    -> {kind} K={K} seed {seed}: val {best['val']:.2f}% "
                    f"ppl {rec['val_ppl']:.2f} @ ep {best['epoch']}, "
                    f"plateau {plateau}, {rec['secs']}s")

    # summary
    log("")
    log("SUMMARY (mean ± std over seeds)")
    for K in K_LIST:
        for kind in MODELS:
            rs = [r for r in done if r["model"] == kind and r["K"] == K]
            if not rs:
                continue
            vs = [r["val"] for r in rs]
            ps = [r["val_ppl"] for r in rs]
            log(f"  {kind:9s} K={K:4d} ({rs[0]['params']:,} params): "
                f"acc {np.mean(vs):6.2f} ± {np.std(vs):4.2f}   "
                f"ppl {np.mean(ps):8.2f} ± {np.std(ps):6.2f}   (n={len(rs)})")
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
