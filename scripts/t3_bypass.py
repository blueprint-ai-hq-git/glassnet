"""t3_bypass.py — step 3: does anything bypass the concept layer?

If the answer layer can produce good logits WITHOUT the concepts, then the
concepts are decoration no matter what their AUROC says. This measures it
directly instead of reasoning about it.

Measurements (all on VALIDATION, one trained readable):
  A. the architecture path, read off the code (is there a second route?)
  B. **h=0 ablation** — force every concept activation to 0, so the logits are
     just the answer bias. Accuracy here vs the majority baseline is the
     decisive number: if it stays near the full-model accuracy, the concepts
     are not carrying the prediction.
  C. bias-vs-concepts logit magnitude: mean |answer.b| against mean |W @ h|.
  D. concept activity: mean activation, fraction dead (h<0.05), fraction
     saturated (h>0.95) — the sparsity penalty pushes activations down, and a
     concept pinned off cannot contribute.
  E. contribution concentration: how many concepts carry 90% of the total
     |W @ h| mass (a few concepts carrying everything = the rest are decoration).
  F. per-concept contribution to the CORRECT class logit, and how many concepts
     have a negative mean effect.

Env: BYP_K (64), BYP_LR (0.3), BYP_SEED (0), BYP_BACKEND (cupy)
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
from glassnet.optim import SGD
from glassnet.tensor import set_backend, backend_name
from run_t2_iter003 import TinyLM3, D, WIN, BATCH

VOCAB_CAP = 1500
K = int(os.environ.get("BYP_K", "64"))
LR = float(os.environ.get("BYP_LR", "0.3"))
SEED = int(os.environ.get("BYP_SEED", "0"))
EPOCHS = int(os.environ.get("BYP_EPOCHS", "96"))
MIN_EPOCH = int(os.environ.get("BYP_MIN_EPOCH", "30"))
PATIENCE = int(os.environ.get("BYP_PATIENCE", "15"))
set_backend(os.environ.get("BYP_BACKEND", "cupy"))

RUNDIR = os.path.join(PROJ, "runs", "t3")
os.makedirs(RUNDIR, exist_ok=True)
OUT = os.path.join(RUNDIR, f"bypass_K{K}.json")
LOG = os.path.join(RUNDIR, "bypass.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def eval_val(net, d, chunk=4096):
    X, y = d["Xva"], d["yva"]
    correct, ce = 0, 0.0
    for i in range(0, len(X), chunk):
        z = host(net.forward(X[i:i + chunk])["logits"].data)
        z = z - z.max(axis=1, keepdims=True)
        ex = np.exp(z); p = ex / ex.sum(axis=1, keepdims=True)
        correct += int((z.argmax(axis=1) == y[i:i + chunk]).sum())
        ce += float(-np.log(p[np.arange(len(y[i:i+chunk])), y[i:i+chunk]] + 1e-12).sum())
    return correct / len(X) * 100, ce / len(X)


def acc_from_logits(Z, y):
    return float((Z.argmax(axis=1) == y).mean() * 100)


def main():
    d = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
    V = d["V"]; UNK = V - 1
    baseline = float((d["yva"] == UNK).mean() * 100)

    log("=" * 78)
    log(f"STEP 3 — BYPASS CHECK  (readable K={K} lr={LR} seed {SEED}, backend {backend_name()})")
    log("=" * 78)

    # ---- A. the architecture path, read off the code ----
    log("A. ARCHITECTURE PATH (TinyLM3.forward, read off the source):")
    log("     emb -> embedding(ids) -> flatten -> concept(Dense) -> sigmoid(h)")
    log("         -> answer(Dense) -> logits")
    log("   The ONLY route from input to logits is through h. There is no skip")
    log("   connection, no residual, no direct input->logit term. Structurally")
    log("   no bypass exists. The risk is therefore not a second path but the")
    log("   concepts being SUPPRESSED or IGNORED. Measured below.")

    # ---- train one readable ----
    net = TinyLM3(V, K, D, WIN, rng=np.random.default_rng(SEED))
    opt = SGD(net.parameters(), lr=LR)
    X, y = d["Xtr"], d["ytr"]
    rngd = np.random.default_rng(SEED * 131 + 7)
    best = {"val": -1.0, "epoch": 0}; bad = 0
    for ep in range(EPOCHS):
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss, _ = net.loss(X[idx], y[idx])
            loss.backward()
            opt.step()
        v, ce = eval_val(net, d)
        if v > best["val"]:
            best = {"val": v, "epoch": ep + 1, "ce": ce}; bad = 0
        else:
            bad += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"   train seed{SEED} ep {ep+1:3d} val {v:6.2f}% ppl {np.exp(ce):7.2f} "
                f"(best {best['val']:.2f} @ {best['epoch']})")
        if bad >= PATIENCE and (ep + 1) >= MIN_EPOCH:
            break
    log(f"   trained: val {best['val']:.2f}% ppl {np.exp(best['ce']):.2f} @ ep {best['epoch']}")

    # ---- B/C/D/E/F on the validation split ----
    set_backend("numpy")
    for p in net.parameters():
        p.data = host(p.data)

    Xva, yva = d["Xva"], d["yva"]
    h = np.asarray(net.forward(Xva)["concepts"].data)     # (N, K)
    W = np.asarray(net.answer.W.data)                      # (K, V)
    b = np.asarray(net.answer.b.data)                      # (V,)
    Z_full = h @ W + b
    acc_full = acc_from_logits(Z_full, yva)

    # B. h = 0 -> logits are exactly the bias
    Z_h0 = np.zeros_like(Z_full) + b
    acc_h0 = acc_from_logits(Z_h0, yva)
    # also: h = 1 (all concepts saturated on)
    Z_h1 = np.ones_like(h) @ W + b
    acc_h1 = acc_from_logits(Z_h1, yva)
    # and: concept contribution only, bias removed
    Z_nb = h @ W
    acc_nb = acc_from_logits(Z_nb, yva)

    log("")
    log("B. ABLATIONS (validation)")
    log(f"     full model                     acc {acc_full:6.2f}%")
    log(f"     h=0  (logits = answer bias)    acc {acc_h0:6.2f}%   <- decisive")
    log(f"     h=1  (all concepts saturated)  acc {acc_h1:6.2f}%")
    log(f"     bias removed (concepts only)   acc {acc_nb:6.2f}%")
    log(f"     majority (<unk>) baseline      acc {baseline:6.2f}%")
    log(f"   => concepts carry {acc_full - acc_h0:+.2f} pts over the bias alone;")
    log(f"      the bias alone gets {acc_h0 - baseline:+.2f} pts over the majority class.")

    # C. magnitude split
    mag_b = float(np.abs(b).mean())
    mag_c = float(np.abs(h @ W).mean())
    log("")
    log("C. LOGIT MAGNITUDE SPLIT")
    log(f"     mean |answer.b|        {mag_b:.4f}")
    log(f"     mean |W @ h|           {mag_c:.4f}")
    log(f"     concepts/bias ratio    {mag_c / (mag_b + 1e-12):.3f}")

    # D. activity
    log("")
    log("D. CONCEPT ACTIVITY (validation)")
    log(f"     mean activation        {h.mean():.4f}")
    log(f"     dead  (h < 0.05)       {float((h < 0.05).mean())*100:6.2f}% of neuron-windows")
    log(f"     saturated (h > 0.95)   {float((h > 0.95).mean())*100:6.2f}%")
    per = h.mean(axis=0)
    log(f"     per-neuron mean act: min {per.min():.4f} median {np.median(per):.4f} max {per.max():.4f}")
    log(f"     neurons effectively off (per-neuron mean < 0.05): "
        f"{int((per < 0.05).sum())}/{K}")

    # E. concentration of |W@h| mass
    # per-neuron total |contribution| = (sum_n |h[n,k]|) * (sum_v |W[k,v]|).
    # The naive (N,K,V) intermediate is ~7.8 GB at K=64 -- this is the same
    # number computed in O(K*V) memory.
    per_neuron = np.abs(h).sum(axis=0) * np.abs(W).sum(axis=1)      # (K,)
    order = np.argsort(-per_neuron)
    cum = np.cumsum(per_neuron[order]) / (per_neuron.sum() + 1e-12)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n50 = int(np.searchsorted(cum, 0.50) + 1)
    log("")
    log("E. CONTRIBUTION CONCENTRATION (|W @ h| mass)")
    log(f"     concepts carrying 50% of the mass: {n50}/{K}")
    log(f"     concepts carrying 90% of the mass: {n90}/{K}")

    # F. per-concept effect on the CORRECT class logit:
    # eff[n,k] = h[n,k] * W[k, y[n]]  ->  (K, N), cheap.
    Wy = W[:, yva]                       # (K, N)
    eff = h.T * Wy                       # (K, N)
    mean_eff = eff.mean(axis=1)          # (K,)
    log("")
    log("F. PER-CONCEPT MEAN EFFECT ON THE CORRECT-CLASS LOGIT")
    log(f"     positive effect: {int((mean_eff > 0).sum())}/{K} neurons")
    log(f"     negative effect: {int((mean_eff < 0).sum())}/{K} neurons")
    log(f"     mean effect: min {mean_eff.min():+.4f} median {np.median(mean_eff):+.4f} "
        f"max {mean_eff.max():+.4f}")
    log(f"     neurons with |mean effect| < 0.01 (negligible): "
        f"{int((np.abs(mean_eff) < 0.01).sum())}/{K}")

    res = {"K": K, "lr": LR, "seed": SEED, "backend": backend_name(),
           "val": best["val"], "val_ppl": float(np.exp(best["ce"])),
           "baseline": baseline,
           "acc_full": acc_full, "acc_h0": acc_h0, "acc_h1": acc_h1, "acc_nb": acc_nb,
           "mag_bias": mag_b, "mag_concept": mag_c,
           "mean_act": float(h.mean()),
           "frac_dead": float((h < 0.05).mean()),
           "frac_sat": float((h > 0.95).mean()),
           "neurons_off": int((per < 0.05).sum()),
           "n_for_50pct": n50, "n_for_90pct": n90,
           "n_positive_effect": int((mean_eff > 0).sum()),
           "n_negligible_effect": int((np.abs(mean_eff) < 0.01).sum()),
           "concept_gain_over_bias": acc_full - acc_h0,
           "bias_gain_over_majority": acc_h0 - baseline}
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
