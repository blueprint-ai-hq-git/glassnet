"""PHASE 2 — ablations: remove one piece of the architecture at a time and
show what changes. Same schedule everywhere (10 epochs, batch 128, lr 0.1).

A0 full model            (concept supervisor + L1 + prune)
A1 no concept supervision (alpha=0)          -> do the names still hold?
A2 no sparsity (lam=0, no prune)             -> avg connections per rule
A3 no rule layer (concepts -> classes)       -> what do rules actually buy?
A4 prune impact                              -> pre vs post prune (from A0)

Run:  python3 phase2_ablations.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.tensor import Tensor  # noqa: E402
from glassnet.optim import SGD  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402
from glassnet.concepts import load as load_concepts  # noqa: E402
from glassnet.eval import accuracy, auroc  # noqa: E402
from glassnet.gates import gate1  # noqa: E402
from glassnet.phase2_model import GlassNetPhase2, NoRulesModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase2")
DATA_DIR = os.path.join(HERE, "data")


def fit(net, Xtr, ytr, Ctr, Xte, yte, Cte, epochs=10, batch=128, lr=0.1,
        alpha=0.5, lam=0.02, prune_keep=None, seed=0, extra_loss_kwargs=None):
    opt = SGD(net.parameters(), lr=lr)
    rng_data = np.random.default_rng(seed * 777 + 1)
    n = Xtr.shape[0]
    for epoch in range(epochs):
        order = rng_data.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb_t = Tensor(Xtr[idx], name="batch")
            opt.zero_grad()
            f = net.forward(xb_t)
            if isinstance(net, NoRulesModel):
                total = net.loss(xb_t, ytr[idx], Ctr[idx], alpha=alpha, f=f)
            else:
                total = net.loss(xb_t, ytr[idx], Ctr[idx], alpha=alpha,
                                 lam=lam, f=f, **(extra_loss_kwargs or {}))
            total.backward()
            opt.step()
    acc = accuracy(net, Xte, yte)
    if prune_keep:
        net.prune_rule_connections(keep=prune_keep)
        post = accuracy(net, Xte, yte)
    else:
        post = acc
    # concept fidelity
    acts = net.predict(Xte)["concepts"].data
    au = float(np.mean([auroc(acts[:, k], Cte[:, k] > 0.5) for k in range(acts.shape[1])]))
    return {"acc_pre": acc, "acc_post": post, "concept_auroc": au, "net": net}


def mean_std(vals):
    return f"{float(np.mean(vals))*100:.2f}% ± {float(np.std(vals))*100:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    print("=" * 66)
    print("GATE 1 first")
    print("=" * 66)
    gate1()
    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR)
    Ctr, Cte, lo, hi = load_concepts(DATA_DIR)

    variants = {
        "A0 full (supervisor+L1+prune)": dict(alpha=1.0, lam=0.005, prune_keep=3),
        "A1 no concept supervision":     dict(alpha=0.0, lam=0.005, prune_keep=3),
        "A2 no sparsity (no L1, no prune)": dict(alpha=1.0, lam=0.0, prune_keep=None),
    }
    results = {}
    for name, kw in variants.items():
        print(f"\n=== {name} ===", flush=True)
        per_seed = []
        for s in args.seeds:
            net = GlassNetPhase2(K=14, R=30, rng=np.random.default_rng(s))
            r = fit(net, Xtr, ytr, Ctr, Xte, yte, Cte, epochs=args.epochs,
                    prune_keep=kw["prune_keep"], seed=s, alpha=kw["alpha"],
                    lam=kw["lam"])
            conns = (net.connection_stats()["avg"]
                     if kw["prune_keep"] is None else net.connection_stats()["avg"])
            per_seed.append({"acc": r["acc_post"], "auroc": r["concept_auroc"],
                             "conns": conns})
            print(f"  seed {s}: test {r['acc_post']*100:.2f}% "
                  f"| conceptAUROC {r['concept_auroc']:.3f} | conns/rule {conns:.2f}",
                  flush=True)
        results[name] = per_seed

    # A3: rule layer removed
    print("\n=== A3 no rule layer (concepts -> classes directly) ===", flush=True)
    a3 = []
    for s in args.seeds:
        net = NoRulesModel(K=14, rng=np.random.default_rng(s))
        r = fit(net, Xtr, ytr, Ctr, Xte, yte, Cte, epochs=args.epochs, seed=s,
                alpha=1.0)
        a3.append({"acc": r["acc_post"], "auroc": r["concept_auroc"]})
        print(f"  seed {s}: test {r['acc_post']*100:.2f}% | conceptAUROC {r['concept_auroc']:.3f}",
              flush=True)
    results["A3 no rule layer"] = a3

    lines = [
        "# PHASE 2 — ablations (remove a layer, show what changes)",
        "",
        f"same schedule everywhere: {args.epochs} epochs, batch 128, lr 0.1, "
        f"seeds {args.seeds}",
        "",
        "| variant | test acc (post-prune) mean±std | concept AUROC mean±std |"
        " avg conns/rule |",
        "|---|---:|---:|---:|",
    ]
    for name in ["A0 full (supervisor+L1+prune)",
                 "A1 no concept supervision",
                 "A2 no sparsity (no L1, no prune)",
                 "A3 no rule layer"]:
        ps = results[name]
        au = mean_std([p["auroc"] for p in ps])
        conns = (f"{float(np.mean([p['conns'] for p in ps])):.2f}"
                 if "conns" in ps[0] else "-")
        lines.append(f"| {name} | {mean_std([p['acc'] for p in ps])} | {au} | {conns} |")
    lines += ["",
              "A4 (prune impact) is inside the A0 numbers: acc_pre vs acc_post "
              "printed below.", ""]

    # A0 pre vs post detail (reuse the full-model variants loop result)
    ps = results["A0 full (supervisor+L1+prune)"]
    lines.append("A0 pre vs post prune (per seed):")
    a0 = []
    for s in args.seeds:
        net = GlassNetPhase2(K=14, R=30, rng=np.random.default_rng(s))
        r = fit(net, Xtr, ytr, Ctr, Xte, yte, Cte, epochs=args.epochs,
                prune_keep=3, seed=s, alpha=1.0, lam=0.005)
        a0.append((r["acc_pre"], r["acc_post"]))
        lines.append(f"  seed {s}: {r['acc_pre']*100:.2f}% -> {r['acc_post']*100:.2f}% "
                     f"({(r['acc_post']-r['acc_pre'])*100:+.2f} pts)")
    lines.append("")
    rep = os.path.join(RUNS, "ablations.md")
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n[wrote] {rep}")


if __name__ == "__main__":
    main()