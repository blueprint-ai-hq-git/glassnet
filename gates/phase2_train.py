"""PHASE 2 — main run: concept/rule/answer architecture on Fashion-MNIST.

Manual hyperparameters (Phase 3 automates them). Same schedule as the
Phase 1 baseline (10 epochs, batch 128, lr 0.1) so the ONLY change is the
architecture. Joint loss: class CE + alpha*concept BCE + lam*L1 on rule
connections. Weak connections pruned afterwards (top-`keep` per rule).

Run:  python3 phase2_train.py --seeds 0 1 2
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.tensor import Tensor  # noqa: E402
from glassnet.optim import SGD  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402
from glassnet.concepts import load as load_concepts  # noqa: E402
from glassnet.eval import accuracy, auroc  # noqa: E402
from glassnet.gates import gate1  # noqa: E402
from glassnet.phase2_model import GlassNetPhase2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase2")
DATA_DIR = os.path.join(HERE, "data")


def train_seed(seed, args, Xtr, ytr, Ctr, Xte, yte, Cte, outdir, watcher=None):
    """Train one seed. watcher(info: dict) is called every 100 batches —
    that's where Phase 3's auto-pilot plugs in."""
    net = GlassNetPhase2(K=args.K, R=args.R, rng=np.random.default_rng(seed))
    opt = SGD(net.parameters(), lr=args.lr)
    rng_data = np.random.default_rng(seed * 777 + 1)
    n = Xtr.shape[0]
    rows = []
    step = 0
    best_test = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng_data.permutation(n)
        tot_loss, tot_ce, tot_cb, tot_l1, tot_correct, nb = 0.0, 0.0, 0.0, 0.0, 0, 0
        for i in range(0, n, args.batch):
            idx = order[i:i + args.batch]
            xb, yb, cb_t = Xtr[idx], ytr[idx], Ctr[idx]
            xb_t = Tensor(xb, name="batch")
            opt.zero_grad()
            f = net.forward(xb_t)
            total, ce, cb, l1 = net.loss(xb_t, yb, cb_t, alpha=args.alpha,
                                         lam=args.lam, return_parts=True, f=f)
            total.backward()
            opt.step()
            tot_loss += float(total.data)
            tot_ce += float(ce.data)
            tot_cb += float(cb.data)
            tot_l1 += float(l1.data)
            tot_correct += int(np.sum(f["logits"].data.argmax(1) == yb))
            nb += 1
            step += 1
            if watcher and step % 100 == 0:
                watcher({"step": step, "epoch": epoch, "net": net,
                         "train_loss": tot_loss / nb,
                         "train_acc": tot_correct / (nb * args.batch)})

        train_loss = tot_loss / nb
        train_acc = tot_correct / n
        test_acc = accuracy(net, Xte, yte)
        best_test = max(best_test, test_acc)
        # concept fidelity at epoch end (mean AUROC over the 14 named neurons)
        acts = net.predict(Xte)["concepts"].data
        aurocs = [auroc(acts[:, k], Cte[:, k] > 0.5) for k in range(args.K)]
        sparsity = float((np.abs(net.rules.W.data) < 1e-3).mean())
        secs = time.time() - t0
        rows.append({"epoch": epoch, "train_loss": train_loss,
                     "train_acc": train_acc, "test_acc": test_acc,
                     "concept_auroc": float(np.mean(aurocs)),
                     "sparsity": sparsity, "seconds": secs})
        print(f"seed {seed} ep {epoch:2d} | loss {train_loss:.4f} "
              f"| train {train_acc*100:5.2f}% | test {test_acc*100:5.2f}% "
              f"| cAUROC {np.mean(aurocs):.3f} | sparse {sparsity:.2f} | {secs:.1f}s",
              flush=True)
        net.save(os.path.join(outdir, f"ckpt_seed{seed}_epoch{epoch}.npz"),
                 epoch=epoch)

    pre_prune_acc = accuracy(net, Xte, yte)
    stats_before = net.connection_stats()
    net.prune_rule_connections(keep=args.keep)
    post_prune_acc = accuracy(net, Xte, yte)
    stats_after = net.connection_stats()

    with open(os.path.join(outdir, f"seed{seed}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    info = {"seed": seed, "pre_prune_acc": pre_prune_acc,
            "post_prune_acc": post_prune_acc, "best_test": best_test,
            "final_epoch_auroc": rows[-1]["concept_auroc"],
            "conns_avg": stats_after["avg"], "conns_min": stats_after["min"],
            "conns_max": stats_after["max"],
            "dead_rules": stats_after["dead_rules"],
            "conns_before_avg": stats_before["avg"]}
    with open(os.path.join(outdir, f"final_seed{seed}.json"), "w") as f:
        json.dump(info, f, indent=2)
    net.save(os.path.join(outdir, f"final_seed{seed}.npz"),
             pre_prune_acc=np.float64(pre_prune_acc),
             post_prune_acc=np.float64(post_prune_acc))
    print(f"seed {seed} | pre-prune {pre_prune_acc*100:.2f}% -> "
          f"post-prune {post_prune_acc*100:.2f}% | avg conns {stats_after['avg']:.2f} "
          f"| dead rules {stats_after['dead_rules']}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=14)
    ap.add_argument("--R", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.005)
    ap.add_argument("--keep", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    print("=" * 66)
    print("GATE 1 first (no training on broken gradients)")
    print("=" * 66)
    gate1()

    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR)
    Ctr, Cte, lo, hi = load_concepts(DATA_DIR)
    print(f"[data] train {Xtr.shape} test {Xte.shape} concepts {Ctr.shape}")

    cfg = vars(args)
    with open(os.path.join(RUNS, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    info = {}
    t0 = time.time()
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        info[seed] = train_seed(seed, args, Xtr, ytr, Ctr, Xte, yte, Cte, RUNS)
    total = time.time() - t0

    # ---- report ----------------------------------------------------------
    pre = [info[s]["pre_prune_acc"] for s in args.seeds]
    post = [info[s]["post_prune_acc"] for s in args.seeds]
    conns = [info[s]["conns_avg"] for s in args.seeds]
    dead = [info[s]["dead_rules"] for s in args.seeds]
    au = [info[s]["final_epoch_auroc"] for s in args.seeds]
    lines = [
        "# PHASE 2 — readable architecture results",
        "",
        f"- arch: 784 -> {args.K} named concepts (sigmoid) -> {args.R} rules "
        f"(sigmoid, L1 + prune top-{args.keep}) -> 10 classes",
        f"- joint loss: CE + {args.alpha}*conceptBCE + {args.lam}*L1",
        f"- same schedule as baseline: {args.epochs} epochs, batch {args.batch}, "
        f"lr {args.lr}, seeds {args.seeds}",
        "",
        "| seed | test pre-prune | test post-prune | avg conns/rule | dead rules | concept AUROC |",
        "|------|---------------:|----------------:|---------------:|-----------:|--------------:|",
    ]
    for s in args.seeds:
        lines.append(f"| {s} | {info[s]['pre_prune_acc']*100:.2f}% | "
                     f"{info[s]['post_prune_acc']*100:.2f}% | {info[s]['conns_avg']:.2f} | "
                     f"{info[s]['dead_rules']} | {info[s]['final_epoch_auroc']:.3f} |")
    pmean, pstd = float(np.mean(pre)), float(np.std(pre))
    omean, ostd = float(np.mean(post)), float(np.std(post))
    lines += [
        "",
        f"**READABLE-MODEL ACC = post-prune test acc, mean ± std: "
        f"{omean*100:.2f}% ± {ostd*100:.2f}%**",
        f"(pre-prune: {pmean*100:.2f}% ± {pstd*100:.2f}%)",
        f"avg connections per rule: {float(np.mean(conns)):.2f} (target 2-3)",
        f"dead rules: {int(np.mean(dead))} of {args.R}",
        f"mean concept fidelity (AUROC, held-out): {float(np.mean(au)):.3f}",
        "",
    ]
    # compare with Phase 1 baseline
    bl_path = os.path.join(HERE, "runs", "phase1", "baseline_summary.json")
    if os.path.exists(bl_path):
        with open(bl_path) as f:
            bl = json.load(f)
        bl_finals = {int(k): v for k, v in bl["per_seed_final"].items()}
        diffs = [(info[s]["post_prune_acc"] - bl_finals[s]) * 100 for s in args.seeds]
        lines += [
            "vs the Phase 1 baseline (86.69% ± 0.59%, same seeds, same schedule):",
        ]
        for s, d in zip(args.seeds, diffs):
            lines.append(f"  seed {s}: {info[s]['post_prune_acc']*100:.2f}% vs "
                         f"{bl_finals[s]*100:.2f}% baseline -> {d:+.2f} pts")
        lines.append(f"  mean delta: {float(np.mean(diffs)):+.2f} pts "
                     f"(mean of {len(diffs)} seeds)")
    lines.append("")
    lines.append(f"command: python3 phase2_train.py --seeds {' '.join(map(str, args.seeds))}"
                 f" --epochs {args.epochs} --K {args.K} --R {args.R} --alpha {args.alpha}"
                 f" --lam {args.lam} --keep {args.keep}")
    rep = os.path.join(RUNS, "PHASE2-BASELINE.md")
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {rep}")
    print(f"TOTAL {total:.0f}s | readable-arch acc: {omean*100:.2f}% ± {ostd*100:.2f}% "
          f"(pre-prune {pmean*100:.2f}% ± {pstd*100:.2f}%) | conns/rule {float(np.mean(conns)):.2f}")


if __name__ == "__main__":
    main()