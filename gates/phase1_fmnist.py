"""TEST GATE 3 — Fashion-MNIST BASELINE SCORE TO BEAT.

Run:  python3 phase1_fmnist.py --seeds 0 1 2 --epochs 10 --hidden 128 \
                               --lr 0.1 --batch 128

What happens here (plain English):
- A plain 3-layer net (784 pixels -> 128 hidden ReLU units -> 10 classes),
  nothing special — that is the point. Phase 2's readable architecture must
  beat THIS number, so the baseline must be plain and honest.
- Plain SGD with ONE manual learning rate. No momentum, no tricks.
- 3 random seeds, everything else identical -> mean ± std (a single seed
  would be a coin flip).
- Every epoch: train loss, train accuracy, test accuracy, checkpoint, and
  a row in a CSV. Nothing is computed afterwards from memory — it is all
  on disk in runs/phase1/.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.tensor import Tensor, relu  # noqa: E402
from glassnet.layers import Dense, Sequential  # noqa: E402
from glassnet.losses import cross_entropy  # noqa: E402
from glassnet.optim import SGD  # noqa: E402
from glassnet.data import load as load_fmnist  # noqa: E402
from glassnet.gates import gate1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase1")
DATA_DIR = os.path.join(HERE, "data")


def accuracy(net, X, y, batch=512):
    correct = 0
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch]
        pred = net(Tensor(xb, name="test_batch")).data
        correct += int(np.sum(pred.argmax(axis=1) == y[i:i + batch]))
    return correct / X.shape[0]


def run_seed(seed, Xtr, ytr, Xte, yte, args):
    rng_data = np.random.default_rng(seed * 1000 + 7)   # shuffle stream
    rng_model = np.random.default_rng(seed)             # init stream
    net = Sequential(
        Dense(784, args.hidden, rng=rng_model, name="l1"), relu,
        Dense(args.hidden, 10, rng=rng_model, name="l2"),
    )
    opt = SGD(net.parameters(), lr=args.lr)
    n = Xtr.shape[0]
    rows = []
    best_test = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng_data.permutation(n)
        tot_loss, tot_correct, nb = 0.0, 0, 0
        for i in range(0, n, args.batch):
            idx = order[i:i + args.batch]
            xb, yb = Xtr[idx], ytr[idx]
            xb_t = Tensor(xb, name="train_batch")
            opt.zero_grad()
            out = net(xb_t)
            loss = cross_entropy(out, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.data)
            tot_correct += int(np.sum(out.data.argmax(axis=1) == yb))
            nb += 1
        train_loss = tot_loss / nb
        train_acc = tot_correct / n
        test_acc = accuracy(net, Xte, yte)
        best_test = max(best_test, test_acc)
        secs = time.time() - t0
        rows.append({"epoch": epoch, "train_loss": train_loss,
                     "train_acc": train_acc, "test_acc": test_acc,
                     "best_test_acc": best_test, "seconds": secs})
        print(f"seed {seed} epoch {epoch:2d} | train_loss {train_loss:.4f} "
              f"| train_acc {train_acc * 100:5.2f}% | test_acc {test_acc * 100:5.2f}% "
              f"| best {best_test * 100:5.2f}% | {secs:.1f}s", flush=True)
        np.savez(os.path.join(RUNS, f"ckpt_seed{seed}_epoch{epoch}.npz"),
                 **{k: v.data for k, v in net.named_parameters().items()},
                 epoch=epoch)

    with open(os.path.join(RUNS, f"baseline_seed{seed}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RUNS, f"baseline_seed{seed}.json"), "w") as f:
        json.dump({"seed": seed, "final_test_acc": rows[-1]["test_acc"],
                   "best_test_acc": best_test,
                   "final_train_loss": rows[-1]["train_loss"]}, f, indent=2)

    # loss curve (best-effort; matplotlib is optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows],
                 "o-", label="train loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot([r["epoch"] for r in rows], [r["test_acc"] * 100 for r in rows],
                 "s-", color="tab:green", label="test acc %")
        ax2.set_ylabel("test acc %", color="tab:green")
        ax1.set_title(f"Fashion-MNIST baseline, seed {seed}")
        fig.tight_layout()
        fig.savefig(os.path.join(RUNS, f"curve_seed{seed}.png"), dpi=130)
        plt.close(fig)
    except Exception as e:  # plot is decoration, never gate the run on it
        print(f"[warn] plot skipped: {e}")

    return {"final_test": rows[-1]["test_acc"], "best_test": best_test,
            "final_train_loss": rows[-1]["train_loss"]}


def write_baseline_md(seeds, info, args):
    finals = [info[s]["final_test"] for s in seeds]
    bests = [info[s]["best_test"] for s in seeds]
    mean = float(np.mean(finals))
    std = float(np.std(finals))
    lines = [
        "# BASELINE SCORE TO BEAT — Fashion-MNIST, plain MLP, plain SGD",
        "",
        f"- Model: 784 -> {args.hidden} ReLU -> 10, softmax cross-entropy",
        f"- Optimizer: our own plain SGD, fixed learning rate {args.lr} (manual, no momentum)",
        f"- Epochs: {args.epochs}, batch: {args.batch}, hidden: {args.hidden}",
        f"- Seeds: {seeds} (identical hyperparameters, only the seed changes)",
        f"- Data: Fashion-MNIST, pixels /255 (float32). License: MIT (DATA_LICENSES.md)",
        "",
        "| seed | final test acc | best test acc (any epoch) |",
        "|------|---------------:|--------------------------:|",
    ]
    for s in seeds:
        lines.append(f"| {s} | {info[s]['final_test'] * 100:.2f}% | {info[s]['best_test'] * 100:.2f}% |")
    lines += [
        "",
        f"**BASELINE = final-epoch test accuracy, mean ± std over {len(seeds)} seeds: "
        f"**{mean * 100:.2f}% ± {std * 100:.2f}%**",
        f"(best-epoch mean: {float(np.mean(bests)) * 100:.2f}% ± {float(np.std(bests)) * 100:.2f}%)",
        "",
        "Phase 2 must beat this number to pass its gate.",
        "",
        f"Reproduce: python3 phase1_fmnist.py --seeds {' '.join(map(str, seeds))} "
        f"--epochs {args.epochs} --hidden {args.hidden} --lr {args.lr} --batch {args.batch}",
        "",
    ]
    path = os.path.join(RUNS, "BASELINE.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    print("=" * 66)
    print("GATE 1 first (no training on broken gradients)")
    print("=" * 66)
    gate1()

    Xtr, ytr, Xte, yte = load_fmnist(DATA_DIR)
    print(f"[data] train {Xtr.shape} test {Xte.shape}")

    info = {}
    t0 = time.time()
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        info[seed] = run_seed(seed, Xtr, ytr, Xte, yte, args)
    total = time.time() - t0
    write_baseline_md(args.seeds, info, args)

    finals = [info[s]["final_test"] for s in args.seeds]
    mean, std = float(np.mean(finals)), float(np.std(finals))
    print(f"\nTOTAL {total:.0f}s | BASELINE (final-epoch test acc, mean±std): "
          f"{mean * 100:.2f}% ± {std * 100:.2f}%")
    with open(os.path.join(RUNS, "baseline_summary.json"), "w") as f:
        json.dump({"seeds": args.seeds,
                   "per_seed_final": {str(s): info[s]["final_test"] for s in args.seeds},
                   "mean": mean, "std": std}, f, indent=2)


if __name__ == "__main__":
    main()