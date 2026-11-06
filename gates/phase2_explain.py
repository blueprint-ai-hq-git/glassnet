"""PHASE 2 — explain(): "I said X because of rule #a (needs these concepts)
and rule #b (blocks that)". Prints a text receipt and saves a picture.

Run:  python3 phase2_explain.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.data import load as load_pixels, LABELS  # noqa: E402
from glassnet.concepts import load as load_concepts, NAMES  # noqa: E402
from glassnet.gates import gate1  # noqa: E402
from glassnet.phase2_model import GlassNetPhase2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase2")
DATA_DIR = os.path.join(HERE, "data")
OUT = os.path.join(RUNS, "explain")
os.makedirs(OUT, exist_ok=True)


def explain(model, x_img, true_label):
    """Full receipt for one image. x_img: (784,) float32."""
    f = model.predict(x_img.reshape(1, 784))
    logits = f["logits"].data[0]
    acts = f["concepts"].data[0]
    rules = f["rules"].data[0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    pred = int(logits.argmax())
    W_ans = model.answer.W.data      # (R, 10)
    contrib = W_ans[:, pred] * rules  # per-rule contribution to pred class
    order = np.argsort(contrib)[::-1]

    lines = []
    lines.append(f"image true label = {true_label} ({LABELS[true_label]})")
    lines.append(f"I said             = {pred} ({LABELS[pred]}) "
                 f"with probability {probs[pred]*100:.1f}%")
    lines.append("")
    lines.append("because of these rules (concepts in brackets = connected, "
                 "w = connection weight, a = current concept activation):")
    top2 = order[:2]
    for r in top2:
        conns = rule_concepts(model, r, acts)
        lines.append(f"  rule #{r} (fires {rules[r]*100:.0f}%) "
                     f"pushes {LABELS[pred]} by {contrib[r]:+.3f}")
        for name, w, a in conns:
            state = "ON" if a > 0.5 else "off"
            lines.append(f"      {name}: w {w:+.2f}, {state} (a={a:.2f})")
    blockers = order[-3:]
    lines.append("  blockers (rules voting against):")
    for r in blockers:
        conns = rule_concepts(model, r, acts)
        names = ", ".join(f"{c[0]}{' ON' if c[2]>0.5 else ''}" for c in conns)
        lines.append(f"      rule #{r} (fires {rules[r]*100:.0f}%) "
                     f"contributes {contrib[r]:+.3f}  [{names}]")
    lines.append("")
    lines.append("class probabilities: " + ", ".join(
        f"{LABELS[c]} {probs[c]*100:.1f}%" for c in np.argsort(probs)[::-1][:4]))
    return "\n".join(lines), pred, probs, acts, rules, contrib, logits


def rule_concepts(model, r, acts, threshold=1e-4):
    w = model.rules.W.data[:, r]
    nz = np.abs(w) > threshold
    idx = np.where(nz)[0]
    idx = idx[np.argsort(np.abs(w[idx]))[::-1]]
    return [(NAMES[i], float(w[i]), float(acts[i])) for i in idx]


def draw(model, x_img, true_label, pred, probs, acts, contrib, logits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    axs[0].imshow(x_img.reshape(28, 28), cmap="gray_r")
    axs[0].set_title(f"true: {LABELS[true_label]}\npred: {LABELS[pred]} "
                     f"({probs[pred]*100:.0f}%)")
    axs[0].axis("off")
    order = np.argsort(acts)[::-1]
    axs[1].barh(np.arange(len(NAMES))[::-1], acts[order])
    axs[1].set_yticks(np.arange(len(NAMES))[::-1])
    axs[1].set_yticklabels([NAMES[i] for i in order], fontsize=8)
    axs[1].set_title("concept activations")
    top = np.argsort(np.abs(contrib))[::-1][:8]
    axs[2].barh(range(len(top)), contrib[top], color=["#2ca02c" if c >= 0 else "#d62728" for c in contrib[top]])
    axs[2].set_yticks(range(len(top)))
    axs[2].set_yticklabels([f"rule #{r}" for r in top], fontsize=9)
    axs[2].set_title(f"rule contributions to {LABELS[pred]}")
    axs[3].bar(range(10), probs)
    axs[3].set_xticks(range(10))
    axs[3].set_xticklabels(LABELS, rotation=90, fontsize=7)
    axs[3].set_title("class probabilities")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    print("=" * 66)
    print("GATE 1 first")
    print("=" * 66)
    gate1()
    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR)
    Ctr, Cte, lo, hi = load_concepts(DATA_DIR)
    with open(os.path.join(RUNS, "config.json")) as f:
        cfg = json.load(f)
    seed = 0
    model = GlassNetPhase2.load(os.path.join(RUNS, f"final_seed{seed}.npz"))

    blobs = []
    for label in [1, 5, 8]:  # trouser, sandal, bag — chosen to show variety
        idx = int(np.where(yte == label)[0][0])
        text, pred, probs, acts, rules, contrib, logits = explain(model, Xte[idx], label)
        blobs.append(f"### example: test #{idx} (true={label})\n\n{text}\n")
        print("-" * 60)
        print(text, flush=True)
        draw(model, Xte[idx], label, pred, probs, acts, contrib, logits,
             os.path.join(OUT, f"explain_idx{idx}.png"))
        with open(os.path.join(OUT, f"explain_idx{idx}.txt"), "w") as f:
            f.write(text + "\n")
    with open(os.path.join(RUNS, "explain.md"), "w") as f:
        f.write("# Phase 2 — explain() receipts\n\n" + "\n".join(blobs) + "\n")
    import glob
    pngs = sorted(glob.glob(os.path.join(OUT, "*.png")))
    print(f"\n[wrote] text receipts + {len(pngs)} pictures in runs/phase2/explain/")


if __name__ == "__main__":
    main()