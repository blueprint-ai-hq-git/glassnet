"""glassnet/report.py — Phase 5: text receipts and visual reports.

receipt()              -> human text: "I said X because of rule #a ..."
draw_explain_figure()  -> matplotlib figure: image, named concept bars,
                          rule contributions, class probabilities.
"""
import numpy as np

from .data import LABELS
from .concepts import NAMES as CONCEPT_NAMES


def receipt(model, x_img, head="image"):
    """Full plain-text why-answer. x_img: (784,) or (28,28) floats [0,1]."""
    x_img = np.asarray(x_img, np.float32).reshape(1, 784)
    f = model.predict(x_img)
    logits = f["logits"].data[0]
    acts = f["concepts"].data[0]
    rules = f["rules"].data[0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    pred = int(logits.argmax())
    contrib = model.answer.W.data[:, pred] * rules
    order = np.argsort(contrib)[::-1]

    lines = [str(head),
             f"I said {pred} ({LABELS[pred]}) with probability "
             f"{probs[pred] * 100:.1f}%",
             "because of these rules (fire % | push | connected concepts "
             "with weight and activation):"]
    for r in order[:3]:
        lines.append(f"  rule #{r} fires {rules[r] * 100:.0f}% "
                     f"pushes {contrib[r]:+.3f}")
        w = model.rules.W.data[:, r]
        nz = np.where(np.abs(w) > 1e-4)[0]
        nz = nz[np.argsort(np.abs(w[nz]))[::-1]]
        for k in nz:
            lines.append(f"      {CONCEPT_NAMES[k]}: w {w[k]:+.2f} "
                         f"{'ON' if acts[k] > 0.5 else 'off'} (a={acts[k]:.2f})")
    lines.append("class probabilities: " + ", ".join(
        f"{LABELS[c]} {probs[c] * 100:.1f}%"
        for c in np.argsort(probs)[::-1][:4]))
    return "\n".join(lines)


def draw_explain_figure(model, x_img, head, path):
    """One picture: the image, the named concept activations, which rules
    push toward the answer (and how hard), and the class probabilities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_img = np.asarray(x_img, np.float32).reshape(1, 784)
    f = model.predict(x_img)
    logits = f["logits"].data[0]
    acts = f["concepts"].data[0]
    rules = f["rules"].data[0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    pred = int(logits.argmax())
    contrib = model.answer.W.data[:, pred] * rules

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    axs[0].imshow(x_img.reshape(28, 28), cmap="gray_r")
    axs[0].set_title(f"{head}\npred: {LABELS[pred]} ({probs[pred] * 100:.0f}%)",
                     fontsize=10)
    axs[0].axis("off")

    order = np.argsort(acts)[::-1]
    axs[1].barh(np.arange(len(CONCEPT_NAMES))[::-1], acts[order])
    axs[1].set_yticks(np.arange(len(CONCEPT_NAMES))[::-1])
    axs[1].set_yticklabels([CONCEPT_NAMES[i] for i in order], fontsize=8)
    axs[1].set_title("concept activations", fontsize=10)

    top = np.argsort(np.abs(contrib))[::-1][:8]
    colors = ["#2ca02c" if c >= 0 else "#d62728" for c in contrib[top]]
    axs[2].barh(range(len(top)), contrib[top], color=colors)
    axs[2].set_yticks(range(len(top)))
    axs[2].set_yticklabels([f"rule #{r}" for r in top], fontsize=9)
    axs[2].set_title(f"rule contributions to {LABELS[pred]}", fontsize=10)

    axs[3].bar(range(10), probs)
    axs[3].set_xticks(range(10))
    axs[3].set_xticklabels(LABELS, rotation=90, fontsize=7)
    axs[3].set_title("class probabilities", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path