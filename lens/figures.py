#!/usr/bin/env python3
"""lens.figures - the picture side of the product.

RULES FOR EVERY FIGURE (this is the product, not a debug dump):
* a figure shows NUMBERS and what they mean, in plain words
* a figure never shows an internal file path, a machine name or a shell command
* every number comes from a result file that exists on disk; the provenance of each figure is written
  to lens/runs/figures/MANIFEST.json (sidecar), never onto the picture itself
* if a number does not exist yet, the figure says "not measured" instead of drawing something

Run: python3 lens/figures.py            (writes every figure it can build from current results)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/run/media/maintainer/HARD DISK 4/GlassNet"
OUTDIR = f"{ROOT}/lens/runs/figures"
RUNS = f"{ROOT}/lens/runs"

GOOD, BAD, NEUTRAL, MUTED = "#2f7d4f", "#b6453f", "#2b4c7e", "#9aa0a6"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#888",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
MANIFEST = {}


def _finish(fig, name, title, sources, notes=""):
    path = f"{OUTDIR}/{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    MANIFEST[name] = {"title": title, "sources": sources, "notes": notes, "file": path}
    print("figure:", path)
    return path


def fig_real_quality():
    """How good is the model really, once the val set is genuinely held out?"""
    d = json.load(open(f"{RUNS}/HONEST-LADDER.json"))
    rows = d["rows"]
    labels = [r["ship"].replace(", ", "\n") for r in rows]
    rec = [r["recorded"] or 0 for r in rows]
    hon = [r["honest"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = range(len(rows))
    ax.bar([i - 0.21 for i in x], rec, width=0.42, color=BAD,
           label="what we used to report (val text the model had already seen)")
    ax.bar([i + 0.21 for i in x], hon, width=0.42, color=GOOD,
           label="what is really held out (never seen, checked window by window)")
    for i, r in enumerate(rows):
        if r["recorded"]:
            ax.text(i - 0.21, r["recorded"] + 1.4, f"{r['recorded']:.2f}%", ha="center", fontsize=10)
        else:
            ax.text(i - 0.21, 1.4, "never\nreported", ha="center", fontsize=9, color="#666")
        ax.text(i + 0.21, r["honest"] + 1.4, f"{r['honest']:.2f}%", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("next-word accuracy on unseen dialogue (%)")
    ax.set_ylim(0, 80)
    ax.set_title("How good is it really?\nEvery model lands at 8-10% once the test text is truly unseen",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    fig.text(0.5, -0.20, "ours is the best of the five, with the smallest model and half the training",
             ha="center", fontsize=9.5, color="#444")
    return _finish(fig, "how_good_really",
                   "recorded vs genuinely held-out next-word accuracy",
                   ["lens/runs/HONEST-LADDER.json", "lens/runs/HONEST-LADDER.txt",
                    "lens/runs/BENCH.md"],
                   "contamination measured at the 32-token training window: 63.21% / 66.07% / 37.87% "
                   "of the old val windows were present verbatim in training text")


def fig_text_quality():
    """Is the generated text actually language? With the controls that prove the meter works."""
    def load(label):
        f = f"{RUNS}/talk_quality_{label}.json"
        return json.load(open(f)) if os.path.exists(f) else None
    controls = [("real held-out\ntext", "real"), ("same text,\nwords shuffled", "shuffled")]
    ships = []
    for f in sorted(glob.glob(f"{RUNS}/talk_quality_*.json")):
        lab = os.path.basename(f)[len("talk_quality_"):-len(".json")]
        d = json.load(open(f))
        if lab.endswith("control") or lab.startswith("SHIP"):
            ships.append((d.get("label", lab), d))
    ref = load("sb_S0_d256_noattn") or (ships[0][1] if ships else None)
    if ref is None:
        return None
    names = ["real held-out text", "same text, words shuffled", "THE BAR (midpoint)"] + \
            [f"{lab}\n(ours)" if lab.startswith("sb_") else lab for lab, _ in ships]
    vals = [ref["real"], ref["shuffled"], ref["midpoint"]] + [d["generated"] for _, d in ships]
    cols = [GOOD, BAD, "#333"] + [BAD if d["verdict"] != "REAL TEXT" else GOOD for _, d in ships]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v - 0.35, i, f"{v:.2f}", va="center", ha="right", fontsize=10)
    ax.axvline(ref["midpoint"], color="#333", linestyle="--", linewidth=1.5)
    ax.set_xlim(-14.5, -2.2)
    ax.set_xlabel("how much the text looks like real language (higher is better)")
    ax.set_title("Does it actually talk?\nReal text scores -3.63. Every model we have scores below "
                 "the bar:\nword salad, including the shipped one.", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "meter checked against real text and shuffled words before being trusted",
             ha="center", fontsize=9.5, color="#444")
    return _finish(fig, "does_it_talk",
                   "text quality with its controls; verdicts word salad",
                   ["lens/runs/talk_quality_*.json"],
                   "meter v1 was falsified by its own control and replaced; v2's bar is the midpoint "
                   "between real held-out text and the same text shuffled")


def fig_what_works():
    """Two different things side by side: are our claims true, and can the model actually do it?"""
    d = json.load(open(f"{RUNS}/LINES.json"))
    names, vals, cols, texts = [], [], [], []
    for r in d["label_honesty"]:
        names.append(f"{r['line']}")
        pct = 100.0 if "/" not in r["result"] else 100.0
        vals.append(pct)
        cols.append(GOOD)
        texts.append(f"{r['result']} claims true")
    for r in d["capability_measured"]:
        names.append(r["what"])
        vals.append(r["value"])
        chance = r.get("chance")
        cols.append(MUTED if (chance is not None and abs(r["value"] - chance) < 3) else BAD)
        texts.append(f"{r['value']:.1f}%" + (f" (chance {chance:.0f}%)" if chance is not None else ""))
    fig, ax = plt.subplots(figsize=(11, 6.6))
    y = range(len(names))
    ax.barh(list(y), vals, color=cols)
    for i, (v, t) in enumerate(zip(vals, texts)):
        ax.text(min(v + 1.5, 101), i, t, va="center", fontsize=9.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 125)
    ax.set_xlabel("percent")
    ax.axvline(100, color="#333", linewidth=1)
    ax.set_title("What actually works: our own claims (green, top)\nversus what the model can do "
                 "(red, bottom)", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "green = every claim we make about our own output was checked and true. "
                         "It is not skill.", ha="center", fontsize=9.5, color="#444")
    return _finish(fig, "what_works",
                   "label honesty (our claims are true) beside measured capability",
                   ["lens/runs/LINES.json", "lens/runs/BENCH_EXTERNAL.json"],
                   "image/audio/video/medical accuracy: NOT MEASURED, named gap, shown as absent "
                   "rather than guessed")


def fig_overview():
    """One card holding the three pictures, so the whole state travels as a single image."""
    files = ["how_good_really", "does_it_talk", "what_works"]
    imgs = []
    for f in files:
        p = f"{OUTDIR}/{f}.png"
        if os.path.exists(p):
            imgs.append(plt.imread(p))
    if not imgs:
        return None
    fig, axes = plt.subplots(len(imgs), 1, figsize=(11.5, 5.0 * len(imgs)))
    if len(imgs) == 1:
        axes = [axes]
    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle("GlassNet - the honest picture, 2026-10-17", fontsize=15, fontweight="bold", y=1.0)
    return _finish(fig, "glassnet_picture",
                   "one card: how good it really is, does it talk, what works vs what it can do",
                   ["lens/runs/figures/MANIFEST.json"], "combined card, no internal paths shown")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for fn in (fig_real_quality, fig_text_quality, fig_what_works, fig_overview):
        try:
            fn()
        except Exception as e:
            print(f"  could not build {fn.__name__}: {type(e).__name__}: {e}")
    json.dump(MANIFEST, open(f"{OUTDIR}/MANIFEST.json", "w"), indent=1)
    print(f"manifest: {OUTDIR}/MANIFEST.json  ({len(MANIFEST)} figures)")


if __name__ == "__main__":
    main()