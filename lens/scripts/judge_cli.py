"""judge_cli.py — USE the data judge: score text or corpus windows as good or bad.

    .venv-gpu/bin/python3 -u lens/scripts/judge_cli.py "some text to score"
    .venv-gpu/bin/python3 -u lens/scripts/judge_cli.py --window-ids 12,45,900,...

Every score is explained by construction: the shipping model is a logistic head on named
features, so the output prints each feature's value AND its weighted contribution. Nothing is
hidden, and a low score can be argued with feature by feature.

The model + features are the ones measured in lens/runs/data_judge/judge_report.json
(held-out AUROC 0.9839, every corruption kind >= 0.94).
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import chat_data                                    # noqa: E402
from data_judge import (FEATS, build_bigram_table, bigram_plaus, make_features)  # noqa: E402

JUDGE = os.path.join(PROJ, "lens", "runs", "data_judge", "judge_linear.npz")
WIN = 32


def load_judge():
    z = np.load(JUDGE)
    W = np.asarray(z["W"], dtype=np.float64)        # (F, 2)
    b = np.asarray(z["b"], dtype=np.float64)
    return W[:, 1] - W[:, 0], float(b[1] - b[0]), [str(s) for s in z["feats"]]


def corpus_context():
    d = chat_data.build()
    ids, vocab = d["ids"], d["vocab"]
    V = len(vocab)
    counts = np.bincount(ids, minlength=V).astype(np.float64)
    freqs_log = np.log1p(counts)
    rare_thr = float(np.percentile(freqs_log[counts > 0], 10))
    punct_ids = {i for i, w in enumerate(vocab) if w and all(not c.isalnum() for c in w)}
    unk_id = vocab.index("<unk>") if "<unk>" in vocab else -1
    assistant_ids = {i for i, w in enumerate(vocab) if w.lower() == "assistant"}
    uni_cnt = counts
    up, pc = build_bigram_table(ids, V)
    return (vocab, freqs_log, rare_thr, punct_ids, unk_id, assistant_ids,
            (up, pc, uni_cnt, V))


def score_windows(X, ctx, W, b, feats):
    vocab, freqs_log, rare_thr, punct_ids, unk_id, assistant_ids, BIG = ctx
    F = make_features(np.asarray(X), freqs_log, rare_thr, punct_ids, unk_id, assistant_ids, BIG)
    logit = F @ W + b
    p = 1.0 / (1.0 + np.exp(-logit))
    return F, p


def text_to_window(text, vocab, win=WIN):
    vmap = {w: i for i, w in enumerate(vocab)}
    toks = [vmap.get(w, vmap.get("<unk>", 0)) for w in text.split()]
    if len(toks) >= win:
        return np.asarray([toks[:win]], dtype=np.int64)
    return np.asarray([toks + [0] * (win - len(toks))], dtype=np.int64)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    W, b, feats = load_judge()
    ctx = corpus_context()
    vocab = ctx[0]
    if args[0] == "--window-ids":
        X = np.asarray([[int(t) for t in args[1].split(",")]], dtype=np.int64)
        label = "window ids"
    else:
        text = " ".join(args)
        X = text_to_window(text, vocab)
        label = text[:60]
    F, p = score_windows(X, ctx, W, b, feats)
    verdict = "GOOD" if p[0] >= 0.5 else "BAD"
    print(f"item: {label!r}")
    print(f"  score {p[0]:.4f} -> {verdict}   (0.5 is the line)")
    contrib = sorted(zip(feats, F[0], F[0] * W), key=lambda t: -abs(t[2]))
    print("  why (feature = value -> weighted contribution):")
    for nm, v, c in contrib:
        print(f"    {nm:<14} {v:>9.3f}  ->  {c:+8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())