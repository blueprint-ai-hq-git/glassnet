"""t3_baselines.py — reviewer item (3), part 1: the count baselines.

Builds the redesigned dataset (vocabulary ~1,500 instead of 8,001) and computes
the two count-based reference predictors on the VALIDATION split only:

  1. UNIGRAM  — predict the most frequent training word; perplexity from the
                training unigram distribution.
  2. BIGRAM   — predict from p(next | previous word) with Dirichlet smoothing
                (alpha=1) backing off to the unigram:
                    p(w|u) = (c(u,w) + alpha * p_uni(w)) / (c(u) + alpha)
                Counts are taken from the TRAIN split only.

Both are reported as accuracy AND perplexity, because with a small vocabulary
accuracy is dominated by the <unk> share and perplexity is the honest signal.

Nothing here trains a neural net; this is the reference line the models must
beat. Test split is never read.
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

WIN = 8
VOCAB_CAP = 1500
ALPHA = 1.0          # Dirichlet smoothing for the bigram
RUNDIR = os.path.join(PROJ, "runs", "t3")
os.makedirs(RUNDIR, exist_ok=True)
OUT = os.path.join(RUNDIR, "baselines.json")
LOG = os.path.join(RUNDIR, "baselines.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def main():
    d = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
    V = d["V"]
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xva, yva = d["Xva"], d["yva"]
    vocab = d["vocab"]
    UNK = V - 1

    log("=" * 74)
    log(f"ITEM (3) BASELINES — vocab_cap={VOCAB_CAP}, win={WIN}")
    log("=" * 74)
    log(f"  V = {V} (incl. <unk> at {UNK})")
    log(f"  windows: train {len(Xtr)}, val {len(Xva)}")
    log(f"  val <unk> share: {float((yva == UNK).mean()*100):.2f}%")
    log(f"  train <unk> share: {float((ytr == UNK).mean()*100):.2f}%")

    # ---- train counts (TRAIN ONLY — no leakage) ----
    uni = np.bincount(ytr, minlength=V).astype(np.float64)
    p_uni = uni / uni.sum()

    big = np.zeros((V, V), dtype=np.float64)
    prev = Xtr[:, -1].astype(np.int64)          # last token of each window
    np.add.at(big, (prev, ytr.astype(np.int64)), 1.0)
    c_u = big.sum(axis=1)
    p_big = (big + ALPHA * p_uni[None, :]) / (c_u[:, None] + ALPHA)

    # ---- evaluate on VAL ----
    yv = yva.astype(np.int64)
    pv = Xva[:, -1].astype(np.int64)

    def report(name, pred, probs):
        acc = float((pred == yv).mean() * 100)
        p_true = probs[np.arange(len(yv)), yv]
        nll = float(-np.log(np.maximum(p_true, 1e-12)).mean())
        ppl = float(math.exp(nll))
        log(f"  {name:26s} acc {acc:6.2f}%   CE {nll:6.4f}   ppl {ppl:8.2f}")
        return {"acc": acc, "ce": nll, "ppl": ppl}

    log("")
    log("  reference predictors (validation only):")

    # majority / unigram-argmax
    maj = int(np.argmax(uni))
    uni_pred = np.full(len(yv), maj, dtype=np.int64)
    uni_probs = np.tile(p_uni[None, :], (len(yv), 1))
    r_uni = report("unigram (argmax)", uni_pred, uni_probs)
    log(f"      -> its class is {maj} "
        f"('{vocab[maj] if maj < len(vocab) else '<unk>'}'), train share "
        f"{p_uni[maj]*100:.2f}%")

    # bigram, backing off through the smoothed distribution
    big_pred = np.argmax(p_big[pv], axis=1)
    big_probs = p_big[pv]
    r_big = report("bigram (smoothed, a=1)", big_pred, big_probs)

    # how often the bigram context was unseen
    unseen = float((c_u[pv] == 0).mean() * 100)
    log(f"      -> unseen bigram context for {unseen:.2f}% of val windows")

    # a pure <unk> predictor, for reference
    unk_pred = np.full(len(yv), UNK, dtype=np.int64)
    r_unk = report("always <unk>", unk_pred, np.tile(p_uni[None, :], (len(yv), 1)))

    log("")
    log("  param counts for the equal-param neural comparison (D=64, W=8):")
    for K in (24, 64, 128, 256):
        n = V * 64 + (64 * WIN) * K + K + K * V + V
        log(f"      K=hidden={K:4d}  -> {n:,} params (readable == normal at equal K)")

    res = {"vocab_cap": VOCAB_CAP, "win": WIN, "V": V,
           "val_unk_share": float((yva == UNK).mean() * 100),
           "train_unk_share": float((ytr == UNK).mean() * 100),
           "n_train_windows": int(len(Xtr)), "n_val_windows": int(len(Xva)),
           "unigram": r_uni, "bigram": r_big, "always_unk": r_unk,
           "bigram_unseen_ctx_pct": unseen,
           "vocab_head": vocab[:12]}
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
