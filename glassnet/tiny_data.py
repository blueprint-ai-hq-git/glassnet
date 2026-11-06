"""glassnet/tiny_data — tiny Shakespeare, WORD-level windows (Rung T2).

Next-word prediction: window of the last WIN tokens -> next token id.
Splits are CONTIGUOUS positional slices of the token stream (train 90% /
val 5% / test 5% — the temporal-LM convention, no shuffling). Vocabulary
is built from TRAIN tokens only (cap most frequent, +<unk> row). Cached
to data/tiny_shakespeare/t2_windows.npz; deterministic.
"""
import os
import re
import json
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "tiny_shakespeare")
TXT = os.path.join(DATA, "input.txt")
CACHE_TMPL = os.path.join(DATA, "t2_windows_win{win}_v{vocab_cap}.npz")


def cache_path(win, vocab_cap=8000):
    """Per-(window, vocab) cache: a single shared path would make a different
    window length OR a different vocabulary size silently return the other
    one's cached data — which would fabricate a result."""
    return CACHE_TMPL.format(win=int(win), vocab_cap=int(vocab_cap))


def tokenize(text):
    """Whitespace tokens (nanoGPT-style word list; punctuation attached)."""
    return text.split()


def build(win=8, vocab_cap=8000, split=(0.90, 0.05, 0.05), force=False):
    """Build (or load cached) windows. Returns dict with Xtr,ytr,Xva,yva,
    Xte,yte (int32 arrays), vocab (list), V (incl. <unk>)."""
    cache = cache_path(win, vocab_cap)
    if os.path.exists(cache) and not force:
        z = np.load(cache, allow_pickle=True)
        seg = json.loads(str(z["seg"][0]))
        if int(seg.get("win", -1)) != int(win):
            raise RuntimeError(f"cache win mismatch: asked {win}, "
                               f"cached {seg.get('win')} ({cache})")
        if int(seg.get("vocab_cap", -1)) != int(vocab_cap):
            raise RuntimeError(f"cache vocab mismatch: asked {vocab_cap}, "
                               f"cached {seg.get('vocab_cap')} ({cache})")
        return dict(Xtr=z["Xtr"], ytr=z["ytr"], Xva=z["Xva"], yva=z["yva"],
                    Xte=z["Xte"], yte=z["yte"],
                    vocab=z["vocab"].tolist(), V=int(z["V"]),
                    win=int(win), seg=seg)
    text = open(TXT, encoding="utf-8").read()
    toks = tokenize(text)
    N = len(toks)
    tr_n = int(N * split[0])
    va_n = int(N * split[1])
    tr, va, te = toks[:tr_n], toks[tr_n:tr_n + va_n], toks[tr_n + va_n:]
    cnt = Counter(tr)
    vocab = [w for w, _ in cnt.most_common(vocab_cap)]
    vmap = {w: i for i, w in enumerate(vocab)}
    V = len(vocab) + 1          # + <unk>
    UNK = V - 1

    def to_windows(seg):
        ids = np.array([vmap.get(w, UNK) for w in seg], dtype=np.int32)
        m = len(ids) - win
        X = np.empty((m, win), dtype=np.int32)
        y = np.empty(m, dtype=np.int32)
        for i in range(win):
            X[:, i] = ids[i:i + m]
        y[:] = ids[win:]
        return X, y

    Xtr, ytr = to_windows(tr)
    Xva, yva = to_windows(va)
    Xte, yte = to_windows(te)
    seg = {"train": tr_n, "val": va_n, "test": N - tr_n - va_n, "tokens": N,
           "win": int(win), "vocab": len(vocab), "vocab_cap": int(vocab_cap)}
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
             vocab=np.array(vocab, dtype="<U48"), V=np.int32(V),
             seg=np.array([json.dumps(seg)]))
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
                vocab=vocab, V=V, win=win, seg=seg)


if __name__ == "__main__":
    d = build()
    print("tokens:", d["seg"])
    print("windows train/val/test:",
          d["Xtr"].shape, d["Xva"].shape, d["Xte"].shape)
    print("V (incl <unk>):", d["V"], "| vocab head:", d["vocab"][:10])
    # sanity: the most frequent target (the LM prior) — <unk> may win
    c = Counter(d["ytr"].tolist())
    top_id, top_c = c.most_common(1)[0]
    unk_c = c.get(d["V"] - 1, 0)
    top_word = d["vocab"][top_id] if top_id < len(d["vocab"]) else "<unk>"
    print("top target id", top_id, repr(top_word),
          "freq %.2f%%" % (100 * top_c / len(d["ytr"])))
    print("<unk> share: %.2f%%" % (100 * unk_c / len(d["ytr"])))