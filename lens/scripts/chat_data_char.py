"""chat_data_char.py — character-level tokenisation for a SMALL model.

WHY THIS IS THE BIGGEST REMAINING LEVER
---------------------------------------
The optimisation diagnostic measured the chat model at d=128, K=9:
  embedding   V*D = 12,000 * 128 = 1,536,000   92.9% of all parameters
  answer      K*V = 9 * 12,000   =   108,000    6.5%

and then the answer layer must pick **1 of 12,000 words** using only 9 numbers. That is 13.6 bits
of choice through a 9-wide pipe.

A character-level model changes both numbers at once:
  - the vocabulary is ~60 characters, not 12,000, so the embedding falls from 1,536,000 to about
    8,000 parameters -- the concept layer stops being 0.6% of the model
  - the output layer picks 1 of ~60, not 1 of 12,000 -- 5.9 bits instead of 13.6
  - there is no <unk>, so the 1.92% of tokens currently thrown away is kept

The cost is honest and known: sequences get ~5x longer for the same text, so each window covers
less conversation. Whether that trade is worth it is exactly what the run measures.

This is a TOKENISATION change, nothing else. Same corpus, same architecture, same windows, same
optimiser, same epochs.
"""
import json
import os

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CH = os.path.join(PROJ, "data", "chat")
CACHE = os.path.join(CH, "prepared")
os.makedirs(CACHE, exist_ok=True)

MAX_CHARS = int(os.environ.get("CD_MAX_CHARS", "4000"))
VAL_FRAC = 0.02
SEED = 0


def load_conversations():
    import gzip
    for name in ("hh_rlhf_helpful_train.jsonl.gz", "hh_rlhf_harmless_train.jsonl.gz"):
        p = os.path.join(CH, name)
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("chosen") or ""
                if t.strip():
                    yield t[:MAX_CHARS]


def build():
    cache_ids = os.path.join(CACHE, "char_ids.npy")
    cache_vocab = os.path.join(CACHE, "char_vocab.json")
    cache_meta = os.path.join(CACHE, "char_meta.json")
    cache_bounds = os.path.join(CACHE, "char_bounds.npy")
    cache_split = os.path.join(CACHE, "char_split.npy")
    if all(os.path.exists(p) for p in (cache_ids, cache_vocab, cache_meta,
                                       cache_bounds, cache_split)):
        ids = np.load(cache_ids)
        vocab = json.load(open(cache_vocab))
        meta = json.load(open(cache_meta))
        bounds = np.load(cache_bounds).tolist()
        split = np.load(cache_split).tolist()
        print(f"  cached: {len(ids):,} characters, vocab {len(vocab):,}")
        return {"ids": ids, "vocab": vocab, "bounds": bounds, "split": split, **meta}

    convs = list(load_conversations())
    print(f"  {len(convs):,} conversations")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(convs))
    n_val = max(1, int(len(convs) * VAL_FRAC))
    val_idx = set(order[:n_val].tolist())

    from collections import Counter
    freq = Counter()
    for i, c in enumerate(convs):
        if i in val_idx:
            continue
        freq.update(c.lower())
    vocab = sorted(freq.keys())
    vmap = {ch: i for i, ch in enumerate(vocab)}
    print(f"  character vocab from TRAIN text only: {len(vocab)} distinct characters")
    print(f"  most common: {freq.most_common(8)}")

    all_ids, bounds, split = [], [], []
    for i, c in enumerate(convs):
        enc = [vmap[ch] for ch in c.lower() if ch in vmap]
        if len(enc) < 16:
            continue
        bounds.append(len(all_ids))
        all_ids.extend(enc)
        split.append(1 if i in val_idx else 0)
    ids = np.asarray(all_ids, dtype=np.int64)
    bounds = np.asarray(bounds, dtype=np.int64)
    split = np.asarray(split, dtype=np.int8)
    meta = {"n_conv": len(bounds), "val_frac": VAL_FRAC, "seed": SEED}
    np.save(cache_ids, ids)
    np.save(cache_bounds, bounds)
    np.save(cache_split, split)
    json.dump(vocab, open(cache_vocab, "w"))
    json.dump(meta, open(cache_meta, "w"))
    print(f"  encoded {len(ids):,} characters over {len(bounds):,} conversations")
    return {"ids": ids, "vocab": vocab, "bounds": bounds.tolist(), "split": split.tolist(), **meta}


def make_windows(ids, bounds, split, window, which, stride=None):
    """Character windows. Stride defaults to window//4 because characters carry less per step than
    words, so more overlapping windows are needed to cover the same text."""
    if stride is None:
        stride = max(1, window // 4)
    X, Y = [], []
    for j in range(len(bounds) - 1):
        if split[j] != which:
            continue
        a, b = bounds[j], bounds[j + 1]
        seg = ids[a:b]
        for s in range(0, len(seg) - window - 1, stride):
            X.append(seg[s:s + window])
            Y.append(seg[s + window])
    return np.asarray(X, dtype=np.int64), np.asarray(Y, dtype=np.int64)


if __name__ == "__main__":
    d = build()
    v = d["vocab"]
    print(f"  total characters: {len(d['ids']):,}   vocab: {len(v):,}")
    print("  sample decoded: " + "".join(v[i] for i in d["ids"][:120]))
