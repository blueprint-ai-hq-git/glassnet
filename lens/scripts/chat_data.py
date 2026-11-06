"""chat_data.py — turn licensed human dialogue into training windows for a talking model.

DATA
----
Anthropic/hh-rlhf (MIT): real human conversations in plain text, already formatted as

    Human: <message>

    Assistant: <reply>

so no format has to be invented. Only the `chosen` field is used -- the reply a human annotator
preferred. The `rejected` field is deliberately NOT used here: it is there for preference training,
which is a different experiment, and mixing it in now would confound this one.

WHY THIS CORPUS AND NOT SHAKESPEARE
-----------------------------------
Shakespeare is word-level, archaic, and not conversational. A model trained on it cannot hold a
conversation, which is the thing the owner actually wants. This is human dialogue.

TOKENISATION
------------
Word-level with a capped vocabulary, same scheme as the concept-layer work, so the existing
concept machinery and `explain()` still apply. Unknown words map to <unk>. The cap is a
pre-registered constant, not tuned.
"""
import gzip
import json
import os
import re
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CH = os.path.join(PROJ, "data", "chat")
# CHAT_DATA_DIR lets a CURATED corpus be trained/evaluated without touching the originals:
# originals stay in data/chat/prepared, the deduplicated copy lives in data/chat/prepared_clean.
# Default is unchanged, so every existing run reproduces exactly.
CACHE = os.environ.get("CHAT_DATA_DIR") or os.path.join(CH, "prepared")
os.makedirs(CACHE, exist_ok=True)

VOCAB_CAP = int(os.environ.get("CD_VOCAB", "12000"))
MAX_CHARS = int(os.environ.get("CD_MAX_CHARS", "4000"))     # per conversation, keeps windows sane
VAL_FRAC = 0.02
SEED = 0

TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")


def tokenize(text):
    return TOKEN_RE.findall(text)


def load_conversations():
    """Yield `chosen` conversations as raw text."""
    for name in ("hh_rlhf_helpful_train.jsonl.gz", "hh_rlhf_harmless_train.jsonl.gz"):
        p = os.path.join(CH, name)
        if not os.path.exists(p):
            print(f"  missing {name}", file=sys.stderr)
            continue
        n = 0
        with gzip.open(p, "rt") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("chosen") or ""
                if not t.strip():
                    continue
                yield t[:MAX_CHARS]
                n += 1
        print(f"  {name}: {n:,} conversations")


def build():
    """Return a dict with token ids, vocab, and the conversation boundaries."""
    cache_ids = os.path.join(CACHE, f"ids_v{VOCAB_CAP}.npy")
    cache_vocab = os.path.join(CACHE, f"vocab_v{VOCAB_CAP}.json")
    cache_meta = os.path.join(CACHE, f"meta_v{VOCAB_CAP}.json")
    cache_bounds = os.path.join(CACHE, f"bounds_v{VOCAB_CAP}.npy")
    cache_split = os.path.join(CACHE, f"split_v{VOCAB_CAP}.npy")
    if all(os.path.exists(p) for p in (cache_ids, cache_vocab, cache_meta,
                                       cache_bounds, cache_split)):
        # MEMMAP, read-only (2026-10-24). This was a plain np.load(), so EVERY process allocated its own
        # private 1.18 GB copy of the token array. One trainer was fine; three in parallel meant three private
        # copies plus three window arrays, which pushed the machine into swap and made the disk spike at
        # startup - measured: 2.3 GB of swap in use, RAM bouncing between 9.6 and 10.7 GB of 14.9 GB.
        # Mapped read-only the pages are SHARED through the page cache: one copy in RAM for any number of
        # parallel arms, and no per-arm burst of 1.18 GB of disk reads. Fancy indexing works on a memmap the
        # same way, so the windows built from it are unchanged - checked value by value below.
        ids = np.load(cache_ids, mmap_mode="r")
        vocab = json.load(open(cache_vocab))
        meta = json.load(open(cache_meta))
        # bounds and split MUST be cached too. Without them a cached load cannot build windows at
        # all -- the first version of this stored only ids/vocab/meta and the trainer died with
        # KeyError: 'bounds' after the 13M-token encode had already succeeded.
        bounds = np.load(cache_bounds).tolist()
        split = np.load(cache_split).tolist()
        print(f"  cached: {len(ids):,} tokens, vocab {len(vocab):,}, {len(bounds):,} conversations")
        return {"ids": ids, "vocab": vocab, "bounds": bounds, "split": split, **meta}

    convs = list(load_conversations())
    print(f"  {len(convs):,} conversations")

    # vocabulary from TRAIN text only -- a val token must not influence the vocab
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(convs))
    n_val = max(1, int(len(convs) * VAL_FRAC))
    val_idx = set(order[:n_val].tolist())

    from collections import Counter
    freq = Counter()
    for i, c in enumerate(convs):
        if i in val_idx:
            continue
        freq.update(tokenize(c.lower()))
    most = [w for w, _ in freq.most_common(VOCAB_CAP - 1)]
    vocab = ["<unk>"] + most
    vmap = {w: i for i, w in enumerate(vocab)}
    print(f"  vocab built from TRAIN text only: {len(vocab):,} words "
          f"({freq.most_common(5)[0][0]!r} most common)")

    # encode; a conversation is one document, separated by <unk>-free sentinel
    all_ids, bounds, split = [], [], []
    for i, c in enumerate(convs):
        toks = tokenize(c.lower())
        enc = [vmap.get(t, 0) for t in toks]
        if len(enc) < 8:
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
    print(f"  encoded {len(ids):,} tokens over {len(bounds):,} conversations")
    return {"ids": ids, "vocab": vocab, "bounds": bounds.tolist(), "split": split.tolist(), **meta}


def make_windows(ids, bounds, split, window, which, return_index=False, max_windows=None, seed=0):
    """Windows that do NOT cross a conversation boundary -- otherwise the model learns to continue
    someone else's sentence, which is a silent data-leak-style defect.

    MEMORY AND SPEED REWRITE (2026-10-19). The original built EVERY window as its own numpy array inside a
    Python loop: 16.9M windows of 32 int64 tokens, each a separate object, which is 7-8 GB of peak memory
    plus the 1.18 GB corpus. That is what OOM-killed the trainer repeatedly (syslog: "Memory cgroup out of
    memory ... glassnet-clean.scope", anon-rss 5.15 GB) and why startup sat on the CPU for minutes with the
    GPU idle. Two changes:
      * when `max_windows` is given, the windows are chosen by arithmetic (window counts per conversation)
        and only the chosen rows are materialised -- the full set is never built;
      * materialisation is vectorised per conversation with sliding_window_view, not one array per window.
    Default behaviour with max_windows=None is unchanged, so every existing caller and recorded number
    keeps its meaning.

    `return_index=True` also returns the absolute start position of every window, which the second (mtp)
    head needs.
    """
    idx = [j for j in range(len(bounds) - 1) if split[j] == which]
    stride = window // 2
    counts = np.asarray([max(0, len(range(0, int(bounds[j + 1] - bounds[j]) - window - 1, stride)))
                         for j in idx], dtype=np.int64)
    total = int(counts.sum())

    if max_windows is not None and total > max_windows:
        rng = np.random.default_rng(seed)
        picks = np.sort(rng.choice(total, size=int(max_windows), replace=False))
        cum = np.cumsum(counts)
        ci = np.searchsorted(cum, picks, side="right")
        prev = np.concatenate([[0], cum[:-1]])
        offs = (picks - prev[ci]) * stride
        conv_idx = np.asarray(idx, dtype=np.int64)[ci]
        chosen = (conv_idx, offs)
    else:
        # uncapped: every window, still without one array object per window
        conv_idx = np.repeat(np.asarray(idx, dtype=np.int64), counts)
        offs_list = [np.arange(0, int(c) * stride, stride, dtype=np.int64) for c in counts]
        offs = np.concatenate(offs_list) if offs_list else np.zeros(0, dtype=np.int64)

    ci_arr, off_arr = conv_idx, offs

    # FULLY VECTORISED MATERIALISATION. The first rewrite still looped per conversation (np.unique over
    # conv_idx), which on this corpus is 740,666 iterations and left startup CPU-bound for minutes with the
    # GPU idle. Windows never cross a conversation boundary because every offset comes from that
    # conversation's own arithmetic, so the absolute start of each window is bounds[conv] + offset and the
    # whole set can be gathered in one index operation.
    #
    # CHUNKED GATHER (2026-10-22). The single index operation above allocated the (N, window) index array AND
    # the result at the same time, in int64: at the full corpus (N = 16,878,132, window = 32) that is 4.32 GB
    # of indices + 4.32 GB of windows + the 1.18 GB token array = ~10 GB peak on a 14.75 GB machine whose
    # desktop holds ~3 GB. Two full-corpus runs died in exactly this call (d=256 at 15:45 with no epoch line;
    # d=512 at 15:48, exit 143). The gather now runs in bounded chunks: the index array can never exceed
    # MAX_GATHER_CELLS int64 cells (~64 MB). THE RETURNED VALUES, ORDER, SHAPE AND DTYPES ARE UNCHANGED - the
    # equivalence to the pre-change function is checked value-by-value in lens/scripts/memfix_check.py.
    MAX_GATHER_CELLS = 1 << 23
    # WIDTH FIX (2026-10-22 16:4x). The chunked gather above removed the TRANSIENT index array, but the
    # RESULT still has to be resident, and it was int64: at N = 16,878,132, window = 32 that is 4.32 GB for
    # X alone, and the full-corpus arm still OOM-killed inside the build at 16:13:56 (10G peak in
    # glassnet-q-2591701866.scope, swap 2.7G; lens/runs/TIED-FULLCORPUS.txt ends before its first epoch).
    # Every token id in this corpus is < 12,002, so int32 holds it EXACTLY (2^31 >> 12,002) and halves X to
    # 2.16 GB. The gather now reads `ids` directly at its own width instead of making an int32 copy of the
    # whole 294M-token corpus first (that copy was another 1.18 GB at the peak). VALUES ARE UNCHANGED; the
    # equivalence is checked value-by-value against the pre-change copy in lens/scripts/memfix2_check.py,
    # with the predictions written BEFORE the edit in lens/runs/MEMFIX2-PREDICTIONS.txt (P64a, P64e).
    WIN_DTYPE = np.int32
    ids_arr = np.asarray(ids)
    starts = np.asarray(bounds)[ci_arr].astype(np.int64) + off_arr
    if len(starts):
        X = np.empty((len(starts), window), dtype=WIN_DTYPE)
        col = np.arange(window, dtype=np.int64)[None, :]
        step = max(1, int(MAX_GATHER_CELLS // max(window, 1)))
        for s0 in range(0, len(starts), step):
            s1 = min(s0 + step, len(starts))
            X[s0:s1] = ids_arr[starts[s0:s1, None] + col]
        Y = ids_arr[starts + window].astype(WIN_DTYPE)
    else:
        X = np.zeros((0, window), dtype=WIN_DTYPE)
        Y = np.zeros(0, dtype=WIN_DTYPE)
    S = starts.astype(np.int64)
    if return_index:
        return X, Y, S
    return X, Y


if __name__ == "__main__":
    d = build()
    print(f"  total tokens: {len(d['ids']):,}   vocab: {len(d['vocab']):,}")
    v = d["vocab"]
    print("  first 40 tokens:", " ".join(v[i] for i in d["ids"][:40]))
    print("  vocab head:", v[1:16])
